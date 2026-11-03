"""Add, remove and swap sources on a running mixer and report what it cost the outputs.

Each source is a slot. Removing one stops its group and tells every compositor to let go of its
frames, which is what ends its decoder; without that the last picture stays held and the teardown
only happens at the next add. Starting the group adds it back, and changing its input `url` first
swaps what it plays. Point --inputs at sources on air, or the compositors hold nothing to let go of.

--keep-decoder stops only the input node instead of the group, so the decoder stays open and idle:
a new stream of the same format needs no decoder, surfaces or textures to be created or destroyed.

--shrink removes every input before adding any back, which is what a show getting smaller does:
past the park limit, decoders are freed. Use more inputs than `decoder_park_limit`.

    python3 tests/mixer/source_churn.py --inputs 0,1,2,3 --swap <file> --output churn.json
"""
import argparse
import json
import socket
import time
import urllib.request


class Control:
    """The line protocol: one reply line, and for 201 a body up to an empty line."""

    def __init__(self, host, port):
        self.sock = socket.create_connection((host, port), timeout=30)
        self.file = self.sock.makefile("r", encoding="utf-8", newline="\n")
        self.file.readline()   # the server's greeting

    def command(self, line):
        self.sock.sendall(line.encode() + b"\n")
        reply = self.file.readline().rstrip("\n")
        code = int(reply.split(" ", 1)[0])
        if code == 201:
            while self.file.readline().rstrip("\n"):
                pass
        if not 200 <= code < 300:
            raise RuntimeError(f"{line}: {reply}")


def state_of(api):
    with urllib.request.urlopen(api + "/api/state", timeout=20) as reply:
        return json.load(reply)


def compositors(api):
    return ["mixer_comp_a", "mixer_comp_b", *(f"aux_{bus['id']}_comp" for bus in state_of(api)["aux_buses"])]


def counters(api):
    """Sums every output's counters: the same totals the operator page reads."""
    state = state_of(api)
    missed = {f"slot_{k}": v.get("missed_deadlines", 0) for k, v in state["status"]["playout"].items() if isinstance(v, dict)}
    drops = {}
    for bus in state.get("aux_buses", []):
        if isinstance(bus.get("playout"), dict):
            missed[bus["id"]] = bus["playout"].get("missed_deadlines", 0)
        drops[bus["id"]] = bus.get("output_drops", 0)
    return {"missed_deadlines": sum(missed.values()), "outputs": len(missed),
            "output_drops": sum(drops.values()), "missed_by_output": missed, "drops_by_output": drops}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--control", default="127.0.0.1:7777")
    parser.add_argument("--api", default="http://127.0.0.1:17681")
    parser.add_argument("--inputs", required=True, help="comma-separated source indexes")
    parser.add_argument("--swap", help="file a source swaps to and from; without it a swap restarts the same file")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--together", action="store_true", help="change every input at once, not one by one")
    parser.add_argument("--hold", type=float, default=2.0, help="seconds a removed source stays away")
    parser.add_argument("--settle", type=float, default=4.0, help="seconds after each change")
    parser.add_argument("--drain", type=float, default=0.25, help="seconds for frames on their way to arrive")
    parser.add_argument("--baseline", type=float, default=10.0, help="seconds of no changes first")
    parser.add_argument("--keep-decoder", action="store_true", help="stop the input node only, not the group")
    parser.add_argument("--shrink", action="store_true", help="remove every input, then add them all back")
    parser.add_argument("--output")
    args = parser.parse_args()

    host, port = args.control.rsplit(":", 1)
    control = Control(host, int(port))
    inputs = [int(i) for i in args.inputs.split(",")]
    batches = [inputs] if args.together else [[i] for i in inputs]
    log = []

    outputs = compositors(args.api)

    def step(name, batch, *commands, wait, then=()):
        before = counters(args.api)
        started = time.time()
        for command in commands:
            control.command(command)
        if then:
            time.sleep(args.drain)
            for command in then:
                control.command(command)
        issued = time.time()
        time.sleep(wait)
        after = counters(args.api)
        hit = lambda key: {k: v - before[key].get(k, 0) for k, v in after[key].items() if v != before[key].get(k, 0)}
        log.append({"op": name, "inputs": batch, "wall": started, "command_s": round(issued - started, 3),
                    "missed": after["missed_deadlines"] - before["missed_deadlines"],
                    "output_drops": after["output_drops"] - before["output_drops"],
                    "outputs_missing": hit("missed_by_output"), "outputs_dropping": hit("drops_by_output")})
        print(json.dumps(log[-1]), flush=True)

    idle = counters(args.api)
    time.sleep(args.baseline)
    now = counters(args.api)
    baseline = {"seconds": args.baseline, "missed": now["missed_deadlines"] - idle["missed_deadlines"],
                "output_drops": now["output_drops"] - idle["output_drops"], "outputs": now["outputs"]}
    print("baseline", json.dumps(baseline), flush=True)
    stop, start = ("node.stop_wait", "node.start") if args.keep_decoder else ("group.stop", "group.start")

    def remove(batch):
        step("remove", batch, *[f"{stop} input_{i}" for i in batch], wait=args.hold,
             then=[f"node.object.set {node} release_input {i}" for i in batch for node in outputs])

    def add(batch):
        step("add", batch, *[f"{start} input_{i}" for i in batch], wait=args.settle)

    for _ in range(args.cycles):
        if args.shrink:
            for batch in batches:
                remove(batch)
            for batch in batches:
                add(batch)
            continue
        for batch in batches:
            remove(batch)
            add(batch)
            swap = [f"node.param.set input_{i} url {json.dumps(args.swap)}" for i in batch] if args.swap else []
            restart = ([f"{c} input_{i}" for i in batch for c in (stop, start)] if args.keep_decoder
                       else [f"group.restart input_{i}" for i in batch])
            step("swap", batch, *swap, *restart, wait=args.settle)
    result = {"baseline": baseline, "ops": log, "decoders": state_of(args.api)["status"].get("decoders"),
              "total": {key: sum(entry[key] for entry in log) for key in ("missed", "output_drops")}}
    print("total", json.dumps(result["total"]), flush=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(result, file, indent=1)


if __name__ == "__main__":
    main()
