"""Cycle advertised aux source pages; restore original layouts/pages on exit.

Run against an exclusively controlled live test mixer. This changes subscriptions
and the backend's saved aux state, but never edits sources or scene assignments.
Pair with capture.py for GPU/counter measurements; this script checks accepted
commands and applied compositions, not source pixel freshness.
"""
import argparse
import json
from pathlib import Path
import re
import time
import urllib.request


def run(args):
    report = {"ok": False, "started_at": time.time(), "commands": [], "errors": [], "checks": []}
    originals, touched, progress = {}, [], {}

    def request(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(args.url.rstrip("/") + path, data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=args.request_timeout) as response:
            result = json.load(response)
        if result.get("error"):
            raise RuntimeError(result["error"])
        return result

    def command(bus, **payload):
        payload["bus"] = bus
        result = request("/api/command", payload)
        report["commands"].append({"at": time.time(), "request": payload, "response": result})
        assert result.get("ok") and not result.get("superseded"), result

    def settle(expected):
        deadline = time.monotonic() + args.settle_timeout
        while True:
            state = request("/api/state")
            buses = {bus["id"]: bus for bus in state.get("aux_buses", [])}
            assert not state.get("status", {}).get("error"), state["status"]
            pending = []
            for name, layout in expected.items():
                bus = buses[name]
                assert not bus.get("error") and not bus.get("composition_error"), (name, bus)
                if not bus.get("running") or bus.get("composition_pending") or bus["layout"] != layout:
                    pending.append(name)
            report["checks"].append({"at": time.time(), "pending": pending})
            if not pending:
                return state, buses
            assert time.monotonic() < deadline, f"compositions did not settle: {pending}"
            time.sleep(.1)

    try:
        report["start"] = request("/api/state")
        for bus in report["start"].get("aux_buses", []):
            layouts = [bus["layout"], *bus.get("layouts", [])]
            if re.fullmatch(r"aux\d+", bus["id"]) and any(l.get("preset") == "source_pages" for l in layouts):
                originals[bus["id"]] = {"layout": bus["layout"], "scenes": bus["scenes"]}
        assert originals, "no extra aux buses advertise source_pages"
        expected = {}
        for name in originals:
            touched.append(name)  # A timed-out request may still have reached the backend.
            expected[name] = {"preset": "source_pages", "page": 0}
            command(name, command="aux_layout", layout=expected[name])
        _, buses = settle(expected)
        pages = {name: buses[name]["pages"] for name in originals}
        assert all(count > 1 for count in pages.values()), "every selected bus needs at least two pages"
        progress = dict.fromkeys(originals, 0)
        report["pages"] = pages
        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline:
            for index, name in enumerate(originals):
                # Stagger buses across pages so independent outputs retain different sources.
                page = (expected[name]["page"] + 1) % pages[name] if progress[name] else 1 + index % (pages[name] - 1)
                command(name, command="aux_page", page=page)
                expected[name] = {"preset": "source_pages", "page": page}
            _, buses = settle(expected)
            for name in originals:
                assert buses[name]["scenes"] == originals[name]["scenes"], f"scene assignments changed: {name}"
                progress[name] += 1
            time.sleep(min(args.interval, max(0, deadline - time.monotonic())))
        assert all(progress.values()), f"no applied page change on a selected bus: {progress}"
    except (Exception, KeyboardInterrupt) as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        restore = {}
        for name in touched:
            try:
                command(name, command="aux_layout", layout=originals[name]["layout"])
                restore[name] = originals[name]["layout"]
            except Exception as error:
                report["errors"].append(f"restore {name}: {error}")
        try:
            report["end"], buses = settle(restore)
            for name in touched:
                assert buses[name]["layout"] == originals[name]["layout"], f"layout not restored: {name}"
                assert buses[name]["scenes"] == originals[name]["scenes"], f"assignments not preserved: {name}"
            report["restored"] = bool(touched) and len(restore) == len(touched)
        except Exception as error:
            report["errors"].append(f"restoration verification: {error}")
        report.update(progress=progress, ended_at=time.time())
        report["ok"] = bool(progress) and all(progress.values()) and report.get("restored", False) and not report["errors"]
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--interval", type=float, default=1)
    parser.add_argument("--request-timeout", type=float, default=5)
    parser.add_argument("--settle-timeout", type=float, default=5)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if min(args.duration, args.interval, args.request_timeout, args.settle_timeout) <= 0:
        parser.error("duration, interval and timeouts must be positive")
    report = run(args)
    print(json.dumps({key: report.get(key) for key in ("ok", "progress", "restored", "errors")}))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
