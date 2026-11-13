"""Build the Compose images once and copy their exact digests to regional registries."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess


STACK = Path(__file__).resolve().with_name("stack.sh")


def run(*args, **kwargs):
    return subprocess.check_output(args, text=True, **kwargs).strip()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def prefix(value):
    if not re.fullmatch(r"[a-z0-9.-]+-docker\.pkg\.dev/[a-z0-9-]+/[a-z0-9._-]+/[a-z0-9._-]+", value):
        raise ValueError("Expected a GAR image prefix: <region>-docker.pkg.dev/<project>/<repository>/<name>")
    return value


def prepare(output, image_prefix, revision):
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("revision must be a full Git commit SHA")
    env = dict(os.environ, MIXER_IMAGE_PREFIX=prefix(image_prefix), MIXER_RELEASE=revision,
               AVPLUMBER_REVISION=revision)
    compose = json.loads(run(str(STACK), "config", "--format", "json", env=env))
    targets = {}
    images = {}
    for service, config in compose["services"].items():
        if "build" not in config:
            continue
        image = config["image"]
        if not image.startswith(image_prefix + "-") or not image.endswith(":" + revision):
            raise ValueError(f"Unexpected image for {service}: {image}")
        images[service] = image
        cache = image.rsplit(":", 1)[0] + ":buildcache"
        targets[service] = {
            "platforms": ["linux/amd64"],
            "labels": {"org.opencontainers.image.revision": revision},
            "cache-from": [f"type=registry,ref={cache}"],
            "cache-to": [f"type=registry,ref={cache},mode=max,image-manifest=true,oci-mediatypes=true"],
        }
    if "mixer" not in images:
        raise ValueError("Compose has no mixer build")
    output.mkdir(parents=True, exist_ok=True)
    # JSON is also YAML; the suffix tells Bake to interpret it as Compose.
    write_json(output / "build.compose.yaml", compose)
    write_json(output / "cache.json", {"target": targets})
    write_json(output / "build.json", {"revision": revision, "prefix": image_prefix,
                                      "images": images, "proxy": compose["services"]["proxy"]["image"]})


def copy_image(source, target, digest):
    subprocess.run(["crane", "copy", source, target], check=True)
    actual = run("crane", "digest", target)
    if actual != digest:
        raise ValueError(f"{target}: expected {digest}, got {actual}")


def release(output, mirrors):
    build = json.loads((output / "build.json").read_text())
    metadata = json.loads((output / "metadata.json").read_text())
    images = {}
    for service, image in build["images"].items():
        digest = metadata[service]["containerimage.digest"]
        if run("crane", "digest", image) != digest:
            raise ValueError(f"Published tag changed during build: {image}")
        images[service] = {"image": image, "digest": digest}
    proxy = build["proxy"]
    digest = run("crane", "digest", proxy)
    proxy_target = build["prefix"] + "-proxy:" + build["revision"]
    copy_image(proxy.split("@", 1)[0] + "@" + digest, proxy_target, digest)
    images["proxy"] = {"image": proxy_target, "digest": digest}
    destinations = dict.fromkeys([build["prefix"], *mirrors])
    receipt = {"revision": build["revision"], "gpu_playback_verified": False, "registries": {}}
    for destination in destinations:
        destination = prefix(destination)
        pins = {}
        for service, image in images.items():
            repository = image["image"].rsplit(":", 1)[0]
            target_repository = destination + repository.removeprefix(build["prefix"])
            if destination != build["prefix"]:
                copy_image(repository + "@" + image["digest"],
                           target_repository + ":" + build["revision"], image["digest"])
            pins[service] = {"image": target_repository + "@" + image["digest"]}
        pins["seed"] = dict(pins["mixer"])
        name = destination.replace("/", "--")
        write_json(output / (name + ".compose.yaml"), {"services": pins})
        receipt["registries"][destination] = pins
    write_json(output / "release.json", receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "release"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mirrors = [prefix(value) for value in os.environ.get("MIXER_GAR_MIRRORS", "").split()]
    if args.command == "prepare":
        prepare(args.output, os.environ["MIXER_IMAGE_PREFIX"], os.environ["GITHUB_SHA"])
    else:
        release(args.output, mirrors)


if __name__ == "__main__":
    main()
