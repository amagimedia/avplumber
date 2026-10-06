"""Shared mixer GUI with the synthetic demo setup policy."""
from .instance_profiles import InstanceType
from .setup_runtime import SetupRuntime
from pyplumber.mixer.gui import web
from pathlib import Path
import argparse


def configure(parser):
    parser.add_argument("--manage-setup", action="store_true", help="Own and restart the demo mixer process")
    instance_types = [t.value for t in InstanceType]
    parser.add_argument("--instance-type", choices=instance_types,
                        help="The host type whose measured source limits the setup applies (instance_profiles.py); "
                             "required with --manage-setup")
    parser.add_argument("--media-dir", type=Path, default=Path("/media"))
    parser.add_argument("--recipe", type=Path, default=Path("/media/demo.json"))
    parser.add_argument("--dmabuf-rest", default="http://127.0.0.1:9009")
    parser.add_argument("--janus-api", metavar="URL",
                        help="Janus HTTP API (e.g. http://127.0.0.1:8088/janus, no API secret) for the setup's "
                             "extra aux outputs, one Streaming mountpoint each; without it the setup offers none")
    parser.add_argument("--mixer-args", nargs="...", default=[])


def parse_args(argv=None):
    args = web.parse_args(argv, configure=configure)
    if args.critical_nice and not args.manage_setup:
        argparse.ArgumentParser().error("--critical-nice needs --manage-setup")
    if args.manage_setup and not args.instance_type:
        argparse.ArgumentParser().error("--manage-setup needs --instance-type: source limits are measured per host type")
    return args


def create_setup(args, bridge):
    if not args.manage_setup:
        return None
    if not args.instance_type:
        raise ValueError("--manage-setup needs --instance-type: source limits are measured per host type")
    return SetupRuntime(args.media_dir, args.recipe, bridge, args.instance_type, args.mixer_args,
                        args.dmabuf_rest, args.janus_api)


if __name__ == "__main__":
    web.main(configure=configure, setup_factory=create_setup)
