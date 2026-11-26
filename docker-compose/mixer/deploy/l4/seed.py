"""Seed an empty media directory; later starts preserve the operator's saved show."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from demos.mixer.demo_recipe import write_atomic
from pyplumber.mixer.tools.extra_aux import extra_buses
from demos.mixer import gpu_spec
from demos.mixer.instance_profiles import INSTANCE_PROFILES, InstanceType, geforce_profile
from demos.mixer.prepare_demo import CLEAN_PORT, plan
from pyplumber.mixer.config import parse
from demos.mixer.setup_runtime import desktop_preset, desktop_suggested, extra_aux_limit, recipe_for


def seed(settings_path, media_dir, instance_type=InstanceType.NVIDIA_L4_CUARRAY):
    media_dir = Path(media_dir)
    media_dir.mkdir(parents=True, exist_ok=True)
    recipe_path, show_path = media_dir / "demo.json", media_dir / "mixer.demo.json"
    if recipe_path.exists() or show_path.exists():
        print("Keeping the saved mixer setup.")
        return
    # GeForce and auto read the GPU from MIXER_GPU (stack.sh): this container has none.
    card = gpu_spec.probe() if instance_type in ("auto", InstanceType.GEFORCE) else None
    instance_type = InstanceType(gpu_spec.instance_type(card) if instance_type == "auto" else instance_type)
    profile = geforce_profile(card) if instance_type == InstanceType.GEFORCE else INSTANCE_PROFILES[instance_type]
    if instance_type == InstanceType.GEFORCE and settings_path in ("fit", "low"):
        # fit: the largest desktop size this GPU carries; low: the 96-input one.
        initial = desktop_preset(desktop_suggested(profile) if settings_path == "fit" else "low", profile)
    else:
        initial = json.loads(Path(settings_path).read_text())
    recipe = recipe_for(profile, initial)
    settings = recipe["setup"]
    encodes = settings["encodes"]
    aux_default = profile["nvenc"]["defaults"]["aux"]
    recipe["aux_buses"] = [
        {"id": "mv", "label": "Program preview", "layout": {"preset": "pgm_pvw_grid"},
         "renditions": [{"id": "monitor", "port": 5008, **encodes.get("mv", aux_default)}]},
        {"id": "mv2", "label": "Multiviewer", "layout": {"preset": "source_pages"},
         "renditions": [{"id": "monitor", "port": 5012, **encodes.get("mv2", aux_default)}]},
    ]
    cfg = parse(plan(recipe, media_dir)[0])
    settings["extra_aux"] = min(settings["extra_aux"], extra_aux_limit(profile, cfg, encodes))
    recipe["aux_buses"] += extra_buses(cfg, [], settings["extra_aux"], CLEAN_PORT, encodes["extra"])
    show = plan(recipe, media_dir)[0]
    write_atomic(recipe_path, json.dumps(recipe, indent=2) + "\n")
    print(f"Seeded {len(show['sources'])} sources and {len(show['scenes'])} scenes; assets generate at startup.")


if __name__ == "__main__":
    seed(*sys.argv[1:])
