"""Seed an empty media directory; later starts preserve the operator's saved show."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from demo_recipe import write_atomic
from extra_aux import extra_buses
from instance_profiles import INSTANCE_PROFILES, InstanceType
from prepare_demo import CLEAN_PORT, plan
from pyplumber.mixer.config import parse
from setup_runtime import extra_aux_limit, recipe_for


def seed(settings_path, media_dir):
    media_dir = Path(media_dir)
    media_dir.mkdir(parents=True, exist_ok=True)
    recipe_path, show_path = media_dir / "demo.json", media_dir / "mixer.demo.json"
    if recipe_path.exists() or show_path.exists():
        print("Keeping the saved mixer setup.")
        return
    profile = INSTANCE_PROFILES[InstanceType.NVIDIA_L4_CUARRAY]
    recipe = recipe_for(profile, json.loads(Path(settings_path).read_text()))
    settings = recipe["setup"]
    encodes = settings["encodes"]
    recipe["aux_buses"] = [
        {"id": "mv", "label": "Program preview", "layout": {"preset": "pgm_pvw_grid"},
         "renditions": [{"id": "monitor", "port": 5008, **encodes["mv"]}]},
        {"id": "mv2", "label": "Multiviewer", "layout": {"preset": "source_pages"},
         "renditions": [{"id": "monitor", "port": 5012, **encodes["mv2"]}]},
    ]
    cfg = parse(plan(recipe, media_dir)[0])
    settings["extra_aux"] = min(settings["extra_aux"], extra_aux_limit(profile, cfg, encodes))
    recipe["aux_buses"] += extra_buses(cfg, [], settings["extra_aux"], CLEAN_PORT, encodes["extra"])
    show = plan(recipe, media_dir)[0]
    write_atomic(recipe_path, json.dumps(recipe, indent=2) + "\n")
    print(f"Seeded {len(show['sources'])} sources and {len(show['scenes'])} scenes; assets generate at startup.")


if __name__ == "__main__":
    seed(*sys.argv[1:])
