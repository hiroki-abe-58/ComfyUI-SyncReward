"""Model locations: `<ComfyUI models>/syncreward/<name>/` (plus any path an administrator adds for `syncreward` in
extra_model_paths.yaml).

Two kinds of folders are listed:
* a SyncReward checkpoint: `config.json` with `"model_type": "syncreward"` and `model.safetensors` (the official
  `syncreward-stage2` folder of Guan123/SyncReward);
* the PE-AV encoder: the Hugging Face folder of facebook/pe-av-large (`config.json` with `"model_type": "pe_audio_video"`,
  `model.safetensors`, `preprocessor_config.json`, `processor_config.json`, ...).
Workflows choose names from these lists only; nothing is downloaded at run time."""

from __future__ import annotations

import json
import os

FOLDER = "syncreward"


def register() -> None:
    import folder_paths

    default = os.path.join(folder_paths.models_dir, FOLDER)
    folder_paths.add_model_folder_path(FOLDER, default, is_default=False)


def _model_type(directory: str):
    try:
        with open(os.path.join(directory, "config.json"), encoding="utf-8") as handle:
            return json.load(handle).get("model_type")
    except (OSError, ValueError):
        return None


def _scan(kind: str, extra_files: tuple[str, ...]) -> list[str]:
    import folder_paths

    found = set()
    for base in folder_paths.get_folder_paths(FOLDER):
        if not os.path.isdir(base):
            continue
        for entry in os.scandir(base):
            if (
                entry.is_dir()
                and _model_type(entry.path) == kind
                and all(os.path.isfile(os.path.join(entry.path, name)) for name in ("model.safetensors", *extra_files))
            ):
                found.add(entry.name)
    return sorted(found)


def checkpoints() -> list[str]:
    return _scan("syncreward", ())


def encoders() -> list[str]:
    return _scan("pe_audio_video", ("preprocessor_config.json", "processor_config.json"))


def _resolve(name: str, allowed: list[str], what: str) -> str:
    import folder_paths

    if name not in allowed:
        raise FileNotFoundError(f"{what} folder {name!r} was not found in models/{FOLDER} (see the README for the download command)")
    for base in folder_paths.get_folder_paths(FOLDER):
        candidate = os.path.realpath(os.path.join(base, name))
        if os.path.dirname(candidate) == os.path.realpath(base) and os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(name)


def checkpoint_path(name: str) -> str:
    return _resolve(name, checkpoints(), "SyncReward checkpoint")


def encoder_path(name: str) -> str:
    return _resolve(name, encoders(), "PE-AV encoder")
