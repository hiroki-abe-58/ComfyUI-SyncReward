"""Minimal reproduction: how many labels does transformers give the timm vision tower inside facebook/pe-av-large's config?

    python scripts/repro_timm_labels.py <folder containing the model's config.json>

Only config.json is read (about 70 kB, no weights, no GPU). The file states `num_classes: 1024` for the nested `timm_wrapper`
(the width of the vision projection). The script prints, for the installed transformers version, what the config file says and what
`AutoConfig` parsed for the nested timm vision tower. Observed (docs/VERIFICATION.md section 4): transformers 5.1.0 -> 1024; 5.20.1 -> 2.

No file is needed to see the same thing in 5.20.1: `PeVideoEncoderConfig().vision_config.num_labels` is 2 there and 1024 in 5.1.0, because
the nested config is built with `TimmWrapperConfig(**dict)` instead of `TimmWrapperConfig.from_dict(dict)`, which skips its
`num_classes` -> `num_labels` conversion.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def raw_timm_entries(node, path="config"):
    if isinstance(node, dict):
        if node.get("model_type") == "timm_wrapper":
            yield path, {k: node[k] for k in ("num_classes", "num_labels", "architecture", "id2label") if k in node} | {"keys": len(node)}
        for key, value in node.items():
            yield from raw_timm_entries(value, f"{path}.{key}")


def parsed_timm_entries(cfg, path="config", seen=None):
    seen = seen if seen is not None else set()
    for key, value in vars(cfg).items():
        if hasattr(value, "model_type") and hasattr(value, "to_dict") and id(value) not in seen:
            seen.add(id(value))
            if value.model_type == "timm_wrapper":
                yield f"{path}.{key}", value.num_labels, getattr(value, "architecture", None)
            yield from parsed_timm_entries(value, f"{path}.{key}", seen)


def main() -> int:
    folder = Path(sys.argv[1])
    import transformers
    from transformers import AutoConfig

    print("transformers", transformers.__version__)
    raw = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    for path, info in raw_timm_entries(raw):
        print("config.json   ", path, info)
    config = AutoConfig.from_pretrained(str(folder))
    for path, labels, arch in parsed_timm_entries(config):
        print("AutoConfig    ", path, "num_labels =", labels, "architecture =", arch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
