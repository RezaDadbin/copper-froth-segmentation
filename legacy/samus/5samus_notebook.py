"""Historical source export; maintained entry points live in scripts/.

Original: codes/SAMUS/5SAMUS.ipynb
Importing this file does not train, install packages, rename files or render.
historical_experiment() is a manual research transcript, not a supported pipeline.
"""


def historical_experiment():
    # %% Original cell 0 (code)
    pass  # IPython/setup disabled: !pip install  git+https://github.com/xianlin7/SAMUS

    # %% Original cell 1 (code)
    # =========================
    # Cell 1 — SAMUS Setup
    # =========================
    import os, sys, time, json, math
    from pathlib import Path

    # Ensure official SAM is available
    try:
        from segment_anything import sam_model_registry
    except Exception:
        pass  # IPython/setup disabled: %pip install -q git+https://github.com/facebookresearch/segment-anything.git
        from segment_anything import sam_model_registry

    import numpy as np
    import torch, torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import tifffile as tiff
    import cv2
    import matplotlib.pyplot as plt

    # ---- paths ----
    TRAIN_DIR = Path("../train1")   # your training folder
    TEST_DIR  = Path("../test1")    # your test folder

    MODEL_TYPE = "vit_b"                               # backbone we’ll use from SAM
    SAM_CHECKPOINT = "./weights/sam_vit_b_01ec64.pth"  # put your SAM ViT-B weights here

    OUT_DIR = Path("./outputs_samus")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- device ----
    use_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if use_mps else "cpu"))
    print("Device:", DEVICE)
    print("Train:", TRAIN_DIR.resolve())
    print("Test :", TEST_DIR.resolve())
    print("Out  :", OUT_DIR.resolve())
    print("SAM ckpt:", SAM_CHECKPOINT)

    # ---- resize helper: longest side → 1024 (SAM convention) ----
    def resize_longest_1024(img_hwc_uint8: np.ndarray) -> np.ndarray:
        h, w = img_hwc_uint8.shape[:2]
        if max(h, w) == 1024:
            return img_hwc_uint8
        scale = 1024.0 / float(max(h, w))
        nh, nw = int(round(h * scale)), int(round(w * scale))
        return cv2.resize(img_hwc_uint8, (nw, nh), interpolation=cv2.INTER_AREA)

    print("SAMUS notebook initialized.")

    # %% Original cell 2 (code)
    # =========================
    # Cell 1b — SAMUS checkpoint inspector
    # =========================
    from pathlib import Path
    import torch

    # <<< set this to your downloaded SAMUS checkpoint >>>
    SAMUS_CKPT = Path("./weights/SAMUS.pth")  # e.g. "./weights/SAMUS.pth"

    assert SAMUS_CKPT.exists(), f"Checkpoint not found: {SAMUS_CKPT}"

    def inspect_samus_ckpt(path: Path, max_keys: int = 40):
        obj = torch.load(str(path), map_location="cpu")
        print(f"Loaded: {path.name}")
        print(f"Type: {type(obj)}")

        def _print_dict(d, prefix=""):
            keys = list(d.keys())
            print(f"{prefix}dict keys ({len(keys)}): {keys[:min(len(keys), 20)]} {'...' if len(keys) > 20 else ''}")

        # Case A: raw state_dict (str->Tensor)
        if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()) and any(
            hasattr(v, "shape") for v in obj.values()
        ):
            print("Detected format: RAW state_dict")
            ks = list(obj.keys())[:max_keys]
            for k in ks:
                v = obj[k]
                if hasattr(v, "shape"):
                    print(f"  {k:60s}  {tuple(v.shape)}  {v.dtype}")
                else:
                    print(f"  {k:60s}  (non-tensor)")
            return {"format": "state_dict", "payload": obj}

        # Case B: wrapped dict (e.g., {'model': state_dict, ...} or {'state_dict': ...})
        if isinstance(obj, dict):
            _print_dict(obj, prefix="Top-level ")
            sd = None
            for candidate in ["model", "state_dict", "samus", "net", "module", "model_state"]:
                if candidate in obj and isinstance(obj[candidate], dict):
                    # must look like a state_dict
                    if any(hasattr(v, "shape") for v in obj[candidate].values()):
                        sd = obj[candidate]
                        print(f"Found nested state_dict under key: '{candidate}'")
                        break
            if sd is not None:
                ks = list(sd.keys())[:max_keys]
                for k in ks:
                    v = sd[k]
                    if hasattr(v, "shape"):
                        print(f"  {k:60s}  {tuple(v.shape)}  {v.dtype}")
                return {"format": "wrapped", "key": candidate, "payload": sd}
            else:
                print("Could not find a nested state_dict. This may be a custom checkpoint format.")
                return {"format": "unknown", "payload": obj}

        print("Unknown checkpoint object type.")
        return {"format": "unknown", "payload": obj}

    ckpt_info = inspect_samus_ckpt(SAMUS_CKPT)
    ckpt_format = ckpt_info.get("format")
    print(f"\n=== Detected checkpoint format: {ckpt_format} ===")

    # %% Original cell 3 (code)
    # Install SAMUS runtime dependencies that are missing on your env
    pass  # IPython/setup disabled: %pip install -q einops timm

    # %% Original cell 4 (code)
    # quick sanity checks
    import einops, timm
    print("Deps OK — einops:", einops.__version__, "| timm:", timm.__version__)

    # %% Original cell 5 (code)
    import os, sys, importlib
    from pathlib import Path
    import torch

    # paths
    SAMUS_ROOT = Path("./SAMUS_repo").resolve()           # where you cloned xianlin7/SAMUS
    CKPT_PATH  = Path("./weights/SAMUS.pth").resolve()    # your SAMUS checkpoint (raw state_dict)

    assert SAMUS_ROOT.exists(), f"Missing repo at {SAMUS_ROOT}"
    if str(SAMUS_ROOT) not in sys.path:
        sys.path.insert(0, str(SAMUS_ROOT))

    # 1) build args the repo way
    cfg_mod = importlib.import_module("utils.config")     # repo file
    args = cfg_mod.get_config()                           # defaults
    # Make sure we pick the SAMUS model variant:
    args.modelname = "SAMUS"
    # Optional tweaks you may want:
    args.num_classes = getattr(args, "num_classes", 1)
    args.in_chans    = getattr(args, "in_chans", 3)

    print("[INFO] args.modelname:", args.modelname)

    # 2) import the factory exactly like the repo's train.py
    model_dict = importlib.import_module("models.model_dict")
    get_model  = getattr(model_dict, "get_model")
    samus_model = get_model(args)                         # NOTE: pass single args, no kwargs
    print("[OK] Built SAMUS model")

    # 3) load your raw state_dict (strict=False to allow minor key drift)
    assert CKPT_PATH.exists(), f"Checkpoint not found: {CKPT_PATH}"
    state = torch.load(str(CKPT_PATH), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    missing, unexpected = samus_model.load_state_dict(state, strict=False)
    print(f"[LOAD] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:   print("  missing (first 15):", missing[:15])
    if unexpected:print("  unexpected (first 15):", unexpected[:15])

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    samus_model = samus_model.to(DEVICE).eval()
    print(f"[READY] SAMUS on {DEVICE}")

    # %% Original cell 6 (code)

    pass


if __name__ == "__main__":
    raise SystemExit("Historical transcript only. Use the linked scripts/ commands in README.md.")
