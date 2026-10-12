"""Historical source export; maintained entry points live in scripts/.

Original: codes/ROBUSTSAM/4ROBUSTSAM.ipynb
Importing this file does not train, install packages, rename files or render.
historical_experiment() is a manual research transcript, not a supported pipeline.
"""


def historical_experiment():
    # %% Original cell 0 (code)
    # === RobustSAM: Cell 1 — env + imports + paths ===
    import os, sys, subprocess, json, math
    from pathlib import Path

    # ---- (A) Minimal deps (safe if already installed) ----
    def pip_install(pkgs):
        import sys, subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

    # Core libs we use across cells
    # to_install = [
    #     "tifffile",
    #     "opencv-python",      # GUI-less is fine too: opencv-python-headless
    #     "tqdm",
    #     "matplotlib",
    #     "numpy",
    #     "torchmetrics>=1.4.0",
    #     "git+https://github.com/facebookresearch/segment-anything.git"
    # ]
    # pip_install(to_install)

    # ---- (B) Torch / CUDA check ----
    import torch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print("CUDA version (PyTorch):", torch.version.cuda)
        print("cuDNN enabled:", torch.backends.cudnn.enabled)
        try:
            import torch_directml  # unlikely on your box; ignore if missing
        except Exception:
            pass
        # Optional: print GPU name
        try:
            print("GPU:", torch.cuda.get_device_name(0))
        except Exception:
            pass

    # ---- (C) SAM constants we keep consistent across notebooks ----
    IMAGE_LONGEST = 1024   # ResizeLongestSide target used by SAM-style models

    # ---- (D) Your dataset layout (adjust paths if needed) ----
    # Expecting:
    # DATA_ROOT/
    #   train1/  (tiff + json from Labelme)
    #   test1/   (tiff + json from Labelme)
    DATA_ROOT       = Path(r"..")      # <-- change if your dataset lives elsewhere
    TRAIN_DIR       = DATA_ROOT / "train1"
    TEST_DIR        = DATA_ROOT / "test1"
    assert TRAIN_DIR.exists(), f"Missing: {TRAIN_DIR}"
    assert TEST_DIR.exists(),  f"Missing: {TEST_DIR}"

    # ---- (E) RobustSAM repo + weights locations ----
    # We’ll clone the official repo next cell into this folder:
    ROBUSTSAM_DIR   = Path(r"F:/dataset/froth_finetune/RobustSam")  # choose any writable folder
    ROBUSTSAM_DIR.parent.mkdir(parents=True, exist_ok=True)

    # Place your RobustSAM checkpoint (ViT-B/L/H) here (download link is in their README)
    # e.g., robustsam_checkpoint_b.pth or robustsam_checkpoint_l.pth
    ROBUSTSAM_CKPT  = DATA_ROOT / "froth_finetune/weights/robustsam_checkpoint_b.pth"   # <-- rename to match your file
    print("Expecting RobustSAM ckpt at:", ROBUSTSAM_CKPT)

    # ---- (F) Basic plotting setup (optional) ----
    import matplotlib.pyplot as plt
    plt.rcParams["figure.dpi"] = 110

    print("Cell 1 ready ✅")

    # %% Original cell 1 (code)
    # === RobustSAM: Cell 2 — clone/import, load weights, decoder-only setup ===
    import sys, subprocess, importlib, inspect, torch
    from torch import nn
    from pathlib import Path
    from segment_anything import sam_model_registry

    # 1) Clone RobustSAM repo if not present
    def git_clone_if_needed(repo_url: str, target_dir: Path):
        target_dir = Path(target_dir)
        if target_dir.exists() and any(target_dir.iterdir()):
            print(f"[INFO] RobustSAM repo folder exists: {target_dir}")
            return
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Cloning RobustSAM repo into {target_dir} ...")
        try:
            subprocess.check_call(["git", "clone", "--depth", "1", repo_url, str(target_dir)])
        except Exception as e:
            print(f"[WARN] git clone failed: {e}. We'll proceed with fallback loader.")

    # Try a few likely repo URLs (first that works wins)
    ROBUSTSAM_REPO_URLS = [
        "https://github.com/RobustNets/RobustSAM.git",
        "https://github.com/robust-research/RobustSAM.git",
    ]
    if not ROBUSTSAM_DIR.exists() or not any(ROBUSTSAM_DIR.iterdir()):
        for url in ROBUSTSAM_REPO_URLS:
            git_clone_if_needed(url, ROBUSTSAM_DIR)
            if ROBUSTSAM_DIR.exists() and any(ROBUSTSAM_DIR.iterdir()):
                break

    # 2) Put repo on sys.path (if present)
    if ROBUSTSAM_DIR.exists():
        if str(ROBUSTSAM_DIR) not in sys.path:
            sys.path.insert(0, str(ROBUSTSAM_DIR))

    # 3) Best-effort: locate a model builder in the repo
    def find_robustsam_builder():
        candidates = [
            "robust_sam", "robustsam", "RobustSAM",
            "robust_sam.models", "RobustSAM.models", "models" , "robust_segment_anything",
        ]
        attrs = ["get_model", "build_model", "create_model", "Model", "RobustSAM", "RobustSam" , "build_sam"]
        for mod in candidates:
            try:
                m = importlib.import_module(mod)
            except Exception:
                continue
            for a in attrs:
                if hasattr(m, a):
                    fn = getattr(m, a)
                    try:
                        sig = inspect.signature(fn)
                    except Exception:
                        sig = None
                    return fn, sig, m.__name__
        return None, None, None

    builder_fn, builder_sig, builder_modname = find_robustsam_builder()
    if builder_fn:
        print(f"[INFO] Found RobustSAM builder: {builder_modname}.{builder_fn.__name__}")
    else:
        print("[WARN] No RobustSAM builder found in repo. Will fall back to vanilla SAM + non-strict state_dict load.")

    # 4) Load RobustSAM (or fallback to SAM) from checkpoint
    assert ROBUSTSAM_CKPT.exists(), f"RobustSAM checkpoint not found at: {ROBUSTSAM_CKPT}"
    MODEL_TYPE = "vit_b"  # adjust if your ckpt is vit_l / vit_h

    def load_state_dict_flexible(sd_path: Path):
        sd = torch.load(str(sd_path), map_location="cpu")
        # Common wrappers
        if isinstance(sd, dict):
            for key in ["state_dict", "model", "module", "ema_state_dict"]:
                if key in sd and isinstance(sd[key], dict):
                    return sd[key]
        return sd  # assume raw state_dict

    def instantiate_model():
        # Try repo builder first
        if builder_fn:
            try:
                # Try calling with a simple config; many repos accept a dict or simple args.
                # We keep it minimal to avoid mismatches; the intent is to get a model object,
                # we’ll load your checkpoint weights next.
                try:
                    model = builder_fn()
                except TypeError:
                    # try with a common 'backbone' arg
                    model = builder_fn(backbone=MODEL_TYPE)
                return model
            except Exception as e:
                print(f"[WARN] Builder call failed: {e}. Falling back to SAM registry.")
        # Fallback: standard SAM ViT
        return sam_model_registry[MODEL_TYPE]()

    # Create model & load weights (non-strict to accept partial matching keys)
    robustsam = instantiate_model()
    state = load_state_dict_flexible(ROBUSTSAM_CKPT)
    missing, unexpected = robustsam.load_state_dict(state, strict=False)
    if missing:
        print(f"[LOAD] Missing keys: {len(missing)} (ok for non-strict)")
    if unexpected:
        print(f"[LOAD] Unexpected keys: {len(unexpected)} (ok for non-strict)")

    robustsam.to(DEVICE)

    # 5) Configure **decoder-only** fine-tuning
    def set_decoder_only(model: nn.Module):
        # freeze all
        for p in model.parameters():
            p.requires_grad = False
        # unfreeze mask decoder (same naming as SAM; if different, adjust here)
        for n, p in model.named_parameters():
            if n.startswith("mask_decoder."):
                p.requires_grad = True
        # If an IoU head exists under decoder, also unfreeze it
        if hasattr(model, "mask_decoder") and hasattr(model.mask_decoder, "iou_prediction_head"):
            for p in model.mask_decoder.iou_prediction_head.parameters():
                p.requires_grad = True
        model.train()
        return model

    robustsam = set_decoder_only(robustsam)

    # 6) Optimizer for the decoder head
    LR = 1e-4
    WEIGHT_DECAY = 0.01
    optimizer = torch.optim.AdamW([p for p in robustsam.parameters() if p.requires_grad],
                                  lr=LR, weight_decay=WEIGHT_DECAY)

    # 7) Print a concise summary
    total_params     = sum(p.numel() for p in robustsam.parameters())
    trainable_params = sum(p.numel() for p in robustsam.parameters() if p.requires_grad)
    print(f"Device: {DEVICE}")
    print(f"RobustSAM checkpoint: {ROBUSTSAM_CKPT.name}")
    print(f"Total params: {total_params/1e6:.2f}M | Trainable (decoder-only): {trainable_params/1e6:.2f}M")
    print("First trainable tensors (should be under 'mask_decoder.*'):")
    shown = 0
    for n, p in robustsam.named_parameters():
        if p.requires_grad:
            print("  ", n, tuple(p.shape))
            shown += 1
            if shown >= 12:
                break

    print("Cell 2 ready ✅  (model loaded, decoder-only set, optimizer built)")

    # %% Original cell 2 (code)

    pass


if __name__ == "__main__":
    raise SystemExit("Historical transcript only. Use the linked scripts/ commands in README.md.")
