"""Command-line entry points. Argument help works without ML dependencies."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config, with_overrides


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def dataset_arguments(parser):
    parser.add_argument("--dataset-root", type=Path, required=True, help="Folder containing train1/ and test1/")
    parser.add_argument("--manifest", type=Path, required=True, help="Combined tags.csv manifest")


def model_arguments(parser):
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, help="Override SAM-family base checkpoint in JSON")
    parser.add_argument("--external-root", type=Path, help="Override SBSNet external implementation folder")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))


def configured(args):
    config = load_config(args.config)
    for key in ("base_checkpoint", "external_root"):
        value = getattr(args, key, None)
        if value is not None:
            config["model"][key] = str(value.expanduser().resolve())
    return config


def train_main(argv=None):
    parser = argparse.ArgumentParser(description="Train a configured model; save checkpoints, history and curves.")
    model_arguments(parser); dataset_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=positive)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--limit-train", type=positive, help="Explicit smoke-test subset size")
    parser.add_argument("--limit-val", type=positive)
    parser.add_argument("--no-pretrained", action="store_true", help="Skip ImageNet/COCO initialization downloads")
    args = parser.parse_args(argv)
    config = with_overrides(configured(args), epochs=args.epochs, seed=args.seed)
    from .engine.training import train
    train(config, args.dataset_root, args.manifest, args.output, args.device,
          args.limit_train, args.limit_val, False if args.no_pretrained else None)


def predict_main(argv=None):
    parser = argparse.ArgumentParser(description="Predict numeric binary masks and score arrays for linked evaluation/rendering.")
    model_arguments(parser); dataset_arguments(parser)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--use-base-weights", action="store_true", help="Explicit SAM-family base-model inference")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test", "all"), default="test")
    parser.add_argument("--limit", type=positive)
    parser.add_argument("--mode", choices=("semantic", "amg"), default="semantic")
    parser.add_argument("--prompt-source", choices=("full_image", "ground_truth"), help="Required for prompted SAM inference; ground_truth is assisted evaluation")
    args = parser.parse_args(argv)
    from .engine.prediction import predict
    predict(configured(args), args.dataset_root, args.manifest, args.output, args.checkpoint,
            split=args.split, device=args.device, limit=args.limit, mode=args.mode,
            prompt_source=args.prompt_source, use_base_weights=args.use_base_weights)


def evaluate_main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate saved masks on their recorded grids and save per-image/summary CSVs.")
    dataset_arguments(parser)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-distances", action="store_true", help="Skip HD and HD95 computation")
    args = parser.parse_args(argv)
    from .engine.evaluation import evaluate
    evaluate(args.predictions, args.dataset_root, args.manifest, args.output, distances=not args.skip_distances)


def render_main(argv=None):
    parser = argparse.ArgumentParser(description="Render model panels and native-grid comparisons as 300-dpi PNG and PDF.")
    dataset_arguments(parser)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=PATH", help="Repeat for model comparisons")
    parser.add_argument("--images", nargs="+", help="Exact manifest image names; quote names containing spaces")
    parser.add_argument("--limit", type=positive, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    runs = {}
    for item in args.run:
        if "=" not in item:
            parser.error("--run must be LABEL=PATH")
        label, path = item.split("=", 1)
        if not label or label in runs:
            parser.error("Each run needs a unique, nonempty label")
        runs[label] = Path(path)
    from .visualization.segmentation import render_results
    render_results(runs, args.dataset_root, args.manifest, args.output, image_names=args.images, limit=args.limit)


def curves_main(argv=None):
    parser = argparse.ArgumentParser(description="Render curves from a saved training history; no model run.")
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output stem without extension")
    args = parser.parse_args(argv)
    from .visualization.training import render_history
    render_history(args.history, args.output)


def postprocess_main(argv=None):
    parser = argparse.ArgumentParser(description="Run ROI watershed or LoG/Voronoi on numeric prediction scores.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("watershed", "log_voronoi"), default="watershed")
    parser.add_argument("--reference-run", type=Path, help="Training prediction run for LoG scale priors")
    parser.add_argument("--k-factor", type=float, default=0.736)
    parser.add_argument("--min-area", type=positive, default=1, help="Minimum LoG/Voronoi region area in evaluation-grid pixels")
    args = parser.parse_args(argv)
    from .postprocessing.runner import postprocess
    postprocess(args.predictions, args.output, method=args.method, reference_run=args.reference_run,
                k_factor=args.k_factor, min_area=args.min_area)


def motion_main(argv=None):
    parser = argparse.ArgumentParser(description="Analyze an explicit ordered frame pair and save displacement CSVs/figures.")
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("fusion", "phase_correlation", "optical_flow", "block_matching", "block_matching_sad"), default="fusion")
    parser.add_argument("--block-size", type=positive, default=64)
    parser.add_argument("--search-range", type=int, default=16)
    parser.add_argument("--spatial-scale", type=float, help="Measured millimetres per pixel")
    parser.add_argument("--seconds-per-frame", type=float, help="Measured interval between the two frames")
    args = parser.parse_args(argv)
    from .motion.runner import run_motion
    run_motion(args.first, args.second, args.output, method=args.method, block_size=args.block_size,
               search_range=args.search_range, spatial_scale=args.spatial_scale, seconds_per_frame=args.seconds_per_frame)


def pipeline_main(argv=None):
    parser = argparse.ArgumentParser(description="Linked train -> predict -> evaluate -> render experiment.")
    model_arguments(parser); dataset_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Skip training and use these existing weights")
    parser.add_argument("--epochs", type=positive)
    parser.add_argument("--limit", type=positive, help="Smoke-test limit for both source splits")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--mode", choices=("semantic", "amg"), default="semantic")
    parser.add_argument("--prompt-source", choices=("full_image", "ground_truth"))
    args = parser.parse_args(argv)
    config = with_overrides(configured(args), epochs=args.epochs)
    from .config import SAM_MODELS
    if config["model"]["name"] in SAM_MODELS and args.mode == "semantic" and args.prompt_source is None:
        parser.error("Prompted SAM evaluation requires an explicit --prompt-source")
    from .engine.training import train
    from .engine.prediction import predict
    from .engine.evaluation import evaluate
    from .visualization.segmentation import render_results
    checkpoint = args.checkpoint or train(config, args.dataset_root, args.manifest, args.output / "training", args.device,
                                          args.limit, args.limit, False if args.no_pretrained else None)
    predictions = predict(config, args.dataset_root, args.manifest, args.output / "predictions", checkpoint,
                          device=args.device, limit=args.limit, mode=args.mode, prompt_source=args.prompt_source)
    evaluate(predictions, args.dataset_root, args.manifest, args.output / "evaluation")
    render_results({config["model"]["name"]: predictions}, args.dataset_root, args.manifest,
                   args.output / "figures", limit=min(5, args.limit or 5))


def external_main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect the external-code/checkpoint inputs of incomplete RobustSAM/SAMUS attempts.")
    parser.add_argument("--model", choices=("robustsam", "samus"), required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args(argv)
    if not args.repository.is_dir():
        parser.error("External repository directory does not exist")
    root = args.repository.resolve()
    package = "robust_segment_anything" if args.model == "robustsam" else "models"
    if not (root / package).is_dir():
        parser.error(f"Expected {package}/ in the selected external repository")
    print(f"External source: {root}\nPython source files: {len(list((root/package).rglob('*.py')))}")
    if args.checkpoint:
        from .models.factory import load_checkpoint
        state = load_checkpoint(args.checkpoint)
        print(f"Checkpoint top-level keys: {list(state)[:15]}")
    print("The supplied notebooks contain setup/inference attempts, not a complete verified training pipeline. See docs/model_support.md.")
