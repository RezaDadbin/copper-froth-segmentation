"""Small real forward/backward, saved-mask evaluation and rendering tests."""

import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from froth.config import load_config
from froth.engine.evaluation import evaluate
from froth.engine.prediction import predict, require_compatible_checkpoint
from froth.engine.training import BCEDiceLoss, train
from froth.models.factory import configure_fine_tuning
from froth.models.unet import UNet
from froth.models.sam_family import box_prompt, prompted_logits
from froth.visualization.segmentation import render_results


ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


class ModelTests(unittest.TestCase):
    def test_unet_binary_logits_and_gradient(self):
        model = UNet()
        result = model(torch.rand(1, 3, 32, 32))
        self.assertEqual(tuple(result.shape), (1, 1, 32, 32))
        result.mean().backward()
        self.assertGreater(float(model.final.weight.grad.abs().sum()), 0)

    def test_prompt_source_is_deliberate(self):
        mask = np.zeros((10, 20), np.uint8); mask[2:5, 4:9] = 1
        np.testing.assert_equal(box_prompt(mask, "full_image"), [0, 0, 19, 9])
        np.testing.assert_equal(box_prompt(mask, "ground_truth"), [4, 2, 8, 4])
        with self.assertRaises(ValueError):
            box_prompt(mask, "automatic_guess")

    def test_real_small_sam_decoder_has_gradients(self):
        try:
            from segment_anything.modeling import Sam, ImageEncoderViT, PromptEncoder, MaskDecoder, TwoWayTransformer
        except ImportError:
            self.skipTest("Optional official SAM package is not installed")
        model = Sam(
            image_encoder=ImageEncoderViT(img_size=32, patch_size=4, embed_dim=32, depth=1,
                                         num_heads=4, mlp_ratio=2, out_chans=32),
            prompt_encoder=PromptEncoder(embed_dim=32, image_embedding_size=(8, 8), input_image_size=(32, 32), mask_in_chans=8),
            mask_decoder=MaskDecoder(transformer_dim=32, num_multimask_outputs=3,
                                     transformer=TwoWayTransformer(depth=1, embedding_dim=32, mlp_dim=64, num_heads=4)))
        configure_fine_tuning(model, "decoder_only")
        image = np.full((24, 40, 3), 100, np.uint8)
        mask = np.zeros((24, 40), np.uint8); mask[5:20, 10:25] = 1
        logits = prompted_logits(model, image, mask, "sam", "full_image")
        self.assertEqual(tuple(logits.shape), (1, 1, 24, 40))
        loss = BCEDiceLoss()(logits, torch.tensor(mask).float()[None, None])
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.mask_decoder.parameters()))
        self.assertTrue(all(p.grad is None for p in model.image_encoder.parameters()))


class PipelineTests(unittest.TestCase):
    def test_train_predict_evaluate_and_render_share_names_and_geometry(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); dataset = root / "dataset"
            for split, names in {"train1": ["a.tif", "b.tiff"], "test1": ["c.tif"]}.items():
                (dataset / split).mkdir(parents=True)
                for name in names:
                    image = np.zeros((48, 64, 3), np.uint8); image[10:35, 15:50] = 160
                    Image.fromarray(image).save(dataset / split / name)
                    (dataset / split / Path(name).with_suffix(".json")).write_text(json.dumps({
                        "imageWidth": 64, "imageHeight": 48, "shapes": [{"label": "froth", "shape_type": "polygon",
                        "points": [[15, 10], [50, 10], [50, 35], [15, 35]]}]}))
            manifest = root / "tags.csv"
            manifest.write_text("image_name,split\na.tif,train\nb.tiff,train\nc.tif,test\n")
            config = load_config(ROOT / "configs/unet.json")
            config["data"] = {"image_size": [32, 32], "normalize": False, "center_crop": None}
            config["training"].update(epochs=1, batch_size=2, num_workers=0, log_every=1)
            checkpoint = train(config, dataset, manifest, root / "training", device="cpu")
            self.assertTrue(checkpoint.is_file())
            self.assertTrue((root / "training/figures/training_curves.pdf").is_file())
            predictions = predict(config, dataset, manifest, root / "predictions", checkpoint, device="cpu")
            metrics = evaluate(predictions, dataset, manifest, root / "evaluation")
            with (metrics / "per_image_metrics.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["image_name"] for row in rows], ["c.tif"])
            self.assertEqual(rows[0]["evaluation_height"], "32")
            render_results({"U-Net": predictions}, dataset, manifest, root / "figures", limit=1)
            self.assertTrue((root / "figures/comparison__c.tif.pdf").is_file())
            incompatible = copy.deepcopy(config); incompatible["data"]["image_size"] = [64, 64]
            with self.assertRaisesRegex(ValueError, "preprocessing"):
                require_compatible_checkpoint(incompatible, checkpoint)
            # Changing labels after prediction must be caught by evaluation.
            annotation = dataset / "test1/c.json"
            annotation.write_text(annotation.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                evaluate(predictions, dataset, manifest, root / "invalid_evaluation")


if __name__ == "__main__":
    unittest.main()
