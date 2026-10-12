#!/usr/bin/env python3
"""Create Section 5/6 tables and figures from saved CSVs, without model runs.

CSV calculations use the standard library. Plotting requires numpy/matplotlib.
Run from any directory; inputs are resolved relative to this file's project.
"""

import argparse
import inspect
import csv
import hashlib
import html
import json
import math
import statistics as st
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = None
OUT = None
METRICS = ("pixel_acc", "iou", "dice", "precision", "recall", "HD", "HD95")
SCORES = METRICS[:5]
SPLITS = ("all", "train", "test")
DOMAINS = ("all", "yellow_copper", "grayscale")
ROLES = {
    "all": "mixed_training_and_reused_validation_descriptive_only",
    "train": "training",
    "test": "validation_used_for_selection_and_reporting",
}
SPLIT_LABELS = {"all": "All images (descriptive)", "train": "Training", "test": "Reused validation"}
MODEL_FILES = {
    "unet": "metrics_unet_all.csv",
    "sbsnet": "metrics_sbsnet_all.csv",
    "deeplabv3_resnet101": "metrics_deeplabv3_resnet101_all.csv",
    "deeplabv3plus_r152_os8": "metrics_deeplabv3plus_r152_os8_all.csv",
    "sam": "metrics_sam_amg_all.csv",
    "hqsam": "metrics_hqsam_hqamg_all.csv",
    "medsam": "metrics_medsam_all.csv",
    "medsam_tiled": "metrics_medsam_all_tiled.csv",
    "robustsam": "metrics_robustsam_all.csv",
}
MODEL_LABELS = {
    "unet": "U-Net", "sbsnet": "SBSNet",
    "deeplabv3_resnet101": "DeepLabV3*", "deeplabv3plus_r152_os8": "DeepLabV3+*",
    "sam": "SAM AMG", "hqsam": "HQ-SAM", "medsam": "MedSAM",
    "medsam_tiled": "MedSAM tiled", "robustsam": "RobustSAM*",
}
TAG_LABELS = {
    "domain_yellow_copper": "Colour domain",
    "structured_light_reflection": "Structured light reflection",
    "uneven_illumination": "Uneven illumination",
    "surface_artifacts_contaminants": "Surface artifacts / contaminants",
    "polydispersity_multiscale_bubbles": "Polydispersity / multiscale bubbles",
    "dense_small_bubble_region": "Dense small-bubble region",
    "elongated_irregular_bubbles": "Elongated / irregular bubbles",
    "poor_focus_defocus_blur": "Poor focus / defocus blur",
    "localized_motion_or_vibration_blur": "Motion / vibration blur",
    "collapsed": "Collapse appearance",
    "no_structured_light_reflection": "No structured light reflection",
    "no_surface_artifacts_contaminants": "No surface artifacts / contaminants",
    "at_most_2_condition_tags": "At most 2 condition tags",
    "at_least_6_condition_tags": "At least 6 condition tags",
}
GROUP_ORDER = list(TAG_LABELS)
SIZE_METRICS = {
    "area_px2": "px2", "area_pct_image": "percent_of_native_image_area",
    "equivalent_diameter_px": "px", "perimeter_px": "px",
    "compactness": "dimensionless", "bbox_aspect_ratio": "dimensionless",
}
CSV_OUTPUTS = {}
FIGURES = []


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not reader.fieldnames or None in reader.fieldnames or any(None in r for r in rows):
            raise ValueError(f"Malformed CSV: {path}")
        return reader.fieldnames, rows


def write_csv(relative, rows, fields=None):
    rows = list(rows)
    fields = list(fields or rows[0].keys())
    path = OUT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: "" if r.get(k) is None else r.get(k) for k in fields} for r in rows)
    saved_fields, saved = read_csv(path)
    assert saved_fields == fields and len(saved) == len(rows), relative
    CSV_OUTPUTS[relative] = len(rows)


def quantile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def summary(values):
    if not values:
        return {k: None for k in ("mean", "std", "median", "q05", "q25", "q75", "q95", "min", "max")}
    return {
        "mean": st.fmean(values), "std": st.stdev(values) if len(values) > 1 else None,
        "median": st.median(values), "q05": quantile(values, .05),
        "q25": quantile(values, .25), "q75": quantile(values, .75),
        "q95": quantile(values, .95), "min": min(values), "max": max(values),
    }


def selected(names, manifest, split="all", domain="all"):
    return [name for name in names
            if (split == "all" or manifest[name]["split"] == split)
            and (domain == "all" or manifest[name]["domain"] == domain)]


def model_summary(names, results):
    records = [results[n] for n in names if n in results]
    row = {"n_images": len(names), "n_scored": len(records), "n_missing": len(names) - len(records),
           "status": "no_images_in_group" if not names else "no_model_results" if not records else "available"}
    for metric in METRICS:
        s = summary([r[metric] for r in records])
        for key in ("mean", "std", "median", "q25", "q75"):
            row[f"{metric}_{key}"] = s[key]
    return row


def load():
    paths = [ROOT / "tags.csv", *sorted((ROOT / "tags").glob("*.csv")),
             *sorted((ROOT / "models").glob("*.csv"))]
    paths += [ROOT / "statistically reports" / name for name in
              ("image_statistics.csv", "froth_measurements.csv", "dataset_summary.csv", "run_report.json")]
    paths += [ROOT / "analysis/group_model_summary.csv",
              *sorted((ROOT / "tags and their accuracy").glob("*/*.csv"))]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    fields, rows = read_csv(ROOT / "tags.csv")
    tags = [f for f in fields if f not in ("image_name", "split")]
    conditions = [t for t in tags if t != "domain_yellow_copper"]
    manifest = {r["image_name"]: dict(r) for r in rows}
    assert len(manifest) == len(rows) and len(conditions) == 9
    for r in manifest.values():
        assert r["split"] in ("train", "test")
        assert all(r[t] in ("0", "1") for t in tags)
        r["domain"] = "yellow_copper" if r["domain_yellow_copper"] == "1" else "grayscale"
        r["condition_count"] = sum(int(r[t]) for t in conditions)
    groups = {}
    for p in sorted((ROOT / "tags").glob("*.csv")):
        head, group = read_csv(p)
        names = [r["image_name"] for r in group]
        assert head == ["image_name"] and len(names) == len(set(names))
        key = p.stem
        if key in tags:
            expected = [n for n, r in manifest.items() if r[key] == "1"]
        elif key.startswith("no_") and key[3:] in tags:
            expected = [n for n, r in manifest.items() if r[key[3:]] == "0"]
        elif key == "at_most_2_condition_tags":
            expected = [n for n, r in manifest.items() if r["condition_count"] <= 2]
        elif key == "at_least_6_condition_tags":
            expected = [n for n, r in manifest.items() if r["condition_count"] >= 6]
        else:
            raise ValueError(f"Unknown group: {key}")
        assert names == expected, f"Tag group mismatch: {key}"
        groups[key] = names
    assert set(groups) == set(GROUP_ORDER)
    assert {p.name for p in (ROOT / "models").glob("*.csv")} == set(MODEL_FILES.values())
    models = {}
    max_formula_error = 0
    for model, filename in MODEL_FILES.items():
        _, records = read_csv(ROOT / "models" / filename)
        results = {}
        for r in records:
            n = r["image_file"]
            assert n in manifest and n not in results
            assert r["split"] == {"train": "train", "test": "val"}[manifest[n]["split"]]
            for key in METRICS:
                r[key] = float(r[key])
                assert math.isfinite(r[key])
                assert r[key] >= 0 and (key not in SCORES or r[key] <= 1)
            for key in ("tp", "tn", "fp", "fn"):
                r[key] = int(r[key])
                assert r[key] >= 0
            tp, tn, fp, fn = (r[k] for k in ("tp", "tn", "fp", "fn"))
            total, positive = tp + tn + fp + fn, tp + fn
            assert total > 0 and positive > 0
            r["evaluated_pixels"] = total
            r["foreground_fraction"] = positive / total
            r["all_foreground_iou"] = positive / total
            r["all_foreground_dice"] = 2 * positive / (total + positive)
            for key, value in {"pixel_acc": (tp+tn)/total, "iou": tp/(tp+fp+fn),
                               "dice": 2*tp/(2*tp+fp+fn),
                               "precision": tp/(tp+fp) if tp+fp else 0,
                               "recall": tp/(tp+fn)}.items():
                max_formula_error = max(max_formula_error, abs(r[key] - value))
            results[n] = r
        models[model] = results
    assert max_formula_error < 1.1e-6, max_formula_error
    assert len({frozenset(r) for r in models.values()}) == 1, "Different scored image sets"
    _, records = read_csv(ROOT / "statistically reports/image_statistics.csv")
    images = {r["image_name"]: r for r in records}
    assert set(images) == set(manifest) and len(records) == len(images)
    numeric_image_fields = ("image_width_px", "image_height_px", "image_area_px2", "froth_count",
                            "valid_size_count", "invalid_geometry_count", "outside_image_count",
                            "within_1px_of_border_count", "polygon_count", "circle_count")
    for n, r in images.items():
        assert r["measurement_scope"] == "all_manifest_froth_annotations"
        assert (r["source_split"], r["domain"]) == (manifest[n]["split"], manifest[n]["domain"])
        for key in numeric_image_fields:
            r[key] = int(r[key])
        for key in ("area_mean_px2", "area_median_px2", "area_mean_pct_image", "area_median_pct_image",
                    "sum_individual_areas_px2", "equivalent_diameter_mean_px"):
            r[key] = float(r[key]) if r[key] else None
    _, saved_objects = read_csv(ROOT / "statistically reports/froth_measurements.csv")
    objects, by_image = [], defaultdict(list)
    for r in saved_objects:
        n = r["image_name"]
        assert n in manifest and r["measurement_scope"] == "all_manifest_froth_annotations"
        assert (r["source_split"], r["domain"]) == (manifest[n]["split"], manifest[n]["domain"])
        obj = {"image_name": n, "shape_index": int(r["shape_index"]), "shape_type": r["shape_type"],
               "valid": r["geometry_valid"] == "1", "issue": r["issue"],
               "input_point_count": int(r["input_point_count"])}
        for key in ("outside_image", "within_1px_of_border"):
            obj[key] = None if r[key] == "" else r[key] == "1"
        for key in SIZE_METRICS:
            obj[key] = float(r[key]) if r[key] else None
            assert obj[key] is None if not obj["valid"] else math.isfinite(obj[key])
        if obj["valid"]:
            assert math.isclose(obj["area_pct_image"], 100*obj["area_px2"]/images[n]["image_area_px2"], rel_tol=1e-12)
            assert math.isclose(obj["equivalent_diameter_px"], math.sqrt(4*obj["area_px2"]/math.pi), rel_tol=1e-12)
        objects.append(obj)
        by_image[n].append(obj)
    del saved_objects
    for n, r in images.items():
        obs = by_image[n]
        assert len(obs) == len({o["shape_index"] for o in obs}) == r["froth_count"]
        assert sum(o["valid"] for o in obs) == r["valid_size_count"]
        assert sum(not o["valid"] for o in obs) == r["invalid_geometry_count"]
        for flag in ("outside_image", "within_1px_of_border"):
            assert sum(o[flag] is True for o in obs) == r[flag+"_count"]
        valid_areas = [o["area_px2"] for o in obs if o["valid"]]
        assert math.isclose(st.fmean(valid_areas), r["area_mean_px2"], rel_tol=1e-12)
    run = json.loads((ROOT / "statistically reports/run_report.json").read_text())
    assert run["status"] == "complete" and run["mode"] == "full_dataset" and not run["failures"]
    assert run["tags_csv_sha256"] == hashes["tags.csv"], "Statistics used a different tag manifest"
    assert run["processed_image_count"] == len(manifest)
    assert run["measured_froth_count"] == len(objects)
    assert run["valid_size_count"] == sum(o["valid"] for o in objects)
    return manifest, tags, conditions, groups, models, images, objects, by_image, run, hashes, max_formula_error


def tag_and_model_tables(manifest, tags, conditions, groups, models):
    group_rows, domain_rows, counts = [], [], []
    all_names = list(manifest)
    for group in GROUP_ORDER:
        own_rows = []
        for split in SPLITS:
            for domain in DOMAINS:
                names = selected(groups[group], manifest, split, domain)
                denominator = len(selected(all_names, manifest, split, domain))
                counts.append({"tag_group": group, "source_split": split, "evaluation_role": ROLES[split],
                               "domain": domain, "n_images": len(names), "denominator_images": denominator,
                               "prevalence_pct": 100*len(names)/denominator if denominator else None})
                if domain != "all" and split == "all":
                    continue
                for model, results in models.items():
                    row = {"tag_group": group, "model": model, "source_split": split,
                           "evaluation_role": ROLES[split], "domain": domain, **model_summary(names, results)}
                    if domain == "all":
                        group_rows.append(row)
                        own_rows.append(row)
                    else:
                        domain_rows.append(row)
        write_csv(f"tags/{group}.csv", own_rows)
    write_csv("tags/model_averages_by_tag.csv", group_rows)
    write_csv("tags/model_averages_by_tag_and_domain.csv", domain_rows)
    write_csv("tags/tag_group_counts.csv", counts)
    overall, references = [], []
    for split in SPLITS:
        for domain in DOMAINS:
            names = selected(all_names, manifest, split, domain)
            for model, results in models.items():
                identifiers = {"model": model, "source_split": split, "evaluation_role": ROLES[split], "domain": domain}
                overall.append({**identifiers, **model_summary(names, results)})
                records = [results[n] for n in names if n in results]
                row = {**identifiers, "n_images": len(names), "n_scored": len(records),
                       "evaluated_pixel_counts": "|".join(map(str, sorted({r["evaluated_pixels"] for r in records}))),
                       "mean_foreground_fraction": st.fmean(r["foreground_fraction"] for r in records) if records else None}
                for key in ("iou", "dice"):
                    row[f"recorded_{key}_mean"] = st.fmean(r[key] for r in records) if records else None
                    row[f"all_foreground_{key}_mean"] = st.fmean(r["all_foreground_"+key] for r in records) if records else None
                    row[f"mean_delta_{key}_percentage_points"] = st.fmean(100*(r[key]-r["all_foreground_"+key]) for r in records) if records else None
                    row[f"n_images_above_all_foreground_{key}"] = sum(r[key] > r["all_foreground_"+key] + 1.1e-6 for r in records)
                references.append(row)
    write_csv("models/overall_model_averages.csv", overall)
    write_csv("models/foreground_reference.csv", references)
    cooccurrence, condition_counts, contrasts = [], [], []
    for split in SPLITS:
        names = selected(all_names, manifest, split)
        for a in conditions:
            a_names = {n for n in names if manifest[n][a] == "1"}
            for b in conditions:
                b_names = {n for n in names if manifest[n][b] == "1"}
                joint, union = len(a_names & b_names), len(a_names | b_names)
                cooccurrence.append({"source_split": split, "evaluation_role": ROLES[split], "tag_a": a, "tag_b": b,
                                     "n_images": len(names), "tag_a_positive": len(a_names), "tag_b_positive": len(b_names),
                                     "both_positive": joint, "either_positive": union,
                                     "jaccard": joint/union if union else None})
        for domain in DOMAINS:
            domain_names = selected(all_names, manifest, split, domain)
            cc = Counter(manifest[n]["condition_count"] for n in domain_names)
            for count in range(10):
                condition_counts.append({"source_split": split, "evaluation_role": ROLES[split], "domain": domain,
                                         "condition_count": count, "n_images": cc[count], "denominator_images": len(domain_names),
                                         "fraction": cc[count]/len(domain_names) if domain_names else None})
            if split == "all":
                continue
            for tag in conditions:
                tag0 = [n for n in domain_names if manifest[n][tag] == "0"]
                tag1 = [n for n in domain_names if manifest[n][tag] == "1"]
                for model, results in models.items():
                    r0, r1 = [results[n] for n in tag0 if n in results], [results[n] for n in tag1 if n in results]
                    row = {"tag": tag, "model": model, "source_split": split, "evaluation_role": ROLES[split],
                           "domain": domain, "tag0_n_images": len(tag0), "tag1_n_images": len(tag1),
                           "tag0_n_scored": len(r0), "tag1_n_scored": len(r1)}
                    for metric in SCORES:
                        mean0 = st.fmean(r[metric] for r in r0) if r0 else None
                        mean1 = st.fmean(r[metric] for r in r1) if r1 else None
                        row.update({f"tag0_{metric}_mean": mean0, f"tag1_{metric}_mean": mean1,
                                    f"{metric}_difference_percentage_points": 100*(mean1-mean0) if mean0 is not None and mean1 is not None else None})
                    contrasts.append(row)
    write_csv("tags/condition_cooccurrence.csv", cooccurrence)
    write_csv("tags/condition_count_distribution.csv", condition_counts)
    write_csv("tags/tag_presence_comparisons_by_domain.csv", contrasts)
    return group_rows, domain_rows, counts, overall, references, cooccurrence, condition_counts, contrasts


def statistical_tables(manifest, groups, images, objects, by_image):
    definitions = [("all", "all", list(manifest))]
    definitions += [("source_split", split, selected(manifest, manifest, split)) for split in ("train", "test")]
    definitions += [("domain", domain, selected(manifest, manifest, domain=domain)) for domain in DOMAINS[1:]]
    definitions += [("split_domain", split+"|"+domain, selected(manifest, manifest, split, domain))
                    for split in ("train", "test") for domain in DOMAINS[1:]]
    geometries = sorted({f"{r['image_width_px']}x{r['image_height_px']}" for r in images.values()})
    definitions += [("image_geometry", geom, [n for n, r in images.items()
                                             if f"{r['image_width_px']}x{r['image_height_px']}" == geom]) for geom in geometries]
    inventory, sizes = [], []
    for kind, value, names in definitions:
        img = [images[n] for n in names]
        obs = [o for n in names for o in by_image[n]]
        valid = [o for o in obs if o["valid"]]
        polygon_points = [o["input_point_count"] for o in obs if o["shape_type"] == "polygon"]
        count_stats = summary([r["froth_count"] for r in img])
        row = {"group_type": kind, "group_value": value,
               "evaluation_role": ROLES[value.split("|")[0]] if kind in ("source_split", "split_domain") else ROLES["all"],
               "n_images": len(img), "train_images": sum(manifest[n]["split"] == "train" for n in names),
               "reused_validation_images": sum(manifest[n]["split"] == "test" for n in names),
               "colour_domain_images": sum(manifest[n]["domain"] == "yellow_copper" for n in names),
               "grayscale_domain_images": sum(manifest[n]["domain"] == "grayscale" for n in names),
               "image_geometries": "|".join(sorted({f"{r['image_width_px']}x{r['image_height_px']}" for r in img})),
               "froth_annotations": len(obs), "valid_size_annotations": len(valid),
               "invalid_geometry_annotations": len(obs)-len(valid),
               "invalid_geometry_pct_annotations": 100*(len(obs)-len(valid))/len(obs),
               "images_with_invalid_geometry": sum(r["invalid_geometry_count"] > 0 for r in img),
               "valid_outside_image_annotations": sum(o["outside_image"] is True for o in valid),
               "images_with_outside_image_geometry": sum(r["outside_image_count"] > 0 for r in img),
               "valid_border_proxy_annotations": sum(o["within_1px_of_border"] is True for o in valid),
               "images_with_border_proxy_geometry": sum(r["within_1px_of_border_count"] > 0 for r in img),
               "json_imagePath_mismatch_images": sum(r["json_imagePath_matches"] != "True" for r in img),
               "polygon_annotations": sum(o["shape_type"] == "polygon" for o in obs),
               "circle_annotations": sum(o["shape_type"] == "circle" for o in obs),
               "polygon_vertices_mean": st.fmean(polygon_points), "polygon_vertices_median": st.median(polygon_points),
               "polygon_vertices_q25": quantile(polygon_points, .25), "polygon_vertices_q75": quantile(polygon_points, .75),
               "polygon_vertices_min": min(polygon_points), "polygon_vertices_max": max(polygon_points)}
        row.update({f"froth_count_per_image_{key}": value_stat for key, value_stat in count_stats.items()})
        inventory.append(row)
        policies = {"all_valid_unclipped": valid,
                    "inside_image_only": [o for o in valid if not o["outside_image"]],
                    "excluding_border_proxy": [o for o in valid if not o["outside_image"] and not o["within_1px_of_border"]]}
        for policy, population in policies.items():
            for metric, unit in SIZE_METRICS.items():
                values = [o[metric] for o in population]
                per_image = defaultdict(list)
                for o in population:
                    per_image[o["image_name"]].append(o[metric])
                sizes.append({"group_type": kind, "group_value": value, "inclusion_policy": policy,
                              "measurement": metric, "unit": unit, "weighting": "object",
                              "n_annotations": len(values), "n_images_contributing": len(per_image),
                              **summary(values),
                              "image_weighted_mean": st.fmean(st.fmean(v) for v in per_image.values()) if per_image else None})
    write_csv("statistics/dataset_inventory.csv", inventory)
    write_csv("statistics/froth_size_summary.csv", sizes)
    by_tag = []
    for group in GROUP_ORDER:
        for split in SPLITS:
            for domain in DOMAINS:
                names = selected(groups[group], manifest, split, domain)
                img = [images[n] for n in names]
                total = sum(r["froth_count"] for r in img)
                nvalid = sum(r["valid_size_count"] for r in img)
                row = {"tag_group": group, "source_split": split, "evaluation_role": ROLES[split], "domain": domain,
                       "n_images": len(img), "froth_annotations": total, "valid_size_annotations": nvalid,
                       "invalid_geometry_annotations": total-nvalid,
                       "invalid_geometry_pct_annotations": 100*(total-nvalid)/total if total else None,
                       "object_weighted_area_mean_px2": sum(r["sum_individual_areas_px2"] for r in img)/nvalid if nvalid else None,
                       "object_weighted_area_mean_pct_image": sum(r["area_mean_pct_image"]*r["valid_size_count"] for r in img)/nvalid if nvalid else None}
                for field in ("froth_count", "area_mean_px2", "area_median_px2", "area_mean_pct_image", "area_median_pct_image"):
                    stats = summary([r[field] for r in img if r[field] is not None])
                    for key in ("mean", "std", "median", "q25", "q75"):
                        row[f"per_image_{field}_{key}"] = stats[key]
                by_tag.append(row)
    write_csv("statistics/statistics_by_tag.csv", by_tag)
    return inventory, sizes, by_tag


def validate_against_previous(manifest, group_rows, overall, inventory, sizes):
    checks = {}
    _, previous = read_csv(ROOT / "analysis/group_model_summary.csv")
    current = {(r["tag_group"], r["model"], r["source_split"]): r for r in group_rows}
    compared = 0
    for old in previous:
        new = current[old["tag_group"], old["model"], old["split"]]
        for key in ("n_images", "n_scored", "n_missing"):
            assert int(old[key]) == new[key]
        for metric in METRICS:
            for stat in ("mean", "std", "median"):
                key = f"{metric}_{stat}"
                assert (old[key] == "") == (new[key] is None)
                if old[key]:
                    assert math.isclose(float(old[key]), new[key], rel_tol=8e-10, abs_tol=8e-10), (old["tag_group"], key)
                compared += 1
    checks["existing_group_summary_numeric_cells_checked"] = compared
    exported_checks = 0
    for group in GROUP_ORDER:
        for model in MODEL_FILES:
            _, exported = read_csv(ROOT / "tags and their accuracy" / group / (model+".csv"))
            assert len(exported) == len({r["image_name"] for r in exported})
            for split in SPLITS:
                members = [r for r in exported if split == "all" or r["split"] == split]
                scored = [r for r in members if r["dice"] != ""]
                new = current[group,model,split]
                assert len(members) == new["n_images"] and len(scored) == new["n_scored"]
                for metric in METRICS:
                    assert all(r[metric] == "" for r in members if r["dice"] == "")
                    direct = math.fsum(float(r[metric]) for r in scored)/len(scored) if scored else None
                    assert direct is None if new[metric+"_mean"] is None else math.isclose(direct,new[metric+"_mean"],rel_tol=1e-12,abs_tol=1e-12)
                    exported_checks += 1
    checks["exported_means_independently_checked_against_126_existing_group_files"] = exported_checks
    _, source = read_csv(ROOT / "statistically reports/dataset_summary.csv")
    inv = {(r["group_type"], r["group_value"]): r for r in inventory}
    siz = {(r["group_type"], r["group_value"], r["measurement"], r["inclusion_policy"]): r for r in sizes}
    for old in source:
        key = old["group_type"], old["group_value"]
        new = inv[key]
        assert int(old["image_count"]) == new["n_images"]
        assert int(old["froth_count"]) == new["froth_annotations"]
        for metric in ("area_px2", "equivalent_diameter_px", "area_pct_image"):
            s = siz[*key, metric, "all_valid_unclipped"]
            source_key = {"area_px2": "area_mean_px2", "equivalent_diameter_px": "equivalent_diameter_mean_px", "area_pct_image": "area_mean_pct_image"}[metric]
            assert math.isclose(float(old[source_key]), s["mean"], rel_tol=1e-12)
        area = siz[*key, "area_px2", "all_valid_unclipped"]
        assert math.isclose(float(old["image_weighted_area_mean_px2"]), area["image_weighted_mean"], rel_tol=1e-12)
    checks["saved_dataset_summary_groups_checked"] = len(source)
    for group in GROUP_ORDER:
        for model in MODEL_FILES:
            a, t, v = (current[group, model, split] for split in SPLITS)
            assert a["n_scored"] == t["n_scored"] + v["n_scored"]
            for metric in METRICS:
                if a["n_scored"]:
                    weighted = sum((r[f"{metric}_mean"] or 0)*r["n_scored"] for r in (t,v))/a["n_scored"]
                    assert math.isclose(weighted, a[f"{metric}_mean"], rel_tol=1e-12, abs_tol=1e-12)
    checks["whole_group_means_equal_count_weighted_split_means"] = True
    checks["tag_definitions_verified_against_manifest"] = True
    checks["all_per_image_geometry_totals_reconciled"] = True
    return checks


def plot_all(manifest, conditions, groups, models, images, objects, tables, stats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "semibold", "axes.titlesize": 11,
                         "figure.dpi": 120, "savefig.dpi": 300,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    colours = {"yellow_copper": "#b96c11", "grayscale": "#476b83"}
    split_colours = {"train": "#2274a5", "test": "#bd3f30"}
    models_order = list(MODEL_FILES)
    group_rows, domain_rows, counts, overall, references, cooccurrence, condition_counts, contrasts = tables
    inventory, sizes, by_tag = stats
    group_index = {(r["tag_group"], r["model"], r["source_split"]): r for r in group_rows}

    def save(fig, name, section, caption):
        path = OUT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path.with_suffix(".png"), bbox_inches="tight", facecolor="white")
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
        FIGURES.append({"name": name, "section": section, "caption": caption})
        plt.close(fig)

    def ecdf(ax, values, label, colour):
        values = np.sort(np.asarray(values, dtype=float))
        ax.step(values, np.arange(1, len(values)+1)/len(values), where="post", label=label, color=colour, lw=1.8)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=.18)
        ax.set_ylabel("Fraction of valid annotations")

    def heatmap(ax, data, rowlabels, columnlabels, vmin=0, vmax=100, cmap="viridis", fmt=".1f", textsize=8):
        palette = plt.get_cmap(cmap).copy()
        palette.set_bad("#eeeeee")
        data = np.asarray(data, dtype=float)
        artist = ax.imshow(np.ma.masked_invalid(data), cmap=palette, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(columnlabels)), columnlabels, rotation=38, ha="right")
        ax.set_yticks(range(len(rowlabels)), rowlabels)
        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                value = data[y,x]
                if not np.isfinite(value):
                    label, ink = "N/A", "#555555"
                else:
                    label = format(value, fmt)
                    rgba = palette((value-vmin)/(vmax-vmin))
                    luminance = .299*rgba[0]+.587*rgba[1]+.114*rgba[2]
                    ink = "black" if luminance > .56 else "white"
                ax.text(x,y,label,ha="center",va="center",fontsize=textsize,color=ink)
        return artist

    # Section 5: composition and tag metadata.
    fig, axs = plt.subplots(1, 2, figsize=(10.8, 4.0))
    for j, domain in enumerate(DOMAINS[1:]):
        bottom = 0
        for split in ("train", "test"):
            n = len(selected(manifest, manifest, split, domain))
            axs[0].bar(j, n, bottom=bottom, color=split_colours[split], width=.55,
                       label=SPLIT_LABELS[split] if j == 0 else None)
            axs[0].text(j, bottom+n/2, str(n), ha="center", va="center", color="white", weight="bold")
            bottom += n
    axs[0].set_xticks([0,1], ["Colour domain", "Grayscale domain"])
    axs[0].set_ylabel("Images")
    axs[0].set_title("a  Dataset composition (N = 595)")
    axs[0].legend(frameon=False)
    x = np.arange(10)
    for split in ("train", "test"):
        y = [sum(r["split"] == split and r["condition_count"] == count for r in manifest.values()) for count in range(10)]
        offset = -.19 if split == "train" else .19
        axs[1].bar(x+offset, y, width=.38, color=split_colours[split], label=SPLIT_LABELS[split])
    axs[1].set_xticks(x)
    axs[1].set_xlabel("Number of positive condition tags (out of 9)")
    axs[1].set_ylabel("Images")
    axs[1].set_title("b  Recorded condition count")
    axs[1].legend(frameon=False)
    fig.tight_layout()
    save(fig, "figures/section5_dataset_composition", 5,
         "Dataset composition by source split and domain, and number of recorded positive conditions per image. "
         "Source test images are reused validation (selection/tuning and reporting). The domain tag is excluded from the nine-condition count; this count is not a calibrated difficulty scale.")

    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    y = np.arange(len(conditions))
    labels = []
    for i, tag in enumerate(conditions):
        n = sum(r[tag] == "1" for r in manifest.values())
        labels.append(f"{TAG_LABELS[tag]}   [{n}/595]")
    for split, offset in (("train", -.17), ("test", .17)):
        names = selected(manifest, manifest, split)
        values = [100*sum(manifest[n][t] == "1" for n in names)/len(names) for t in conditions]
        bars = ax.barh(y+offset, values, height=.32, color=split_colours[split], label=f"{SPLIT_LABELS[split]} (N={len(names)})")
        for bar, value, tag in zip(bars, values, conditions):
            ax.text(min(value+1, 100), bar.get_y()+bar.get_height()/2,
                    f"{sum(manifest[n][tag] == '1' for n in names)}", va="center", fontsize=8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 109)
    ax.set_xlabel("Images with condition (%) within source split; labels show image counts")
    ax.set_title("Recorded condition prevalence")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "figures/section5_tag_prevalence", 5,
         "Prevalence of the nine condition tags within training and reused-validation splits. Bar-end values are positive image counts; bracketed counts pool the 595 manifest images. Conditions overlap and must not be summed as independent images. Collapse appearance is an image-level tag, not an event count.")

    fig, axs = plt.subplots(1, 3, figsize=(17, 6.0))
    coindex = {(r["source_split"],r["tag_a"],r["tag_b"]): r for r in cooccurrence}
    labels = [TAG_LABELS[t].replace(" / ", "/").replace("Structured light reflection", "Light reflection") for t in conditions]
    for ax, split in zip(axs, SPLITS):
        data = [[100*coindex[split,a,b]["jaccard"] if coindex[split,a,b]["jaccard"] is not None else np.nan for b in conditions] for a in conditions]
        im = heatmap(ax,data,labels if split == "all" else [""]*9,labels,fmt=".0f",textsize=7)
        ax.set_title(f"{SPLIT_LABELS[split]} (N={len(selected(manifest,manifest,split))})")
    fig.subplots_adjust(left=.14, right=.94, bottom=.38, top=.90, wspace=.12)
    cax = fig.add_axes([.95,.39,.012,.48])
    fig.colorbar(im, cax=cax, label="Tag-set Jaccard (%)")
    save(fig, "figures/section5_condition_cooccurrence", 5,
         "Pairwise Jaccard overlap of positive tag sets, 100 times the number of images with both tags divided by the number with either tag. Panels separate all images, training and reused validation. Diagonal entries are self-overlap; this matrix is not a causal effect or a correlation coefficient.")

    fig, axs = plt.subplots(1, 2, figsize=(11.3, 4.5))
    rng = np.random.default_rng(831)
    for i, domain in enumerate(DOMAINS[1:]):
        names = selected(manifest, manifest, domain=domain)
        values = [images[n]["froth_count"] for n in names]
        axs[0].scatter(np.full(len(names),i)+rng.uniform(-.18,.18,len(names)), values,
                       s=10, alpha=.32, color=colours[domain], rasterized=True)
        b = axs[0].boxplot([values], positions=[i], widths=.42, **{"tick_labels" if "tick_labels" in inspect.signature(axs[0].boxplot).parameters else "labels": [""]},
                           showfliers=False, patch_artist=True)
        for patch in b["boxes"]:
            patch.set(facecolor="none", edgecolor="black")
        axs[1].scatter([images[n]["froth_count"] for n in names],
                       [images[n]["area_mean_pct_image"] for n in names],
                       s=14, alpha=.45, color=colours[domain],
                       label=f"{'Colour' if i == 0 else 'Grayscale'} (N={len(names)})", rasterized=True)
    axs[0].set_xticks([0,1], ["Colour\nN=340", "Grayscale\nN=255"])
    axs[0].set_ylabel("Froth annotations per image (all geometries)")
    axs[0].set_title("a  Annotation-count distribution")
    axs[1].set_xlabel("Froth annotations per image (all geometries)")
    axs[1].set_ylabel("Mean valid annotation area (% of native image)")
    axs[1].set_title("b  Image-level count and relative size")
    axs[1].legend(frameon=False)
    for ax in axs:
        ax.grid(alpha=.18)
    fig.tight_layout()
    save(fig, "figures/section5_annotation_counts", 5,
         "Each point represents one image. (a) Annotation counts include geometrically flagged shapes; boxes mark the median and interquartile range, with 1.5-IQR whiskers. (b) Mean image-relative area uses valid geometries only. The count-size relationship is descriptive and constrained by frame occupancy and annotation rules; native fields of view are not established as equal.")

    fig, axs = plt.subplots(2,2,figsize=(10.5,7.3))
    specs = [("area_px2", "a  Projected annotation area (px²)", True),
             ("area_pct_image", "b  Annotation area (% of native image)", True),
             ("equivalent_diameter_px", "c  Equivalent circular diameter (px)", True),
             ("compactness", "d  Compactness, 4πA/P²", False)]
    for ax, (metric, label, log) in zip(axs.flat,specs):
        for domain in DOMAINS[1:]:
            vals = [o[metric] for o in objects if o["valid"] and manifest[o["image_name"]]["domain"] == domain]
            ecdf(ax, vals, f"{'Colour' if domain == 'yellow_copper' else 'Grayscale'} (n={len(vals):,})", colours[domain])
        if log:
            ax.set_xscale("log")
        ax.set_xlabel(label)
    axs[0,0].legend(frameon=False, loc="lower right")
    fig.suptitle("Valid annotation geometry (n = 78,257); native pixels, uncalibrated physical scale", fontsize=12)
    fig.tight_layout(rect=[0,0,1,.96])
    save(fig, "figures/section5_froth_size_distributions", 5,
         "Object-weighted empirical cumulative distributions of valid projected area, image-relative area, equal-area circular diameter and compactness, separated by image domain. The 430 flagged geometries are excluded; three valid outside-frame annotations are retained unclipped. Colour and grayscale images have different native resolutions, and no physical calibration or equal field of view is assumed. Curves describe annotations rather than independent physical acquisitions.")

    # Section 6: all tag means and a chart for each original group CSV.
    for split in ("train", "test"):
        fig, axs = plt.subplots(1,2,figsize=(15.0,8.1))
        rowlabels = [f"{TAG_LABELS[g]}  (n={group_index[g,models_order[0],split]['n_scored']})" for g in GROUP_ORDER]
        for ax, metric in zip(axs,("dice","iou")):
            data = [[100*group_index[g,m,split][f"{metric}_mean"]
                     if group_index[g,m,split][f"{metric}_mean"] is not None else np.nan for m in models_order] for g in GROUP_ORDER]
            im = heatmap(ax,data,rowlabels if metric=="dice" else [""]*len(GROUP_ORDER),
                         [MODEL_LABELS[m] for m in models_order],textsize=8)
            ax.set_title(f"Mean {metric.upper() if metric=='iou' else 'Dice'} (%)")
        fig.suptitle(f"Tag-group averages — {SPLIT_LABELS[split]}", fontsize=13)
        fig.subplots_adjust(left=.28,right=.94,bottom=.14,top=.91,wspace=.09)
        cax=fig.add_axes([.95,.17,.012,.70])
        fig.colorbar(im,cax=cax,label="Per-image mean score (%)")
        fig.text(.28,.015,"Recorded runs; grids/crops differ. * Run identity needs confirmation. Groups overlap; n = scored images.",fontsize=9)
        suffix = "training" if split=="train" else "reused_validation"
        save(fig, f"figures/section6_tag_averages_{suffix}", 6,
             f"Mean per-image Dice and IoU for every original tag CSV on {SPLIT_LABELS[split].lower()} images. "
             "Row labels give scored image counts; N/A denotes an empty group. Every image has equal weight within each group. Groups overlap; the colour-domain row is metadata rather than a condition. These supplied runs use different evaluation grids/crops and do not establish a controlled model ranking. Asterisks identify unresolved run provenance.")

    for group in GROUP_ORDER:
        fig, axs = plt.subplots(1,2,figsize=(10.3,4.9),sharey=True)
        for ax, metric in zip(axs,("dice","iou")):
            for split, offset, marker in (("train",-.13,"o"),("test",.13,"s")):
                rows = [group_index[group,m,split] for m in models_order]
                n_scored = rows[0]["n_scored"]
                if n_scored:
                    values = [100*r[f"{metric}_mean"] for r in rows]
                    ax.scatter(values,np.arange(9)+offset,s=42,marker=marker,color=split_colours[split],
                               label=f"{SPLIT_LABELS[split]} (n={n_scored})",zorder=3)
                else:
                    ax.plot([],[],marker=marker,ls="none",color=split_colours[split],label=f"{SPLIT_LABELS[split]}: N/A (n=0)")
            ax.set_xlim(-2,102)
            ax.set_xlabel(f"Mean {metric.upper() if metric=='iou' else 'Dice'} (%)")
            ax.grid(axis="x",alpha=.2)
            ax.set_yticks(range(9),[MODEL_LABELS[m] for m in models_order])
        axs[0].invert_yaxis()
        axs[1].legend(frameon=False,loc="lower left",fontsize=8)
        total=len(groups[group])
        fig.suptitle(f"{TAG_LABELS[group]}  |  {total} images in source tag CSV",fontsize=12)
        fig.text(.14,.015,"Arithmetic means of supplied per-image scores; grids/crops differ. * Run identity needs confirmation.",fontsize=8)
        fig.tight_layout(rect=[0,.05,1,.94])
        save(fig,f"tags/charts/{group}",6,
             f"Mean per-image Dice and IoU for the `{group}.csv` group ({total} manifest images), "
             "with training and reused validation kept separate. Legend counts refer to scored images, excluding missing model outputs. "
             "N/A indicates no images. Values are descriptive scores from the supplied runs, with different evaluation grids/crops; asterisks mark unresolved run identity.")

    fig,axs=plt.subplots(1,2,figsize=(12.4,5.7),sharey=True)
    rng=np.random.default_rng(932)
    for ax,domain in zip(axs,DOMAINS[1:]):
        names=selected(manifest,manifest,"test",domain)
        for j,model in enumerate(models_order):
            vals=np.asarray([models[model][n]["dice"]*100 for n in names if n in models[model]])
            ax.scatter(vals,j+rng.uniform(-.19,.19,len(vals)),s=11,alpha=.45,color=colours[domain],rasterized=True)
            ax.plot([np.quantile(vals,.25),np.quantile(vals,.75)],[j,j],color="black",lw=3,zorder=3)
            ax.scatter([np.mean(vals)],[j],color="black",marker="D",s=24,zorder=4)
        ax.set_yticks(range(9),[MODEL_LABELS[m] for m in models_order])
        ax.set_xlabel("Per-image Dice (%)")
        ax.set_xlim(-2,102)
        ax.set_title(f"{'Colour' if domain=='yellow_copper' else 'Grayscale'} domain (N={len(names)})")
        ax.grid(axis="x",alpha=.2)
    axs[0].invert_yaxis()
    fig.suptitle("Reused-validation score distributions by domain",fontsize=13)
    fig.text(.15,.025,"Points: images; black diamond: mean; black line: IQR. Recorded runs use different grids/crops; * identity unconfirmed.",fontsize=9)
    fig.tight_layout(rect=[0,.06,1,.95])
    save(fig,"figures/section6_score_distributions_by_domain",6,
         "Per-image Dice distributions on the 110 reused-validation images, split into 60 colour and 50 grayscale images. Black diamonds show means; thick lines show interquartile ranges, not confidence intervals. Runs are kept in architecture-family order rather than ranked. Different grids/crops and unresolved run identities limit comparisons.")

    fig,axs=plt.subplots(1,2,figsize=(11.5,5.4),sharey=True)
    refindex={(r["model"],r["source_split"],r["domain"]):r for r in references}
    for ax,metric in zip(axs,("dice","iou")):
        for j,model in enumerate(models_order):
            row=refindex[model,"test","all"]
            recorded=100*row[f"recorded_{metric}_mean"]
            baseline=100*row[f"all_foreground_{metric}_mean"]
            ax.plot([recorded,baseline],[j,j],color="#aaaaaa",lw=1)
            ax.scatter(recorded,j,s=42,color="#2274a5",label="Recorded model" if j==0 else None)
            ax.scatter(baseline,j,s=52,marker="|",color="#bd3f30",label="All foreground on same GT grid" if j==0 else None)
        ax.set_yticks(range(9),[MODEL_LABELS[m] for m in models_order])
        ax.set_xlabel(f"Mean {metric.upper() if metric=='iou' else 'Dice'} (%)")
        ax.set_xlim(-2,102)
        ax.grid(axis="x",alpha=.18)
    axs[0].invert_yaxis()
    axs[0].legend(frameon=False,loc="lower left",fontsize=8)
    fig.suptitle("Foreground-occupancy reference — reused validation (N = 110)",fontsize=12)
    fig.text(.15,.02,"Each reference uses that run's own evaluated ground-truth grid; this does not harmonize model targets or fields of view.",fontsize=8)
    fig.tight_layout(rect=[0,.06,1,.94])
    save(fig,"figures/section6_foreground_reference",6,
         "Recorded mean scores alongside an exact all-foreground prediction reference on each run's own evaluated ground-truth grid. With P=TP+FN and N=TP+TN+FP+FN, reference IoU=P/N and Dice=2P/(N+P), calculated per image and then averaged. This illustrates foreground saturation and does not normalize grids or validate bubble boundaries/instances.")

    fig,axs=plt.subplots(1,3,figsize=(17.5,6.3))
    contrast_index={(r["tag"],r["model"],r["source_split"],r["domain"]):r for r in contrasts}
    finite=[abs(r["dice_difference_percentage_points"]) for r in contrasts
            if r["source_split"]=="test" and r["dice_difference_percentage_points"] is not None]
    lim=math.ceil(max(finite)/5)*5
    for ax,domain in zip(axs,DOMAINS):
        rows=[contrast_index[t,models_order[0],"test",domain] for t in conditions]
        labels=[f"{TAG_LABELS[t]} ({r['tag1_n_scored']}/{r['tag0_n_scored']})" for t,r in zip(conditions,rows)]
        data=[[contrast_index[t,m,"test",domain]["dice_difference_percentage_points"] for m in models_order] for t in conditions]
        data=[[np.nan if v is None else v for v in row] for row in data]
        im=heatmap(ax,data,labels if domain=="all" else [f"{r['tag1_n_scored']}/{r['tag0_n_scored']}" for r in rows],
                   [MODEL_LABELS[m] for m in models_order],vmin=-lim,vmax=lim,cmap="RdBu",textsize=7)
        ax.set_title({"all":"All domains","yellow_copper":"Colour domain","grayscale":"Grayscale domain"}[domain])
    fig.subplots_adjust(left=.23,right=.94,bottom=.20,top=.90,wspace=.15)
    cax=fig.add_axes([.95,.22,.012,.64])
    fig.colorbar(im,cax=cax,label="Mean Dice(tag=1) − mean Dice(tag=0), pp")
    fig.text(.23,.025,"Reused validation; labels give positive/negative scored N. Descriptive associations; overlapping tags remain unadjusted.",fontsize=9)
    save(fig,"figures/section6_tag_contrasts_by_domain",6,
         "Difference in mean Dice between tag-positive and tag-negative reused-validation images, in percentage points. Panels show all domains and within-domain comparisons; row labels give positive/negative scored counts. N/A means at least one group is empty. Domain stratification does not adjust other overlapping tags, source sessions or selection effects; these are descriptive associations, not causal effects or significance tests.")
    return {"numpy":np.__version__,"matplotlib":matplotlib.__version__}


def write_documents(manifest, conditions, groups, models, images, objects, run, tables, stats, checks, versions):
    group_rows, domain_rows, counts, overall, references, cooccurrence, condition_counts, contrasts = tables
    inventory, sizes, by_tag = stats
    all_inventory=next(r for r in inventory if r["group_type"]=="all")
    sizeindex={(r["group_type"],r["group_value"],r["inclusion_policy"],r["measurement"]):r for r in sizes}
    area=sizeindex["all","all","all_valid_unclipped","area_px2"]
    inside=sizeindex["all","all","inside_image_only","area_px2"]
    interior=sizeindex["all","all","excluding_border_proxy","area_px2"]
    diameter=sizeindex["all","all","all_valid_unclipped","equivalent_diameter_px"]
    readme="""# Evaluations for the professor's Sections 5 and 6

These reports were calculated from the saved model, tag and annotation-statistics
CSVs. No models were trained, no JSON geometry traversal was repeated, and no
source images, annotations, tags or model results were changed.

## Start here

- [Paper notes and numerical summary](paper_notes.md): selected results and how to write them.
- [All tag/model averages](tags/model_averages_by_tag.csv): one table for every group, run and split.
- `tags/<original-tag-name>.csv`: one **averages** CSV for each of the 14 source tag CSVs.
- `tags/charts/`: a Dice/IoU comparison chart for each tag group, in PNG and vector PDF.
- [Dataset inventory](statistics/dataset_inventory.csv): composition, annotation counts, quality flags and polygon tracing detail.
- [Froth size summaries](statistics/froth_size_summary.csv): means, medians, quantiles and sensitivity populations.
- [Annotation statistics by tag](statistics/statistics_by_tag.csv): image counts, annotation counts and image/object size averages within each group.
- [Figure captions](figure_captions.md) and [visual index](index.html): figures aligned with Sections 5 and 6.
- [Verification record](verification.json): input hashes, checks and output counts.

## Section 5 — dataset statistics

`dataset_inventory.csv` separates the whole resource, source split, domain,
split × domain and native image geometry. Annotation counts include invalid shapes.
Polygon vertex counts include all polygons (valid and flagged); the single circle
is excluded from polygon-vertex summaries. Vertex density describes tracing effort,
and is not proof of annotation accuracy.

`froth_size_summary.csv` contains six credible 2D geometry measures, each with
mean, sample SD, median, 5th/25th/75th/95th percentiles, minimum and maximum:
projected area (px²), area as a percentage of native image area, equivalent circular
diameter (px), perimeter (px), compactness (4πA/P²), and axis-aligned bounding-box
aspect ratio (longer side / shorter side). The source geometry script uses population
SD; this presentation consistently uses **sample SD (n−1)**. Quantiles use linear
interpolation at (n−1)p. SD is blank for n<2.

The main weighting gives every valid annotation equal weight. `image_weighted_mean`
first averages within an image and then averages those image means, giving equal
weight to contributing images. The `n_images_contributing` column gives its denominator.

Three explicitly named size populations are supplied: `all_valid_unclipped` (the
original report policy), `inside_image_only` (excluding the three outside-frame
valid annotations), and `excluding_border_proxy` (also excluding annotations whose
bounding boxes reach within one pixel of an image border). This last population is
a sensitivity description; it is not a verified complete-bubble population.

Native pixels are uncalibrated: compare domains/resolutions descriptively, without
claiming a physical bubble-size difference. Relative area does not establish equal
field of view. None of these measurements is 3D volume. Perimeters/compactness depend
on boundary sampling; box aspect ratio depends on image orientation. Summed
individual areas can double-count overlaps and are not union foreground coverage.

Existing individual sizes remain in `../statistically reports/froth_measurements.csv`
and `froth_sizes_by_image.json`; per-image summaries remain in `image_statistics.csv`.
The small `test_5_froths` demonstration reports were excluded from this analysis.

## Section 6 — averages for every tag CSV

Each original tag group has 27 rows: nine recorded model configurations × three
source-split scopes (`all`, `train`, `test`). Every image with a recorded result has
equal weight. **The requested whole-group average is `source_split=all`**; it mixes
training and validation and is for descriptive lookup, not a generalization estimate.
It is calculated directly over images, not as an unweighted average of split means.

All seven supplied metrics are preserved: pixel accuracy, IoU, Dice, precision,
recall, HD and HD95. CSV overlap/accuracy scores are fractions in [0,1]; charts use
percent. `*_mean` is the requested average; sample SD, median and quartiles show
dispersion. HD/HD95 retain each run's source distance units; no cross-run distance
chart or unit conversion is inferred. The model files do not permit a trustworthy
common-grid boundary comparison.

`n_images` is source group membership; `n_scored` is the denominator for every score;
`n_missing` is their difference. Empty groups have blank numeric summaries, explicit
status and N/A in figures. Missing images are excluded, never scored as zero.

Additional interpretation tables:

- `tag_group_counts.csv`: group sizes/prevalence by split/domain, independent of model missingness.
- `model_averages_by_tag_and_domain.csv`: within-domain group averages, training and validation separately.
- `tag_presence_comparisons_by_domain.csv`: tag-positive and tag-negative means plus differences in percentage points, including scored Ns; these are descriptive, unadjusted contrasts.
- `condition_cooccurrence.csv`: pairwise Jaccard overlaps of positive condition sets (not correlations).
- `condition_count_distribution.csv`: counts for all 0–9 condition totals, excluding domain.
- `models/overall_model_averages.csv`: overall and domain-specific reference means.
- `models/foreground_reference.csv`: exact same-GT-grid all-foreground references and model-minus-reference differences.

## Interpretation that must accompany manuscript tables

1. Source `test` means **reused validation**: these 110 images were used for
   selection/tuning and reporting, as confirmed by the user. Model CSVs call them
   `val`. The 485 training images have 477 recorded results per run; all nine runs
   omit the same eight training images. Do not call these an independent test set.
2. Domain is metadata, not one of the nine conditions. Positive/negative tags and
   composite groups overlap, so their sizes cannot be summed as a resource total.
   The six-or-more-condition group contains zero validation images. A zero tag
   indicates absence of that recorded condition, not independently certified cleanliness.
3. Native evaluated pixel counts differ by run (256², 512², 1024² or 1440×1080 in
   the saved files); masks, crops and prompts also need harmonized provenance.
   Results are labeled **recorded runs**, not an established controlled architecture ranking.
4. DeepLab backbone labels conflict with inspected notebooks, and the RobustSAM
   notebook contains an ordinary-SAM fallback. Asterisks mark these run identities
   pending the records that produced the CSVs. MedSAM prompt provenance also
   requires confirmation; no automatic-versus-oracle comparison is claimed here.
5. All-foreground reference IoU=P/N and Dice=2P/(N+P), with P=TP+FN and
   N=TP+TN+FP+FN, are computed per image and then averaged on each run's own GT grid.
   High region scores do not prove bubble separation or correct instance sizes.
6. Tags correlate with domain and each other. Sparse Ns and empty strata remain
   visible. No significance tests, confidence intervals or causal statements are
   fabricated from overlapping groups with unknown acquisition dependence.
7. Geometry flags are disclosed, not silently repaired. No between-polygon overlap
   audit or independent human annotation review was performed for this report.

## Reproduce

For numerical tables only (Python standard library):

```sh
python3 evaluations/build_evaluations.py --csv-only   # run from the project folder
```

For figures, use a virtual environment with the packages in `requirements.txt`, then
run the same script without `--csv-only`. This run used an isolated temporary
environment, leaving the user's global Python installation unchanged. Inputs must
remain in their existing project locations. `--csv-only` updates tables, notes and
verification, but does not regenerate existing charts or their gallery.

## Why these outputs fit the outline and literature

The professor's Section 5 asks for label counts, sizes, distributions and tag
prevalence; Section 6 asks for baseline metrics and performance within recorded
conditions. Dataset validation is therefore presented separately from annotation
geometry, with explicit denominators for both.

[Yantén et al. (2023)](https://doi.org/10.3390/data8040069) provides a directly
relevant froth data-descriptor precedent for documenting source, visual metadata,
examples and baseline validation. Its classification labels do not supply our
per-bubble geometry validation. [Metrics Reloaded](https://doi.org/10.1038/s41592-023-02151-z)
and [Reinke et al.](https://doi.org/10.1038/s41592-023-02150-0) motivate matching
metrics to the target, declaring aggregation, and exposing distributions and
subgroups. The particular charts, sensitivity populations and table organization
here are our project-specific choices, not requirements asserted by those papers.

Relevant literature notes were revisited and accessible primary-source sections
were checked again for this task. No claim is made that every paper in the library
was reread in full. Acquisition history and physical calibration remain pending the
Sungun engineer's responses; the present analysis does not invent them.
"""
    (OUT/"README.md").write_text(readme,encoding="utf-8")
    lines=["# Numerical notes for the professor's Sections 5 and 6", "",
           "These are descriptive reporting notes, not a final manuscript or a controlled model ranking.", "",
           "## Section 5: dataset and annotation statistics", "",
           f"The manifest contains **{len(manifest)} images**: 485 source-training and 110 source-test images (reused validation). "
           "There are 340 colour-domain and 255 grayscale-domain images. Native geometries are 1440×1080 (338), 1448×1084 (2), and 640×480 (255).", "",
           f"The saved annotations contain **{len(objects):,} froth shapes** ({all_inventory['polygon_annotations']:,} polygons and "
           f"{all_inventory['circle_annotations']} circle). Valid-size measurements are available for **{len([o for o in objects if o['valid']]):,}**. "
           f"The geometry check flagged **430 shapes ({all_inventory['invalid_geometry_pct_annotations']:.3f}%) across {all_inventory['images_with_invalid_geometry']} images** "
           "as self-intersecting or self-touching; these shapes are included in annotation counts and excluded from size calculations. "
           "A flag is a conservative review finding, not an independent judgement that an annotation is unusable.", "",
           f"Annotations per image: mean **{all_inventory['froth_count_per_image_mean']:.2f}**, "
           f"sample SD {all_inventory['froth_count_per_image_std']:.2f}, median {all_inventory['froth_count_per_image_median']:.0f}, "
           f"IQR {all_inventory['froth_count_per_image_q25']:.0f}–{all_inventory['froth_count_per_image_q75']:.0f}, "
           f"range {all_inventory['froth_count_per_image_min']:.0f}–{all_inventory['froth_count_per_image_max']:.0f}.", "",
           f"On all valid geometries, the **object-weighted mean area is {area['mean']:,.2f} px²**, "
           f"with median {area['median']:,.2f} px² and IQR {area['q25']:,.2f}–{area['q75']:,.2f} px². "
           f"The **image-weighted mean area is {area['image_weighted_mean']:,.2f} px²**. "
           f"Equivalent circular diameter has mean {diameter['mean']:.2f} px and median {diameter['median']:.2f} px. "
           "The pooled means describe mixed native resolutions; they are not physical measurements or proof of domain-specific physical size differences.", "",
           f"Sensitivity descriptions: excluding the three outside-frame valid annotations leaves {inside['n_annotations']:,} sizes "
           f"and mean area {inside['mean']:,.2f} px²; excluding the border proxy leaves {interior['n_annotations']:,} "
           f"and mean area {interior['mean']:,.2f} px². Border annotations total 11,284, affecting 591 images. "
           "This subset has a different object population and is not automatically a superior estimate.", "",
           f"Polygon tracing detail: median {all_inventory['polygon_vertices_median']:.0f} vertices "
           f"(IQR {all_inventory['polygon_vertices_q25']:.0f}–{all_inventory['polygon_vertices_q75']:.0f}; "
           f"range {all_inventory['polygon_vertices_min']}–{all_inventory['polygon_vertices_max']}). "
           "This measures representation detail, not annotation accuracy. Eight stale internal JSON image paths and two unlisted JSON files are already disclosed in the original run report.", "",
           "Image-level collapse appearance is present in 123 images; it cannot supply the number of collapse events or a temporal rate. "
           "Do not call 2D areas volumes. Neither flow velocity, recovery, copper grade nor optimal reagent doses can be inferred from these annotation CSVs alone.", "",
           "## Section 6: model averages and tag comparisons", "",
           "All 14 source tag CSVs have a corresponding averages CSV and Dice/IoU chart. The combined table contains "
           f"{len(group_rows)} rows = 14 groups × 9 runs × 3 split scopes. Additional within-domain tables retain sparse/empty strata. "
           "The `all` scope answers the whole-group average request; manuscript validation summaries should use the separately disclosed reused-validation scope.", "",
           "Recorded reused-validation means (110 scored images per run). Values below are percentages; they are not a controlled architecture ranking.", "",
           "| Recorded run | IoU | Dice | Pixel accuracy | Precision | Recall |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in overall:
        if r["source_split"]=="test" and r["domain"]=="all":
            lines.append(f"| {MODEL_LABELS[r['model']]} | "+" | ".join(f"{100*r[m+'_mean']:.2f}" for m in ("iou","dice","pixel_acc","precision","recall"))+" |")
    lines += ["", "*DeepLab and RobustSAM labels require CSV-producing run provenance. All means come from the saved per-image metric values; no resizing or re-evaluation was performed.*", "",
              "Tag-group image counts and scored counts (same image coverage for all nine runs):", "",
              "| Tag CSV | All images | Train images / scored | Reused-validation images / scored |", "| --- | ---: | ---: | ---: |"]
    gi={(r["tag_group"],r["model"],r["source_split"]):r for r in group_rows}
    for group in GROUP_ORDER:
        a,t,v=(gi[group,"unet",s] for s in SPLITS)
        lines.append(f"| {group} | {a['n_images']} | {t['n_images']} / {t['n_scored']} | {v['n_images']} / {v['n_scored']} |")
    lines += ["", "The six-or-more-condition group has **no validation images**, so its validation means are N/A. "
              "The condition-count groups exclude the domain tag and are descriptive composites. "
              "Reflection is strongly confounded with split/domain: all 48 positive validation images are colour, "
              "whereas all 50 grayscale validation images are negative; 199 of 205 grayscale training images are positive. "
              "This could reflect acquisition differences or inconsistent criteria and requires checking before interpretation.", "",
              "The foreground-reference table shows high constant-foreground scores on each run's own GT grid. "
              "Report those references when interpreting region overlap; they do not provide boundary/instance validation. "
              "Use the tag/domain contrast charts to describe observed associations, without claiming a tag caused a performance change.", "",
              "## Suggested compact main-paper selection", "",
              "Use one dataset-inventory table, a multipanel composition/count/size figure, one baseline summary table, "
              "and the reused-validation tag heatmap. Add the distribution/domain view or the foreground reference where needed "
              "to explain evaluation limits. Keep the 14 individual tag charts and large CSVs as repository/supplement material "
              "according to the eventual journal format. The charts are candidates; do not insert every chart into the article.", "",
              "The engineer's acquisition metadata and the team's run/annotation provenance are still needed for the Methods. "
              "Boundary/instance metrics require matched masks and an agreed target; they cannot be recovered from these per-image aggregate CSVs.", ""]
    (OUT/"paper_notes.md").write_text("\n".join(lines),encoding="utf-8")
    if FIGURES:
        captions=["# Figure candidates and captions", "", "PNG files are 300 dpi; PDFs preserve vector text and plot elements (dense point layers may be rasterized).", "",
                  "Candidate numbering follows the professor's sections rather than fixing final manuscript figure numbers.", ""]
        gallery=["<!doctype html><html lang='en'><meta charset='utf-8'><title>Paper evaluations</title>",
                 "<style>body{font:16px system-ui;max-width:1400px;margin:30px auto;padding:0 24px;color:#20262e}"
                 "img{max-width:100%;border:1px solid #ddd}figure{margin:35px 0 50px}figcaption{max-width:1000px;line-height:1.5}"
                 "a{color:#125c9c}h1{font-size:30px}p{line-height:1.5}</style>",
                 "<h1>Dataset statistics and tag evaluations</h1>",
                 "<p>Professor's Sections 5 and 6. <a href='README.md'>Definitions</a> · <a href='paper_notes.md'>Numerical notes</a> · "
                 "<a href='tags/model_averages_by_tag.csv'>All tag averages</a></p>",
                 "<p>Source test = reused validation. Recorded runs have different evaluation grids/crops; these charts do not establish a controlled model ranking.</p>"]
        for item in FIGURES:
            name=item["name"]
            title=Path(name).name.replace("_"," ")
            captions += [f"## {title}", "", f"[PNG]({name}.png) · [PDF]({name}.pdf)", "", item["caption"], ""]
            gallery.append(f"<figure><h2>{html.escape(title)}</h2><a href='{html.escape(name)}.pdf'>Vector PDF</a> · "
                           f"<a href='{html.escape(name)}.png'>300-dpi PNG</a><br><img loading='lazy' src='{html.escape(name)}.png' "
                           f"alt='{html.escape(title)}'><figcaption>{html.escape(item['caption'])}</figcaption></figure>")
        gallery.append("</html>")
        (OUT/"figure_captions.md").write_text("\n".join(captions),encoding="utf-8")
        (OUT/"index.html").write_text("\n".join(gallery),encoding="utf-8")
        (OUT/"requirements.txt").write_text("numpy=="+versions["numpy"]+"\nmatplotlib=="+versions["matplotlib"]+"\n",encoding="utf-8")


def main(argv=None):
    global ROOT, OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-only",action="store_true",help="Skip plotting; no third-party dependencies required")
    parser.add_argument("--project-root",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args(argv)
    ROOT,OUT=args.project_root.resolve(),args.output_dir.resolve()
    if OUT == ROOT:
        parser.error("Output must be separate from the project input directory")
    OUT.mkdir(parents=True,exist_ok=True)
    CSV_OUTPUTS.clear()
    FIGURES.clear()
    print("1/6 Reading and verifying saved CSV sources...",flush=True)
    manifest,tags,conditions,groups,models,images,objects,by_image,run,hashes,error=load()
    print(f"    {len(manifest)} images; {len(objects):,} annotations; {len(models)} runs; {len(groups)} groups",flush=True)
    print("2/6 Averaging all seven model metrics for every tag CSV...",flush=True)
    tables=tag_and_model_tables(manifest,tags,conditions,groups,models)
    print("3/6 Preparing Section 5 count, size and quality summaries...",flush=True)
    stats=statistical_tables(manifest,groups,images,objects,by_image)
    print("4/6 Cross-checking previous reports and split-weighted means...",flush=True)
    checks=validate_against_previous(manifest,tables[0],tables[3],stats[0],stats[1])
    versions={}
    if not args.csv_only:
        print("5/6 Drawing Section 5/6 figures and all 14 tag charts...",flush=True)
        versions=plot_all(manifest,conditions,groups,models,images,objects,tables,stats)
    else:
        print("5/6 Plotting skipped (--csv-only)",flush=True)
    print("6/6 Saving reporting notes and source verification...",flush=True)
    write_documents(manifest,conditions,groups,models,images,objects,run,tables,stats,checks,versions)
    for relative,digest in hashes.items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest, f"Source changed: {relative}"
    report={"status":"complete","generated_utc":datetime.now(timezone.utc).isoformat(),
            "analysis_scope":"professor_outline_sections_5_and_6",
            "model_runs":len(models),"tag_groups":len(groups),"manifest_images":len(manifest),
            "froth_annotations":len(objects),"valid_size_annotations":sum(o["valid"] for o in objects),
            "missing_training_results_per_run":{m:[n for n in manifest if n not in r] for m,r in models.items()},
            "max_metric_error_vs_confusion_formula":error,"checks":checks,"source_sha256":hashes,
            "sources_unchanged":True,"csv_outputs":CSV_OUTPUTS,"figures_regenerated":not args.csv_only,
            "figure_candidates":len(FIGURES),"figure_files":len(FIGURES)*2,"plotting_versions":versions,
            "no_new_model_runs":True,"no_json_geometry_traversal":True,
            "original_statistics_warnings":run["warnings"],
            "excluded_unlisted_json_files":run["excluded_unlisted_json_files"]}
    (OUT/"verification.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(f"Done: {len(CSV_OUTPUTS)} CSVs; {len(FIGURES)} figure candidates (PNG + PDF).",flush=True)
    print("Verified input preservation and independent agreement with existing summary tables.",flush=True)


if __name__=="__main__":
    main()
