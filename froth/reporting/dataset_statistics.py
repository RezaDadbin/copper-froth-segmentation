#!/usr/bin/env python3
"""Measure LabelMe froth annotations using only the Python standard library.

Run without arguments for the complete tags.csv manifest. --test-five measures
exactly five real annotations and writes to test_5_froths/, never the main reports.
"""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time


PROJECT = Path(__file__).resolve().parents[2] / "data"
VERSION = "1.0"
METRIC_FIELDS = [
    "area_px2", "area_pct_image", "equivalent_diameter_px", "perimeter_px",
    "compactness", "centroid_x_px", "centroid_y_px", "bbox_xmin_px",
    "bbox_ymin_px", "bbox_xmax_px", "bbox_ymax_px", "bbox_width_px",
    "bbox_height_px", "bbox_aspect_ratio", "bbox_fill_fraction",
]
FROTH_FIELDS = [
    "image_name", "source_split", "domain", "measurement_scope", "json_file",
    "shape_index", "label", "group_id", "shape_type", "input_point_count",
    "geometry_valid", "issue", "outside_image", "within_1px_of_border",
    *METRIC_FIELDS,
]


def quantile(values, probability):
    """Linear interpolation between sorted observations, index (n - 1) * p."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def describe(values, prefix, unit):
    if not values:
        return {f"{prefix}_{name}_{unit}": None for name in
                ("mean", "median", "std", "min", "p05", "p25", "p75", "p95", "max")}
    return {
        f"{prefix}_mean_{unit}": statistics.fmean(values),
        f"{prefix}_median_{unit}": statistics.median(values),
        f"{prefix}_std_{unit}": statistics.pstdev(values),
        f"{prefix}_min_{unit}": min(values),
        f"{prefix}_p05_{unit}": quantile(values, 0.05),
        f"{prefix}_p25_{unit}": quantile(values, 0.25),
        f"{prefix}_p75_{unit}": quantile(values, 0.75),
        f"{prefix}_p95_{unit}": quantile(values, 0.95),
        f"{prefix}_max_{unit}": max(values),
    }


def orientation(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def on_segment(a, b, p):
    return (min(a[0], b[0]) - 1e-9 <= p[0] <= max(a[0], b[0]) + 1e-9
            and min(a[1], b[1]) - 1e-9 <= p[1] <= max(a[1], b[1]) + 1e-9)


def segments_intersect(a, b, c, d):
    if (max(a[0], b[0]) < min(c[0], d[0]) - 1e-9
            or max(c[0], d[0]) < min(a[0], b[0]) - 1e-9
            or max(a[1], b[1]) < min(c[1], d[1]) - 1e-9
            or max(c[1], d[1]) < min(a[1], b[1]) - 1e-9):
        return False
    o1, o2 = orientation(a, b, c), orientation(a, b, d)
    o3, o4 = orientation(c, d, a), orientation(c, d, b)
    if ((o1 > 1e-9 and o2 < -1e-9) or (o1 < -1e-9 and o2 > 1e-9)) and (
            (o3 > 1e-9 and o4 < -1e-9) or (o3 < -1e-9 and o4 > 1e-9)):
        return True
    return any(abs(o) <= 1e-9 and on_segment(u, v, p) for o, u, v, p in
               [(o1, a, b, c), (o2, a, b, d), (o3, c, d, a), (o4, c, d, b)])


def polygon_is_simple(points):
    n = len(points)
    # Adjacent edges may share their endpoint, but may not double back/overlap.
    for i, b in enumerate(points):
        a, c = points[i - 1], points[(i + 1) % n]
        if abs(orientation(a, b, c)) <= 1e-9:
            if (a[0] - b[0]) * (c[0] - b[0]) + (a[1] - b[1]) * (c[1] - b[1]) > 1e-9:
                return False
    for i in range(n):
        a, b = points[i], points[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if segments_intersect(a, b, points[j], points[(j + 1) % n]):
                return False
    return True


def measure_shape(shape, width, height):
    result = {field: None for field in METRIC_FIELDS}
    result.update(geometry_valid=0, issue="", outside_image=None,
                  within_1px_of_border=None, input_point_count=len(shape.get("points") or []))
    shape_type = shape.get("shape_type") or "polygon"
    try:
        raw = shape.get("points") or []
        if any(not isinstance(p, (list, tuple)) or len(p) != 2 for p in raw):
            raise ValueError("malformed_point")
        points = [(float(p[0]), float(p[1])) for p in raw]
        if any(not math.isfinite(v) for p in points for v in p):
            raise ValueError("nonfinite_coordinate")
        if shape_type == "circle":
            if len(points) != 2:
                raise ValueError("circle_requires_two_points")
            (cx, cy), edge = points
            radius = math.dist(points[0], edge)
            if radius <= 0:
                raise ValueError("zero_radius")
            area, perimeter = math.pi * radius ** 2, 2 * math.pi * radius
            xmin, xmax, ymin, ymax = cx - radius, cx + radius, cy - radius, cy + radius
        elif shape_type == "polygon":
            clean = []
            for point in points:
                if not clean or point != clean[-1]:
                    clean.append(point)
            if len(clean) > 1 and clean[0] == clean[-1]:
                clean.pop()
            points = clean
            if len(points) < 3:
                raise ValueError("fewer_than_three_distinct_points")
            if not polygon_is_simple(points):
                raise ValueError("self_intersection_or_self_touch")
            # Translate first to avoid loss of precision in the shoelace formula.
            ox, oy = points[0]
            shifted = [(x - ox, y - oy) for x, y in points]
            pairs = list(zip(shifted, shifted[1:] + shifted[:1]))
            crosses = [a[0] * b[1] - b[0] * a[1] for a, b in pairs]
            twice_signed_area = math.fsum(crosses)
            area = abs(twice_signed_area) / 2
            if area <= 1e-12:
                raise ValueError("zero_or_numerically_degenerate_area")
            perimeter = math.fsum(math.dist(a, b) for a, b in pairs)
            cx = ox + math.fsum((a[0] + b[0]) * c for (a, b), c in zip(pairs, crosses)) / (3 * twice_signed_area)
            cy = oy + math.fsum((a[1] + b[1]) * c for (a, b), c in zip(pairs, crosses)) / (3 * twice_signed_area)
            xmin, xmax = min(p[0] for p in points), max(p[0] for p in points)
            ymin, ymax = min(p[1] for p in points), max(p[1] for p in points)
        else:
            raise ValueError("unsupported_shape_type:" + str(shape_type))
        bw, bh = xmax - xmin, ymax - ymin
        if not all(math.isfinite(x) for x in [area, perimeter, cx, cy, bw, bh]) or bw <= 0 or bh <= 0:
            raise ValueError("nonfinite_or_degenerate_geometry")
        result.update(
            geometry_valid=1, area_px2=area, area_pct_image=100 * area / (width * height),
            equivalent_diameter_px=math.sqrt(4 * area / math.pi), perimeter_px=perimeter,
            compactness=4 * math.pi * area / perimeter ** 2,
            centroid_x_px=cx, centroid_y_px=cy,
            bbox_xmin_px=xmin, bbox_ymin_px=ymin, bbox_xmax_px=xmax, bbox_ymax_px=ymax,
            bbox_width_px=bw, bbox_height_px=bh,
            bbox_aspect_ratio=max(bw, bh) / min(bw, bh), bbox_fill_fraction=area / (bw * bh),
            outside_image=int(xmin < -1e-7 or ymin < -1e-7 or xmax > width + 1e-7 or ymax > height + 1e-7),
            within_1px_of_border=int(xmin <= 1 or ymin <= 1 or xmax >= width - 1 or ymax >= height - 1),
        )
    except (TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
        result["issue"] = str(exc)
    return result


def read_manifest(tags_path, dataset_root):
    records, names = [], set()
    with tags_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"split", "image_name", "domain_yellow_copper"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("tags.csv requires columns: " + ", ".join(sorted(required)))
        for row in reader:
            name, split, domain = row["image_name"].strip(), row["split"].strip(), row["domain_yellow_copper"].strip()
            if not name or name in names or Path(name).name != name or "\\" in name:
                raise ValueError("Empty, duplicate, or unsafe image_name: " + repr(name))
            if split not in {"train", "test"} or domain not in {"0", "1"}:
                raise ValueError("Unexpected split/domain for " + name)
            folder = dataset_root / ("train1" if split == "train" else "test1")
            records.append(dict(image_name=name, source_split=split,
                                domain="yellow_copper" if domain == "1" else "grayscale",
                                image_file=folder / name, json_file=folder / Path(name).with_suffix(".json")))
            names.add(name)
    if not records:
        raise ValueError("Empty manifest")
    paths = [r["json_file"] for r in records]
    if len(set(paths)) != len(paths):
        raise ValueError("Multiple manifest images refer to the same JSON")
    return records


def load_document(record):
    if not record["image_file"].is_file():
        raise FileNotFoundError("Missing paired image: " + str(record["image_file"]))
    data = json.loads(record["json_file"].read_text(encoding="utf-8-sig"))
    width, height = data.get("imageWidth"), data.get("imageHeight")
    if (not isinstance(width, int) or not isinstance(height, int)
            or isinstance(width, bool) or isinstance(height, bool) or width <= 0 or height <= 0):
        raise ValueError("Missing/invalid JSON image dimensions")
    if not isinstance(data.get("shapes"), list):
        raise ValueError("JSON shapes is not a list")
    return data


def is_froth(shape):
    return isinstance(shape, dict) and str(shape.get("label", "")).strip().lower() == "froth"


def test_selection(records):
    """Four split/domain combinations plus the known paired circle annotation."""
    choices = []
    for split in ("train", "test"):
        for domain in ("yellow_copper", "grayscale"):
            record = next(r for r in records if r["source_split"] == split and r["domain"] == domain)
            data = load_document(record)
            index = next(i for i, shape in enumerate(data["shapes"])
                         if is_froth(shape) and (shape.get("shape_type") or "polygon") == "polygon")
            choices.append((record, data, [index]))
    record = next((r for r in records if r["json_file"].name == "S40 (12).json"), None)
    if record is None:
        raise ValueError("The fifth test expects the paired file S40 (12).json")
    data = load_document(record)
    index = next(i for i, shape in enumerate(data["shapes"])
                 if is_froth(shape) and shape.get("shape_type") == "circle")
    choices.append((record, data, [index]))
    return choices


def analytic_checks():
    """Known areas/perimeters and failure cases, independent of this dataset."""
    fixtures = [
        ("square", "polygon", [[2, 2], [12, 2], [12, 12], [2, 12]], 100, 40, (7, 7)),
        ("reversed_square", "polygon", [[2, 12], [12, 12], [12, 2], [2, 2]], 100, 40, (7, 7)),
        ("right_triangle", "polygon", [[1, 1], [4, 1], [1, 5]], 6, 12, (2, 7 / 3)),
        ("concave_L", "polygon", [[1, 1], [5, 1], [5, 2], [2, 2], [2, 5], [1, 5]], 7, 16, (33 / 14, 33 / 14)),
        ("circle", "circle", [[10, 10], [13, 14]], 25 * math.pi, 10 * math.pi, (10, 10)),
        ("closed_duplicate_square", "polygon", [[2, 2], [12, 2], [12, 2], [12, 12], [2, 12], [2, 2]], 100, 40, (7, 7)),
    ]
    for name, kind, points, area, perimeter, centroid in fixtures:
        result = measure_shape({"shape_type": kind, "points": points}, 100, 100)
        assert result["geometry_valid"], (name, result)
        for key, expected in [("area_px2", area), ("perimeter_px", perimeter),
                              ("centroid_x_px", centroid[0]), ("centroid_y_px", centroid[1])]:
            assert math.isclose(result[key], expected, rel_tol=1e-11, abs_tol=1e-10), (name, key, result[key], expected)
    invalid = [
        {"shape_type": "polygon", "points": [[0, 0], [4, 4], [0, 4], [4, 0]]},
        {"shape_type": "polygon", "points": [[0, 0], [1, 1], [2, 2]]},
        {"shape_type": "polygon", "points": [[0, 0], [1, 1]]},
        {"shape_type": "circle", "points": [[2, 2], [2, 2]]},
        {"shape_type": "polygon", "points": [[0, 0], [float("nan"), 1], [1, 0]]},
    ]
    for shape in invalid:
        assert not measure_shape(shape, 100, 100)["geometry_valid"], shape
    assert quantile([1, 2, 3, 4], 0.25) == 1.75
    assert math.isclose(statistics.pstdev([1, 2, 3, 4]), math.sqrt(1.25))
    return {"analytic_geometry_cases_passed": len(fixtures),
            "invalid_geometry_cases_passed": len(invalid), "statistic_checks_passed": 2}


def time_string(seconds):
    minutes, seconds = divmod(max(0, int(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def grouping_keys(image):
    return [("all", "all"), ("source_split", image["source_split"]),
            ("domain", image["domain"]),
            ("split_domain", image["source_split"] + "|" + image["domain"]),
            ("image_geometry", f'{image["image_width_px"]}x{image["image_height_px"]}')]


def make_summaries(image_rows, objects_by_image, scope):
    groups = defaultdict(list)
    for image in image_rows:
        for key in grouping_keys(image):
            groups[key].append(image)
    summaries, distributions = [], []
    for (kind, value), images in sorted(groups.items()):
        objects = [obj for image in images for obj in objects_by_image[image["image_name"]]]
        valid = [obj for obj in objects if obj["geometry_valid"]]
        inside = [obj for obj in valid if not obj["outside_image"]]
        areas = [obj["area_px2"] for obj in valid]
        normalized = [obj["area_pct_image"] for obj in valid]
        diameters = [obj["equivalent_diameter_px"] for obj in valid]
        counts = [image["froth_count"] for image in images]
        row = dict(group_type=kind, group_value=value, measurement_scope=scope,
                   image_count=len(images), froth_count=len(objects), valid_size_count=len(valid),
                   invalid_geometry_count=len(objects) - len(valid),
                   outside_image_count=sum(obj["outside_image"] for obj in valid),
                   within_1px_of_border_count=sum(obj["within_1px_of_border"] for obj in valid),
                   froth_count_mean_per_image=statistics.fmean(counts),
                   froth_count_median_per_image=statistics.median(counts),
                   froth_count_min_per_image=min(counts), froth_count_max_per_image=max(counts),
                   object_weighted_area_mean_px2=statistics.fmean(areas) if areas else None,
                   image_weighted_area_mean_px2=statistics.fmean(
                       image["area_mean_px2"] for image in images if image["valid_size_count"])
                       if valid else None,
                   inside_image_area_mean_px2=statistics.fmean(obj["area_px2"] for obj in inside) if inside else None)
        row.update(describe(areas, "area", "px2"))
        row.update(describe(diameters, "equivalent_diameter", "px"))
        row.update(describe(normalized, "area", "pct_image"))
        summaries.append(row)
        for measure, values in [("area_px2", areas), ("area_pct_image", normalized)]:
            bins = Counter(math.floor(math.log2(number)) for number in values if number > 0)
            cumulative = 0
            if not bins:
                continue
            for exponent in range(min(bins), max(bins) + 1):
                count = bins[exponent]
                cumulative += count
                lower, upper = 2.0 ** exponent, 2.0 ** (exponent + 1)
                distributions.append(dict(
                    group_type=kind, group_value=value, measurement_scope=scope, measure=measure,
                    bin_lower_inclusive=lower, bin_upper_exclusive=upper,
                    froth_count=count, total_valid_froths=len(values), fraction=count / len(values),
                    cumulative_fraction=cumulative / len(values),
                    probability_density=count / (len(values) * (upper - lower))))
    return summaries, distributions


def run(args):
    started = time.monotonic()
    checks = analytic_checks()
    tags_path, dataset_root = args.tags.resolve(), args.dataset_root.resolve()
    records = read_manifest(tags_path, dataset_root)
    output = args.output_dir.resolve()
    if args.test_five:
        output = output / "test_5_froths"
    output.mkdir(parents=True, exist_ok=True)
    scope = "TEST_ONLY_five_selected_annotations" if args.test_five else "all_manifest_froth_annotations"
    choices = test_selection(records) if args.test_five else [(r, None, None) for r in records]
    print(f"{'TEST: EXACTLY FIVE ANNOTATIONS' if args.test_five else 'FULL DATASET RUN'} | manifest: {len(records)} images", flush=True)
    print(f"Output: {output}", flush=True)
    print("Units: px, px²; no physical volume. Counts describe annotation shapes.", flush=True)
    if args.test_five:
        print("Per-image test statistics cover selected shapes only, not all bubbles in those images.", flush=True)
    image_rows, all_objects, objects_by_image, sizes = [], [], {}, {}
    failures, warnings = [], []
    for number, (record, cached, selected) in enumerate(choices, 1):
        try:
            data = cached if cached is not None else load_document(record)
            width, height, shapes = data["imageWidth"], data["imageHeight"], data["shapes"]
            bad_shapes = [i for i, shape in enumerate(shapes) if not isinstance(shape, dict)]
            if bad_shapes:
                raise ValueError("Non-object shapes at indices " + str(bad_shapes))
            image_path_name = str(data.get("imagePath") or "").replace("\\", "/").rsplit("/", 1)[-1]
            path_matches = image_path_name == record["image_name"]
            if not path_matches:
                warnings.append(dict(image_name=record["image_name"], issue="stale_or_missing_JSON_imagePath",
                                     json_imagePath=data.get("imagePath")))
            indices = selected if selected is not None else [i for i, s in enumerate(shapes) if is_froth(s)]
            objects = []
            for index in indices:
                shape = shapes[index]
                obj = {key: record[key] for key in ("image_name", "source_split", "domain")}
                obj.update(measurement_scope=scope, json_file=str(record["json_file"].relative_to(dataset_root)),
                           shape_index=index, label=shape.get("label"), group_id=shape.get("group_id"),
                           shape_type=shape.get("shape_type") or "polygon")
                obj.update(measure_shape(shape, width, height))
                objects.append(obj)
            valid = [obj for obj in objects if obj["geometry_valid"]]
            areas = [obj["area_px2"] for obj in valid]
            row = {key: record[key] for key in ("image_name", "source_split", "domain")}
            row.update(
                measurement_scope=scope, image_width_px=width, image_height_px=height,
                image_area_px2=width * height, json_imagePath_matches=path_matches,
                total_shapes_in_json=len(shapes), total_froth_annotations_in_json=sum(is_froth(s) for s in shapes),
                non_froth_shape_count=sum(not is_froth(s) for s in shapes),
                froth_count=len(objects), valid_size_count=len(valid), invalid_geometry_count=len(objects) - len(valid),
                polygon_count=sum(obj["shape_type"] == "polygon" for obj in objects),
                circle_count=sum(obj["shape_type"] == "circle" for obj in objects),
                outside_image_count=sum(obj["outside_image"] for obj in valid),
                within_1px_of_border_count=sum(obj["within_1px_of_border"] for obj in valid),
                sum_individual_areas_px2=math.fsum(areas),
                sum_individual_area_pct_image=100 * math.fsum(areas) / (width * height))
            row.update(describe(areas, "area", "px2"))
            row.update(describe([obj["equivalent_diameter_px"] for obj in valid], "equivalent_diameter", "px"))
            row.update(describe([obj["area_pct_image"] for obj in valid], "area", "pct_image"))
            row["area_coefficient_of_variation"] = statistics.pstdev(areas) / statistics.fmean(areas) if areas else None
            row["mean_perimeter_px"] = statistics.fmean(obj["perimeter_px"] for obj in valid) if valid else None
            row["mean_compactness"] = statistics.fmean(obj["compactness"] for obj in valid) if valid else None
            image_rows.append(row)
            all_objects.extend(objects)
            objects_by_image[record["image_name"]] = objects
            # Parallel arrays keep their original shape-index alignment. Invalid = null.
            sizes[record["image_name"]] = dict(
                source_split=record["source_split"], domain=record["domain"],
                image_width_px=width, image_height_px=height, measurement_scope=scope,
                shape_indices=[obj["shape_index"] for obj in objects],
                areas_px2=[obj["area_px2"] for obj in objects],
                equivalent_diameters_px=[obj["equivalent_diameter_px"] for obj in objects],
                geometry_valid=[obj["geometry_valid"] for obj in objects],
                outside_image=[obj["outside_image"] for obj in objects])
            status = f'{len(objects)} froths, {len(valid)} valid sizes, {len(objects) - len(valid)} invalid'
        except (OSError, ValueError, TypeError, KeyError) as exc:
            failures.append(dict(image_name=record["image_name"], issue=str(exc)))
            status = "ERROR: " + str(exc)
        elapsed = time.monotonic() - started
        eta = elapsed / number * (len(choices) - number)
        print(f'[{number:03d}/{len(choices):03d}] {100 * number / len(choices):5.1f}% | {record["image_name"]} | {status} | elapsed {time_string(elapsed)} | ETA {time_string(eta)}', flush=True)
    summaries, distributions = make_summaries(image_rows, objects_by_image, scope)
    print("Writing CSV/JSON reports...", flush=True)
    write_csv(output / "froth_measurements.csv", all_objects, FROTH_FIELDS)
    write_csv(output / "image_statistics.csv", image_rows)
    write_csv(output / "dataset_summary.csv", summaries)
    write_csv(output / "area_distribution.csv", distributions)
    (output / "froth_sizes_by_image.json").write_text(json.dumps(sizes, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    used = {record["json_file"] for record in records}
    # File names only: extra JSONs are never parsed or counted.
    unused = sorted(str(p.relative_to(dataset_root)) for sub in ("train1", "test1")
                    for p in (dataset_root / sub).glob("*.json") if p not in used)
    counts = Counter(obj["issue"] for obj in all_objects if not obj["geometry_valid"])
    report = dict(
        script_version=VERSION, timestamp_utc=datetime.now(timezone.utc).isoformat(),
        mode="test_five_annotations" if args.test_five else "full_dataset", measurement_scope=scope,
        tags_csv=str(tags_path), tags_csv_sha256=hashlib.sha256(tags_path.read_bytes()).hexdigest(),
        dataset_root=str(dataset_root), manifest_image_count=len(records), requested_image_count=len(choices),
        processed_image_count=len(image_rows), measured_froth_count=len(all_objects),
        valid_size_count=sum(obj["geometry_valid"] for obj in all_objects),
        invalid_geometry_counts_by_reason=dict(counts), failures=failures, warnings=warnings,
        excluded_unlisted_json_files=unused, elapsed_seconds=time.monotonic() - started,
        analytic_checks=checks, inputs_modified=False,
        status="incomplete_input_errors" if failures else ("test_only" if args.test_five else "complete"))
    try:
        validate_exports(output, image_rows, all_objects, summaries, distributions, args.test_five)
    except (AssertionError, ValueError, OSError) as exc:
        report["status"] = "failed_export_validation"
        report["export_validation_error"] = f"{type(exc).__name__}: {exc}"
        (output / "run_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        raise
    report["export_consistency_checks_passed"] = True
    (output / "run_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f'Done: {len(image_rows)}/{len(choices)} images; {len(all_objects)} froth annotations; {report["valid_size_count"]} valid sizes.', flush=True)
    print(f'Invalid geometries: {sum(counts.values())}; metadata warnings: {len(warnings)}; excluded unlisted JSON files: {len(unused)}.', flush=True)
    if failures:
        print("INCOMPLETE: see failures in run_report.json. Fix inputs and rerun.", file=sys.stderr)
        return 1
    print("Export consistency checks passed.", flush=True)
    return 0


def validate_exports(output, images, objects, summaries, distributions, test_five):
    """Check the saved artifacts, not just the in-memory calculations."""
    with (output / "froth_measurements.csv").open(newline="", encoding="utf-8") as handle:
        saved = list(csv.DictReader(handle))
    saved_by_image = defaultdict(list)
    for row in saved:
        saved_by_image[row["image_name"]].append(row)
    assert len(saved) == len(objects) == sum(image["froth_count"] for image in images)
    assert len({(row["image_name"], row["shape_index"]) for row in saved}) == len(objects)
    mapping = json.loads((output / "froth_sizes_by_image.json").read_text())
    assert set(mapping) == {image["image_name"] for image in images}
    for image in images:
        entry = mapping[image["image_name"]]
        selected = saved_by_image[image["image_name"]]
        assert len(entry["areas_px2"]) == len(selected) == image["froth_count"]
        assert entry["shape_indices"] == [int(row["shape_index"]) for row in selected]
        for area, row in zip(entry["areas_px2"], selected):
            assert (area is None and row["area_px2"] == "") or (area is not None and area == float(row["area_px2"]))
        valid_areas = [area for area in entry["areas_px2"] if area is not None]
        assert math.isclose(math.fsum(valid_areas), image["sum_individual_areas_px2"], rel_tol=1e-12, abs_tol=1e-9)
    if summaries:
        total = next(row for row in summaries if row["group_type"] == "all")
        assert total["froth_count"] == len(objects)
        assert total["image_count"] == len(images)
    bins = defaultdict(list)
    for row in distributions:
        bins[(row["group_type"], row["group_value"], row["measure"])].append(row)
    for rows in bins.values():
        assert sum(row["froth_count"] for row in rows) == rows[0]["total_valid_froths"]
        assert math.isclose(math.fsum(row["fraction"] for row in rows), 1, abs_tol=1e-12)
    if test_five:
        assert len(objects) == 5, "Test must measure exactly five real annotations"
        assert all(row["geometry_valid"] for row in objects)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", type=Path, default=PROJECT / "tags.csv")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT / "Dataset")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[2] / "results/generated/statistics")
    parser.add_argument("--test-five", action="store_true", help="Measure exactly five real shapes; save in test_5_froths/.")
    args = parser.parse_args()
    try:
        return run(args)
    except (OSError, ValueError, AssertionError, StopIteration) as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
