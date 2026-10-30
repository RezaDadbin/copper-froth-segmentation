"""Supplementary review tables: all 68 reviewed records, and the eligible studies that could not be tabulated."""

import json

from paper.paths import REVIEW, TABLES


def esc(x):
    x = str(x)
    for a, b in [
        ("\\", ""),
        ("&", "\\&"),
        ("%", "\\%"),
        ("_", " "),
        ("#", "\\#"),
        ("$", "\\$"),
        ("~", "$\\sim$"),
        ("^", ""),
        ("{", ""),
        ("}", ""),
        ("–", "--"),
    ]:
        x = x.replace(a, b)
    return x.replace("∩", "$\\cap$").replace("∪", "$\\cup$")


def brk(x):
    return x.replace("/", "/\\allowbreak{}")


def short(x, n):
    x = esc(x).strip()
    if len(x) <= n:
        return brk(x)
    return brk(x[:n].rsplit(" ", 1)[0].rstrip(",;:(")) + "\\ldots"


def field(v):
    if isinstance(v, dict):
        return "; ".join(
            f"{k.replace('_', ' ')}: {x}" for k, x in v.items() if x not in (None, "not reported")
        )
    return v if v is not None else "not reported"


def all_studies(records):
    rows = []
    for r in records:
        d = r.get("data") or {}
        n = d.get("n_images") if isinstance(d, dict) else d
        rows.append(
            " & ".join(
                [
                    f"\\citet{{{r['key']}}}",
                    esc(r["category"]),
                    short(r["mineral_and_site"], 240),
                    short(n if n else "not reported", 260),
                    short(field(r["method"]), 420),
                    short(r.get("held_out_test") or "not reported", 160),
                    esc(r["evidence_level"]),
                ]
            )
            + " \\\\"
        )
    return "\n".join(rows) + "\n"


def not_tabulated(entries):
    rows = []
    for x in sorted(entries, key=lambda x: x["year"]):
        doi = "\\url{" + x["doi"].replace(" ", "_") + "}" if x.get("doi") else "--"
        rows.append(
            f"{brk(esc(x['title'])[:140])} & {x['year']} & {brk(esc(x.get('venue') or 'n.r.'))} & {doi} & {brk(esc(x['openalex_oa_status']))} \\\\"
        )
    return "\n".join(rows) + "\n"


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "all_studies.tex").write_text(all_studies(json.load(open(REVIEW / "records.json"))))
    (TABLES / "not_tabulated.tex").write_text(
        not_tabulated(json.load(open(REVIEW / "search.json"))["eligible_not_tabulated"])
    )
    print("written:", TABLES / "all_studies.tex", TABLES / "not_tabulated.tex")


if __name__ == "__main__":
    main()
