#!/usr/bin/env python3
"""
hit_call.py — strain hit calling on coverage_calls_lw.py output (*.coverage.tsv).

For every (strain, sample) row this reports

  strain | sample | hit_call_at_precision | hit_call_column_value |
  lineage_present | close_relative_present | lineage_id | gtdb_taxonomy |
  lineage_coverage

hit_call_at_precision
    Highest precision (PR row of the cutoffs table) whose cutoff is still met by
    the chosen hit-call column. NA if the value is below every cutoff.

lineage_present
    lineage_breadth >= --lineage-min-breadth (default 0.20).

lineage_coverage
    lineage_breadth itself: the fraction of the lineage's k-mers seen at least
    once, pooled over all of them. The value lineage_present is thresholding.

close_relative_present
    From the close-relative test in coverage_calls_lw.py (shared vs reference
    target blocks, quasi-Poisson rate ratio; see that script's docstring).
    Independent of lineage_present. Re-thresholded here so the cutoffs can be
    changed without re-running the coverage step:

        close_relative_present = p (or BH q with --relative-bh) < --relative-alpha
                                 and target_relative_rate_ratio >= --relative-min-ratio

    --relative-bh adjusts over all rows of all input files together.
    If the coverage table predates the test (no target_relative_pvalue), the
    precomputed target_close_relative is used if present, else NA.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl

# --hit-col shortcut -> (column in cutoffs table, column in coverage table)
HIT_COLUMNS = {
    "z1": ("coverage kept kmer blocks (z=1)", "target_kept_breadth"),
    "z2": ("coverage kept kmer blocks (z=2)", "target_kept_breadth"),
    "z3": ("coverage kept kmer blocks (z=3)", "target_kept_breadth"),
    "all": ("coverage all kmer blocks", "target_total_block_breadth"),
    "unique": ("coverage unique kmers", "target_breadth_unique"),
}

OUT_COLS = [
    "strain", "sample", "hit_call_at_precision", "hit_call_column_value",
    "lineage_present", "close_relative_present", "lineage_id", "gtdb_taxonomy",
    "lineage_coverage",
]
EXTRA_COLS = [
    "relative_unique_depth", "relative_nonunique_depth", "relative_rate_ratio",
    "relative_pvalue", "relative_qvalue",
]


def read_table(path: Path) -> pl.DataFrame:
    """Read a .tsv/.csv, sniffing the separator from the header line."""
    with open(path) as fh:
        header = fh.readline()
    sep = "\t" if header.count("\t") >= header.count(",") else ","
    return pl.read_csv(path, separator=sep, null_values=["NaN", "nan", "NA", ""],
                       infer_schema_length=10000)


def load_cutoffs(path: Path, cutoff_col: str) -> list[tuple[float, float]]:
    """Return [(precision, cutoff), ...] sorted by precision, highest first."""
    cut = read_table(path)
    if cutoff_col not in cut.columns:
        sys.exit(f"cutoff column '{cutoff_col}' not in {path}. "
                 f"Available: {cut.columns}")
    pr_col = cut.columns[0]  # 'PR'
    pairs = [(float(p), float(c)) for p, c in cut.select(pr_col, cutoff_col).iter_rows()]
    pairs.sort(key=lambda x: -x[0])
    # cutoffs should get looser as precision drops; warn if not
    for (p_hi, c_hi), (p_lo, c_lo) in zip(pairs, pairs[1:]):
        if c_lo > c_hi:
            print(f"warning: cutoff at PR {p_lo} ({c_lo}) > cutoff at PR {p_hi} ({c_hi})",
                  file=sys.stderr)
    return pairs


def precision_at(value, cutoffs) -> float | None:
    """Highest precision whose cutoff the value still meets."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    for pr, c in cutoffs:  # highest precision first
        if value >= c:
            return pr
    return None


def bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values; NaN stays NaN and is not counted."""
    q = np.full(pvals.shape, np.nan)
    ok = ~np.isnan(pvals)
    p = pvals[ok]
    if p.size == 0:
        return q
    order = np.argsort(p)
    ranked = p[order] * p.size / np.arange(1, p.size + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("coverage", nargs="+", type=Path, help="*.coverage.tsv file(s)")
    ap.add_argument("-c", "--cutoffs", type=Path, required=True,
                    help="cutoff table: PR column + one column per hit-call metric")
    ap.add_argument("-o", "--out", type=Path, default=None, help="output TSV (default stdout)")
    ap.add_argument("--hit-col", choices=HIT_COLUMNS, default="z2",
                    help="metric used for the hit call (default z2). The z1/z2/z3 "
                         "choices all read target_kept_breadth, so the coverage file "
                         "must come from a run with the matching --z.")
    ap.add_argument("--cutoff-col", default=None,
                    help="override: column name in the cutoffs table")
    ap.add_argument("--value-col", default=None,
                    help="override: column name in the coverage table")
    ap.add_argument("--lineage-min-breadth", type=float, default=0.20,
                    help="lineage_breadth needed to call lineage present (default 0.20)")
    ap.add_argument("--relative-alpha", type=float, default=0.01,
                    help="p (or q with --relative-bh) cutoff for close relative (default 0.01)")
    ap.add_argument("--relative-min-ratio", type=float, default=1.2,
                    help="min shared/reference depth ratio for close relative (default 1.2)")
    ap.add_argument("--relative-bh", action="store_true",
                    help="threshold Benjamini-Hochberg q-values instead of raw p-values")
    ap.add_argument("--extra", action="store_true",
                    help="also write the close-relative statistics columns")
    args = ap.parse_args()

    cutoff_col, value_col = HIT_COLUMNS[args.hit_col]
    cutoff_col = args.cutoff_col or cutoff_col
    value_col = args.value_col or value_col
    cutoffs = load_cutoffs(args.cutoffs, cutoff_col)

    cov = pl.concat([read_table(p) for p in args.coverage], how="diagonal_relaxed")
    if value_col not in cov.columns:
        sys.exit(f"value column '{value_col}' not in coverage table")

    def col_or_null(name, dtype=pl.Float64):
        return pl.col(name) if name in cov.columns else pl.lit(None, dtype=dtype)

    cov = cov.with_columns(
        col_or_null("target_relative_rate_ratio").cast(pl.Float64).alias("relative_rate_ratio"),
        col_or_null("target_relative_unique_depth").cast(pl.Float64).alias("relative_unique_depth"),
        col_or_null("target_relative_nonunique_depth").cast(pl.Float64).alias("relative_nonunique_depth"),
        col_or_null("target_relative_pvalue").cast(pl.Float64).alias("relative_pvalue"),
    )
    pvals = cov["relative_pvalue"].fill_null(np.nan).to_numpy().astype(float)
    cov = cov.with_columns(pl.Series("relative_qvalue", bh(pvals)).fill_nan(None))

    has_test = "target_relative_pvalue" in cov.columns
    if has_test:
        stat = "relative_qvalue" if args.relative_bh else "relative_pvalue"
        rel = (pl.when(pl.col(stat).is_null() | pl.col("relative_rate_ratio").is_null())
                 .then(None)
                 .otherwise((pl.col(stat) < args.relative_alpha)
                            & (pl.col("relative_rate_ratio") >= args.relative_min_ratio)))
    else:
        print("warning: no target_relative_pvalue in the coverage table (older "
              "coverage_calls_lw.py?); close_relative_present taken from "
              "target_close_relative if present, else NA", file=sys.stderr)
        rel = col_or_null("target_close_relative", pl.Boolean)

    value = pl.col(value_col).cast(pl.Float64)
    prec = pl.Series("hit_call_at_precision",
                     [precision_at(v, cutoffs) for v in cov[value_col].cast(pl.Float64)],
                     dtype=pl.Float64)

    out = (cov.with_columns(prec)
              .select(
                  pl.col("strain"),
                  pl.col("sample"),
                  pl.col("hit_call_at_precision"),
                  value.alias("hit_call_column_value"),
                  (col_or_null("lineage_breadth").cast(pl.Float64) >= args.lineage_min_breadth)
                    .fill_null(False).alias("lineage_present"),
                  rel.cast(pl.Boolean).alias("close_relative_present"),
                  col_or_null("lineage_call", pl.Utf8).alias("lineage_id"),
                  col_or_null("gtdb_tax", pl.Utf8).alias("gtdb_taxonomy"),
                  col_or_null("lineage_breadth").cast(pl.Float64)
                    .alias("lineage_coverage"),
                  *EXTRA_COLS,
              )
              .select(OUT_COLS + (EXTRA_COLS if args.extra else [])))

    if args.out:
        out.write_csv(args.out, separator="\t", null_value="NA")
    else:
        sys.stdout.write(out.write_csv(separator="\t", null_value="NA"))


if __name__ == "__main__":
    main()