"""Lineage and target strain calls from k-mer block coverage, LW model only.

BREADTH, DEPTH
    breadth = fraction of k-mers seen at least once   (the `cov` of the LW model)
    depth   = mean count per k-mer
    The LW model maps between them:  breadth = a * (1 - exp(-b * depth)).
    A block's `breadth` is its own fraction of k-mers seen; the sample-level
    breadth is the same measure pooled over all k-mers, i.e. the length-weighted
    mean of the block breadths.

LINEAGE
    The old rule (block hit = mean_depth >= 3 and breadth >= 0.95, then the
    fraction of blocks hit) was a hidden ~4x depth threshold, so a lineage
    present at 1x was missed even when the target strain was called. It is
    replaced by the same treatment the target already gets:

        lineage_breadth = seen lineage k-mers / all lineage k-mers
        lineage_depth   = lw_inverse(lineage_breadth)

    The expected fraction of a block's k-mers that is seen does not depend on
    the block's length -- only the variance does -- so pooling over all k-mers
    removes the block-length effect entirely. Pooled breadth is used rather than
    the unweighted mean of block breadths: same expectation, less noise, and it
    does not let a 31-k-mer block count as much as a 3000-k-mer one.

    lineage_depth is the summed depth of every strain of that lineage in the
    sample. Comparing it with the target's expected_depth separates "our strain"
    from "our strain plus a resident one".

TARGET
    Unchanged in substance: sample-level breadth over the informative target
    k-mers -> expected / max depth; blocks deeper than the sample allows are
    dropped. Threshold on target_breadth_unique, not on the block means: those
    collapse at low depth, because most blocks are then removed as too_deep and
    mostly the empty ones are left to average.

    Three target breadths are reported:
        target_breadth_unique       informative k-mers only (presence level 0 or 1)
        target_kept_breadth         all target k-mers, too_deep blocks removed
        target_total_block_breadth  all target k-mers, all blocks (no filtering)

INFORMATIVE K-MER SET (--min_unique_kmers, default 1000)
    presence_list records which other DB strains carry a target k-mer, so
    presence level 0 = unique to this strain, level 1 = shared with at most one
    other. Level 0 is preferred, but some strains have very few of them (123 for
    one B. adolescentis, 408 for a B. pseudocatenulatum), which makes breadth
    coarse (quantised in steps of 1/n) and easily knocked about by a single
    abundant relative. So the set falls back to level 1 when level 0 holds fewer
    than min_unique_kmers; target_presence_level records which was used.

DEPTH / BREADTH CONSISTENCY  (target_depth_ratio_p0 / _p1)
    observed depth of the informative k-mers, over the depth their own breadth
    implies. Both presence levels are always computed, whichever is used for the
    call.

    ~1        the hits look like a strain at that depth
    >> 1      few k-mers, each deep: hits from an abundant relative, not from
              this strain. A strain at breadth 0.31 should sit near 0.57x; 1.04x
              is 1.8x too deep.
    p1 >> p0  the extra level-1 k-mers are hit much harder than the unique ones,
              i.e. a relative carries them -> lineage present, this strain not
              (the donor-replacement case).

    Not applied as a filter: reported, so the threshold stays yours.

CLOSE RELATIVE  (target_relative_*, target_close_relative)
    Question: is something other than the target strain contributing to the
    target's k-mers? Only a relative can do that, and it can only do it on the
    target k-mers it carries, i.e. the SHARED blocks (presence level > 0). A
    block's presence level is fixed by the database, not by the sample, so it
    splits the target into two groups before looking at any counts:

        reference blocks  presence <= --relative_max_presence (default 0)
        shared blocks     presence >  --relative_max_presence

    H0: only the target strain is there -> every target block is covered at
        the same rate lambda (k-mer hits per k-mer), shared or not.
    H1: a relative adds hits to the shared blocks only -> shared rate higher.

    Fitted per sample as a two-group quasi-Poisson rate model on block totals
    (C_i = summed k-mer counts of block i, m_i = its k-mer count):

        log E[C_i] = log m_i + b0 + b1 * shared_i

        group depths  D_U = C_U / M_U  (unique / reference blocks)
                      D_S = C_S / M_S  (shared / non-unique blocks)
        dispersion    phi = max(1, Pearson X^2 / (n_blocks - 2)),
                      residuals of each block against its own group's rate
        eff. counts   E_g = C_g / phi                (~ reads, see below)
        rate ratio    RR = (D_S + k) / (D_U + k),  k = --relative_depth_offset

    The p-value is a conditional SCORE test, not a Wald test on log RR. Hold the
    total E = E_S + E_U fixed; under H0 every k-mer is covered at the same rate,
    so each unit of coverage falls in the shared group with probability
    pi0 = M_S / (M_S + M_U) -- a quantity fixed by the database, not the sample:

        z = (E_S - E * pi0) / sqrt(E * pi0 * (1 - pi0)),  one-sided p = P(Z > z)

    A Wald test on log RR cannot be used here, because its SE is
    sqrt(1/E_S + 1/E_U), which diverges exactly when C_U = 0 -- i.e. it reports
    "no information" in the case that carries the most information: zero hits
    across every unique k-mer while the shared blocks are deeply covered. That
    is not fixable with a pseudo-count: adding one makes the SE finite but
    arbitrary, so the p-value then reflects the size of the pseudo-count rather
    than the data. The score test needs no pseudo-count, stays finite at zero,
    and its power scales with M_U / (M_S + M_U) -- the fraction of the strain's
    k-mers that are actually unique to it -- which is the honest limit: a strain
    whose genome is 98% shared cannot support a strong call either way.

    The offset k (default 1x depth) only damps the REPORTED ratio; the test
    itself is unchanged. Without it, RR is (target + relative) / target, which
    explodes when the target is absent -- 300x on a strain that is not there at
    all, purely because the denominator is ~0. With k = 1 the denominator can
    never fall below 1x, so RR stays on a readable scale and means "how much
    does the shared signal exceed the unique signal, measured against a floor of
    1x coverage". At high target depth k is negligible and RR is unchanged; at
    low depth it is what stops a handful of stray reads looking like a 300-fold
    effect. Read it together with the two depths, which are reported raw.

    phi comes out near the number of k-mers a read covers (read length - k + 1)
    plus any extra block-to-block scatter, which is why C / phi behaves like a
    read count. Expect phi ~ 100 on real data; that is not a fault.

    Why this shape:
      - the groups are fixed by the database, so there is no selection on the
        outcome (the old too_deep-vs-kept comparison was circular: a block is
        pruned BECAUSE it is deep, so pruned blocks are covered by definition;
        in a target-absent sample max_depth is ~0.03 and one hit prunes a block).
      - the block, not the k-mer, is the unit, and phi is estimated from the
        sample's own block-to-block scatter. Neighbouring k-mers come from the
        same reads, so k-mer-level tests (Fisher on k-mer counts) are hugely
        overconfident; phi absorbs that plus GC/copy-number noise.
      - it uses depth, not breadth, so it still works when the target is
        saturated (breadth ~ a, depth still rising) and when the target is
        absent (C_U = 0 is a valid, and strong, observation for the score test;
        the offset k keeps the reported ratio finite).
      - independent of the lineage call.

    All of E = 0 (nothing covered) gives z = 0, p = 0.5: no evidence either way.

    Reported: target_relative_unique_depth (D_U) and
    target_relative_nonunique_depth (D_S), the raw depths of the two block
    groups; target_relative_rate_ratio, target_relative_dispersion,
    target_relative_z, target_relative_pvalue, and
        target_close_relative = pvalue < --relative_alpha
                                and rate_ratio >= --relative_min_ratio
    D_S - D_U is the relative's depth if it carries every shared block, and a
    lower bound otherwise. The ratio floor is there because multi-copy elements (IS, rRNA, phage)
    are over-represented among shared blocks and push RR a little above 1 even
    in a pure sample; calibrate it on samples known to hold the target alone.
    One test per strain x sample: if you screen many, adjust the p-values
    (e.g. Benjamini-Hochberg) downstream.

BLOCK OUTPUT  (--block_out)
    Also writes {strain}.target_blocks.tsv: one row per target block x sample with
    the block's own k-mer count and len_presence (number of other DB strains carrying
    the block's k-mers, 0 = unique; one value per block because blocks are cut where
    presence_list changes), mean/SD depth and breadth, the sample's
    expected_depth / max_depth / saturated, and `pruned` (True = the block is
    deeper than max_depth and is left out of target_kept_breadth).
"""
import argparse
import gzip
import json
import os
from math import erfc, sqrt

import numpy as np
import polars as pl

# streaming engine: processes the scans in chunks instead of loading everything at once
ENGINE = 'streaming'


def read_sample_cols(path):
    """Sample names from the kmer_hits header, without reading the file."""
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt') as fh:
        header = fh.readline().rstrip('\n').split('\t')
    return [c for c in header if c != '#kmer']


def wide_to_long(df, key_cols, sample_cols, stats):
    """Turn prefixed wide aggregate columns (stat__sample) into long format:
    one row per key × sample, one column per stat."""
    out = None
    for stat in stats:
        cols = [f'{stat}__{s}' for s in sample_cols]
        part = (df.select(*key_cols, *cols)
                  .rename(dict(zip(cols, sample_cols)))
                  .unpivot(index=key_cols, on=sample_cols,
                           variable_name='sample', value_name=stat))
        out = part if out is None else out.join(part, on=[*key_cols, 'sample'])
    return out


def block_stats(lf, block_col, sample_cols):
    """Per block × sample: k-mer count, mean/SD depth, breadth (fraction seen).
    Aggregates on the wide table, so the k-mer × sample long table is never built."""
    s = pl.col(sample_cols)
    df = (lf.group_by(block_col)
            .agg(pl.len().alias('block_kmers'),
                 s.mean().name.prefix('mean_depth__'),
                 s.std().name.prefix('sd_depth__'),
                 (s > 0).mean().name.prefix('breadth__'))
            .collect(engine=ENGINE))
    return (wide_to_long(df, [block_col], sample_cols, ['mean_depth', 'sd_depth', 'breadth'])
            .join(df.select(block_col, 'block_kmers'), on=block_col))


def n_presence(col='presence_list'):
    """presence_list -> number of OTHER DB strains carrying the k-mer.
    '[]' -> 0, '[123]' -> 1, '[12, 34]' -> 2."""
    return (pl.when(pl.col(col) == '[]').then(0)
              .otherwise(pl.col(col).str.count_matches(',') + 1)
              .cast(pl.UInt32))


def unique_stats(lf, sample_cols, max_presence=0):
    """Per sample: breadth and mean depth over the informative k-mers, i.e. those
    carried by at most max_presence other DB strains."""
    s = pl.col(sample_cols)
    df = (lf.filter(pl.col('n_presence') <= max_presence)
            .select(pl.len().alias('n_kmers_unique'),
                    s.mean().name.prefix('mean_depth_unique__'),
                    (s > 0).mean().name.prefix('breadth_unique__'))
            .collect(engine=ENGINE)
            .with_columns(pl.lit(1).alias('_k')))
    return (wide_to_long(df, ['_k', 'n_kmers_unique'], sample_cols,
                         ['mean_depth_unique', 'breadth_unique'])
            .drop('_k')
            .with_columns(pl.col('n_kmers_unique').cast(pl.UInt32),
                          pl.col('mean_depth_unique', 'breadth_unique').cast(pl.Float64)))


# ---------------------------------------------------------------------------
# Lander-Waterman model
# ---------------------------------------------------------------------------

def lw_inverse(breadth, model):
    """Depth that gives this breadth under breadth = a(1 - exp(-b*depth))."""
    frac = np.clip(np.asarray(breadth, dtype=float) / model['a'], 0, 1 - 1e-9)
    return -np.log1p(-frac) / model['b']


def lw_resid_sd(depth, model):
    sd = np.interp(np.log10(np.clip(depth, 1e-3, None)),
                   model['resid_sd_log10_depth'], model['resid_sd'])
    return np.maximum(sd, model['sd_floor'])


def expected_depth(df, breadth_col, model, z=3.0, prefix=''):
    """breadth -> depth, plus the upper depth the breadth is still consistent with.

    saturated: the breadth is at (or within noise of) the model's asymptote a, so
    the depth is only a lower bound -- lw_inverse blows up there."""
    b = df[breadth_col].to_numpy()
    d_hat = lw_inverse(b, model)
    b_hi = b + z * lw_resid_sd(d_hat, model)
    saturated = b_hi >= model['a']
    d_max = np.where(saturated, np.inf, lw_inverse(b_hi, model))
    return (df.with_columns(pl.Series(f'{prefix}expected_depth', d_hat),
                            pl.Series(f'{prefix}max_depth', d_max),
                            pl.Series(f'{prefix}saturated', saturated))
              # no k-mers -> NaN; turn into null
              .with_columns(pl.col(f'{prefix}expected_depth', f'{prefix}max_depth').fill_nan(None),
                            pl.when(pl.col(breadth_col).is_null()).then(None)
                              .otherwise(pl.col(f'{prefix}saturated')).alias(f'{prefix}saturated')))


def pooled_breadth(df_blocks, sample_cols):
    """Sample-level breadth: seen k-mers / all k-mers, i.e. the length-weighted
    mean of the block breadths. Equivalent to counting over the k-mer table."""
    return (df_blocks.group_by('sample')
            .agg(pl.len().alias('n_blocks'),
                 pl.col('block_kmers').sum().alias('n_kmers'),
                 ((pl.col('breadth') * pl.col('block_kmers')).sum()
                  / pl.col('block_kmers').sum()).alias('breadth'),
                 ((pl.col('mean_depth') * pl.col('block_kmers')).sum()
                  / pl.col('block_kmers').sum()).alias('mean_depth'),
                 # unweighted mean over blocks, kept for comparison with the old output
                 pl.col('breadth').mean().alias('mean_block_breadth'),
                 pl.col('breadth').std().alias('sd_block_breadth')))


# ---------------------------------------------------------------------------
# Calls
# ---------------------------------------------------------------------------

def lineage_call(lf, sample_cols, model, z=3.0):
    df_blocks = block_stats(lf, 'db_block_id', sample_cols)
    df = expected_depth(pooled_breadth(df_blocks, sample_cols), 'breadth', model, z=z)
    return (df.rename({c: f'lineage_{c}' for c in df.columns if c != 'sample'})
              .sort('sample'))


def depth_ratio(df, model):
    """observed depth of the informative k-mers / the depth their breadth implies.
    Saturated breadths give no usable expectation, so they come back null."""
    b = df['breadth_unique'].to_numpy()
    d_exp = lw_inverse(b, model)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where((b > 0) & (b < model['a']),
                     df['mean_depth_unique'].to_numpy() / d_exp, np.nan)
    return pl.Series('depth_ratio', r)


def relative_test(df_blocks, max_presence=0, offset=1.0):
    """Close-relative test: shared vs reference block coverage rate, per sample.
    Two-group quasi-Poisson on block totals; see CLOSE RELATIVE in the docstring.
    df_blocks: block x sample with block_kmers, mean_depth, len_presence."""
    b = (df_blocks
         .filter(pl.col('len_presence').is_not_null())
         .with_columns((pl.col('len_presence') > max_presence).alias('_shared'),
                       (pl.col('mean_depth') * pl.col('block_kmers')).alias('_c')))
    # group rates, then Pearson residuals of every block against its group rate
    rates = (b.group_by('sample', '_shared')
              .agg((pl.col('_c').sum() / pl.col('block_kmers').sum()).alias('_r')))
    b = (b.join(rates, on=['sample', '_shared'])
          .with_columns((pl.col('block_kmers') * pl.col('_r')).alias('_mu'))
          .with_columns(pl.when(pl.col('_mu') > 0)
                          .then((pl.col('_c') - pl.col('_mu')) ** 2 / pl.col('_mu'))
                          .otherwise(0.0).alias('_x2')))
    sh, ref = pl.col('_shared'), ~pl.col('_shared')
    agg = (b.group_by('sample')
            .agg(pl.col('_c').filter(sh).sum().alias('c_s'),
                 pl.col('block_kmers').filter(sh).sum().alias('m_s'),
                 pl.col('_c').filter(ref).sum().alias('c_u'),
                 pl.col('block_kmers').filter(ref).sum().alias('m_u'),
                 pl.len().alias('n'),
                 pl.col('_x2').sum().alias('x2'))
            .sort('sample'))

    c_s, m_s = agg['c_s'].to_numpy().astype(float), agg['m_s'].to_numpy().astype(float)
    c_u, m_u = agg['c_u'].to_numpy().astype(float), agg['m_u'].to_numpy().astype(float)
    n, x2 = agg['n'].to_numpy().astype(float), agg['x2'].to_numpy()
    ok = (m_s > 0) & (m_u > 0) & (n > 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        phi = np.where(ok, np.maximum(1.0, x2 / (n - 2)), np.nan)
        d_s = np.where(m_s > 0, c_s / np.where(m_s > 0, m_s, 1), np.nan)
        d_u = np.where(m_u > 0, c_u / np.where(m_u > 0, m_u, 1), np.nan)
        # effective counts (~reads): a read hits up to (read length - k + 1)
        # neighbouring k-mers, so k-mer hits are not independent; phi measures that
        e_s, e_u = c_s / phi, c_u / phi
        e, pi0 = e_s + e_u, m_s / (m_s + m_u)
        # conditional score test: shared share of the coverage vs the share of the
        # k-mers that is shared. Finite at c_u == 0; no pseudo-count needed.
        zval = np.where(ok & (e > 0),
                        (e_s - e * pi0) / np.sqrt(e * pi0 * (1 - pi0)),
                        np.where(ok, 0.0, np.nan))
        rate_ratio = np.where(ok, (d_s + offset) / (d_u + offset), np.nan)
    pval = np.array([0.5 * erfc(v / sqrt(2)) if np.isfinite(v) else np.nan for v in zval])
    return pl.DataFrame({
        'sample': agg['sample'],
        'relative_unique_depth': d_u,
        'relative_nonunique_depth': d_s,
        'relative_rate_ratio': rate_ratio,
        'relative_dispersion': phi,
        'relative_z': zval,
        'relative_pvalue': pval,
    }).with_columns(pl.col(pl.Float64).fill_nan(None))


def target_call(lf, sample_cols, model, z=3.0, min_unique_kmers=1000,
                df_presence=None, rel_max_presence=0, rel_alpha=0.01, rel_min_ratio=1.2,
                rel_offset=1.0):
    """Returns (per-sample target call, per-block x sample table with too_deep).
    df_presence: block_id -> len_presence, needed for the close-relative test."""
    # 1. informative k-mers at both presence levels; keep level 0 if it is big
    #    enough, else fall back to level 1 (few unique k-mers -> coarse, fragile
    #    breadth). Both ratios are reported either way.
    levels = {lvl: unique_stats(lf, sample_cols, max_presence=lvl) for lvl in (0, 1)}
    n0 = levels[0]['n_kmers_unique'][0] if levels[0].height else 0
    level = 0 if n0 >= min_unique_kmers else 1
    n1 = levels[1]['n_kmers_unique'][0] if levels[1].height else 0
    if level == 1:
        print(f'only {n0} k-mers unique to this strain (< {min_unique_kmers}); '
              f'using presence level 1 ({n1} k-mers) for the LW estimate')

    df_ratio = (levels[0].select('sample', depth_ratio(levels[0], model)
                                             .alias('target_depth_ratio_p0'))
                .join(levels[1].select('sample', depth_ratio(levels[1], model)
                                                   .alias('target_depth_ratio_p1')),
                      on='sample'))

    # 2. sample-level breadth over the chosen set -> expected / max depth
    df_un = expected_depth(levels[level], 'breadth_unique', model, z=z)
    # 3. all target k-mers per block, drop blocks deeper than the sample allows
    df_blocks = (block_stats(lf, 'block_id', sample_cols)
                 .join(df_presence, on='block_id', how='left')
                 .join(df_un, on='sample', how='left')
                 .with_columns((pl.col('mean_depth') > pl.col('max_depth')).alias('too_deep')))
    df_kept = pooled_breadth(df_blocks.filter(~pl.col('too_deep')), sample_cols)
    # 4. pooled breadth over all target blocks, nothing removed
    df_all = pooled_breadth(df_blocks, sample_cols)
    # 5. close relative: shared vs reference blocks (independent of pruning)
    df_rel = (relative_test(df_blocks, max_presence=rel_max_presence, offset=rel_offset)
              .with_columns(((pl.col('relative_pvalue') < rel_alpha)
                             & (pl.col('relative_rate_ratio') >= rel_min_ratio))
                            .alias('close_relative')))
    df_rel = df_rel.rename({c: f'target_{c}' for c in df_rel.columns if c != 'sample'})
    df_result = (
        df_un.join(
            df_blocks.group_by('sample').agg(pl.len().alias('n_blocks'),
                                             pl.col('too_deep').sum().alias('n_removed')),
            on='sample', how='left')
        .join(df_kept.select('sample',
                             pl.col('breadth').alias('kept_breadth'),
                             pl.col('mean_depth').alias('kept_mean_depth'),
                             pl.col('mean_block_breadth').alias('kept_mean_block_breadth')),
              on='sample', how='left')
        .join(df_all.select('sample',
                            pl.col('breadth').alias('target_total_block_breadth')),
              on='sample', how='left')
        .rename({c: f'target_{c}' for c in df_un.columns if c != 'sample'})
        .rename({'n_blocks': 'target_n_blocks', 'n_removed': 'target_n_removed',
                 'kept_breadth': 'target_kept_breadth',
                 'kept_mean_depth': 'target_kept_mean_depth',
                 'kept_mean_block_breadth': 'target_kept_mean_block_breadth'})
        .with_columns(pl.lit(level, dtype=pl.UInt32).alias('target_presence_level'),
                      pl.lit(n0, dtype=pl.UInt32).alias('target_n_kmers_p0'),
                      pl.lit(n1, dtype=pl.UInt32).alias('target_n_kmers_p1'))
        .join(df_ratio, on='sample', how='left')
        .join(df_rel, on='sample', how='left')
        .with_columns(pl.col('target_depth_ratio_p0', 'target_depth_ratio_p1').fill_nan(None))
        .sort('sample')
    )
    return df_result, df_blocks


# empty lineage table, used when a strain has no lineage k-mers
LINEAGE_SCHEMA = {
    'sample': pl.String,
    'lineage_n_blocks': pl.UInt32,
    'lineage_n_kmers': pl.UInt32,
    'lineage_breadth': pl.Float64,
    'lineage_mean_depth': pl.Float64,
    'lineage_mean_block_breadth': pl.Float64,
    'lineage_sd_block_breadth': pl.Float64,
    'lineage_expected_depth': pl.Float64,
    'lineage_max_depth': pl.Float64,
    'lineage_saturated': pl.Boolean,
}


def main():
    parser = argparse.ArgumentParser(
        description='Lineage and target strain calls from k-mer block coverage (LW model).')
    parser.add_argument('--kmer_blocks', required=True,
                        help='*.rare_kmers.mapped.tsv from kmer_blocks')
    parser.add_argument('--kmer_detect', required=True,
                        help='*.kmer_hits.tsv.gz from strain_detect')
    parser.add_argument('--model', required=True,
                        help='Lander-Waterman coverage model json (coverage_model_lw.json)')
    parser.add_argument('--output_dir', default='.')
    parser.add_argument('--basename', default=None,
                        help='Output basename / strain name (default: derived from --kmer_blocks)')
    parser.add_argument('--z', type=float, default=3.0,
                        help='residual SDs used for max_depth (default: 3)')
    parser.add_argument('--min_unique_kmers', type=int, default=1000,
                        help='if fewer than this many target k-mers are unique to the strain '
                             '(presence level 0), fall back to level 1 -- k-mers shared with at '
                             'most one other DB strain -- for the LW estimate (default: 1000)')
    parser.add_argument('--relative_max_presence', type=int, default=0,
                        help='close-relative test: blocks with presence <= this are the '
                             'reference, the rest are "shared" (default: 0)')
    parser.add_argument('--relative_alpha', type=float, default=0.01,
                        help='close-relative test: one-sided p-value cutoff (default: 0.01)')
    parser.add_argument('--relative_depth_offset', type=float, default=1.0,
                        help='close-relative test: depth added to both groups before taking '
                             'the reported rate ratio, so a near-zero unique depth cannot '
                             'produce a huge ratio (default: 1.0; 0 = raw ratio)')
    parser.add_argument('--relative_min_ratio', type=float, default=1.2,
                        help='close-relative test: minimum shared/reference depth ratio '
                             '(default: 1.2)')
    parser.add_argument('--block_out', action='store_true',
                        help='also write {strain}.target_blocks.tsv: per block x sample '
                             'breadth/depth and whether the block was pruned (too_deep)')
    args = parser.parse_args()

    strain = args.basename if args.basename else \
        os.path.basename(args.kmer_blocks).replace('.rare_kmers.mapped.tsv', '')
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.model) as fh:
        lw = json.load(fh)

    # lazy scans; only the columns that are needed get read.
    # counts are parsed as UInt64 because the total_evaluated row holds totals > UInt32 max;
    # that row is dropped and the per-k-mer counts cast to UInt32 before the join,
    # which halves the size of the hits table held for the join.
    sample_cols = read_sample_cols(args.kmer_detect)
    lf_hits = (pl.scan_csv(args.kmer_detect, separator='\t',
                           schema_overrides={c: pl.UInt64 for c in sample_cols})
                 .filter(pl.col('#kmer') != 'total_evaluated')
                 .with_columns(pl.col(sample_cols).cast(pl.UInt32)))
    lf_kmers = pl.scan_csv(args.kmer_blocks, separator='\t', null_values='NA')

    def with_hits(lf):
        # left join keeps k-mers absent from kmer_hits; their counts become 0
        return (lf.join(lf_hits, how='left', on='#kmer')
                  .with_columns(pl.col(sample_cols).fill_null(0))
                  .drop('#kmer'))

    # lineage (skipped if the strain has no lineage k-mers)
    lf_lineage = lf_kmers.filter(pl.col('kmer_type') == 'lineage')
    lineage_meta = (lf_lineage.select(pl.col('lineage_call').drop_nulls().unique(),
                                      pl.col('gtdb_tax').drop_nulls().unique())
                              .collect())
    if lineage_meta.height > 0:
        lineage_name = lineage_meta['lineage_call'].item()
        gtdb_tax = lineage_meta['gtdb_tax'].item()
        df_lineage_call = lineage_call(with_hits(lf_lineage.select('#kmer', 'db_block_id')),
                                       sample_cols, lw, z=args.z)
    else:
        print(f'WARNING: no lineage k-mers for {strain}, lineage columns left empty')
        lineage_name, gtdb_tax = None, None
        df_lineage_call = pl.DataFrame(schema=LINEAGE_SCHEMA)

    # number of OTHER DB strains carrying each target block's k-mers (0 = unique to the
    # strain). kmer_blocks cuts a block wherever presence_list changes, so it is one value
    # per block; needs no hits, so cheap. Used by the close-relative test and --block_out.
    df_presence = (lf_kmers.filter(pl.col('kmer_type') == 'target')
                           .select('block_id', n_presence().alias('n_presence'))
                           .group_by('block_id')
                           .agg(pl.col('n_presence').max().alias('len_presence'),
                                pl.col('n_presence').min().alias('_min'))
                           .collect(engine=ENGINE))
    if (df_presence['len_presence'] != df_presence['_min']).any():
        print('WARNING: presence_list differs within some blocks; len_presence is the maximum')
    df_presence = df_presence.drop('_min')

    # target strain detection; presence_list reduced to a count before the join
    lf_target = with_hits(
        lf_kmers.filter(pl.col('kmer_type') == 'target')
                .select('#kmer', 'block_id', n_presence().alias('n_presence'))
    )
    df_target_call, df_target_blocks = target_call(
        lf_target, sample_cols, lw, z=args.z, min_unique_kmers=args.min_unique_kmers,
        df_presence=df_presence, rel_max_presence=args.relative_max_presence,
        rel_alpha=args.relative_alpha, rel_min_ratio=args.relative_min_ratio,
        rel_offset=args.relative_depth_offset)

    if args.block_out:
        df_block_out = (
            df_target_blocks
            .select(pl.lit(strain).alias('strain'),
                    'block_id', 'sample', 'block_kmers',
                    'len_presence',
                    'mean_depth', 'sd_depth', 'breadth',
                    'expected_depth', 'max_depth', 'saturated',
                    # no max_depth (no informative k-mers) -> block is not kept either,
                    # so it counts as pruned, matching target_kept_breadth
                    pl.col('too_deep').fill_null(True).alias('pruned'))
            .sort('block_id', 'sample')
        )
        block_path = os.path.join(args.output_dir, f'{strain}.target_blocks.tsv')
        df_block_out.write_csv(block_path, separator='\t', float_precision=4)
        print(f'Wrote {block_path}')

    # left join keeps every target sample, even without lineage results
    df_results = (
        df_target_call.join(df_lineage_call, on='sample', how='left')
        .with_columns(
            pl.lit(strain).alias('strain'),
            pl.lit(lineage_name, dtype=pl.String).alias('lineage_call'),
            pl.lit(gtdb_tax, dtype=pl.String).alias('gtdb_tax'),
            # depth of the lineage not explained by the target strain
            (pl.col('lineage_expected_depth') - pl.col('target_expected_depth'))
              .alias('lineage_excess_depth'),
        )
        .select('strain', 'lineage_call', 'gtdb_tax', 'sample',
                pl.exclude('strain', 'lineage_call', 'gtdb_tax', 'sample'))
        # string columns get NA, numeric columns stay empty
        .with_columns(pl.col(pl.String).fill_null('NA'))
    )

    out_path = os.path.join(args.output_dir, f'{strain}.coverage.tsv')
    df_results.write_csv(out_path, separator='\t', float_precision=4)
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()