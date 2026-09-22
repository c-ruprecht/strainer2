import argparse
import os
import gzip

import polars as pl
import pandas as pd
import ahocorasick
from Bio import SeqIO
from Bio.Seq import Seq


LINEAGE_COL = 'lineage_id'
TAX_COL = 'gtdb_taxonomy'


def load_genome(genome_path):
    opener = gzip.open if genome_path.endswith('.gz') else open
    records = {}
    with opener(genome_path, 'rt') as fh:
        for record in SeqIO.parse(fh, 'fasta'):
            records[record.id] = record.seq
    return records


def strain_name_from_path(path):
    base = os.path.basename(path)
    for ext in ('.fna.gz', '.fasta.gz', '.fa.gz', '.fna', '.fasta', '.fa'):
        if base.endswith(ext):
            return base[: -len(ext)]
    return base.split('.')[0]


def get_lowest_percentile(df, percentile=0.05, drug_scrub='percentile'):

    if df['drug_count'].isna().all():
        print('No drug counts found, continuing without drugscrub')
        df['drug_count'] = 0

    if drug_scrub == 'percentile':
        lowest = df[
            (df['reference_count'] <= df['reference_count'].quantile(percentile)) &
            (df['pangenome_count'] <= df['pangenome_count'].quantile(percentile)) &
            (df['metagenome_count'] <= df['metagenome_count'].quantile(percentile)) &
            (df['drug_count'] <= df['drug_count'].quantile(percentile))
        ].copy()

    if drug_scrub == 'count_hard':
        lowest = df[
            (df['reference_count'] <= df['reference_count'].quantile(percentile)) &
            (df['pangenome_count'] <= df['pangenome_count'].quantile(percentile)) &
            (df['metagenome_count'] <= df['metagenome_count'].quantile(percentile)) &
            (df['drug_count'] == 0)
        ].copy()
    return lowest


def build_mapped_kmers_ahocorasick(records, kmers, terminal_dist):
    # Build Aho-Corasick automaton with forward and reverse complement kmers
    # Important: only pass single-count kmers; only the first hit per kmer is kept
    kmers = list(kmers)
    A = ahocorasick.Automaton()
    for kmer in kmers:
        A.add_word(kmer, (kmer, False))
        A.add_word(str(Seq(kmer).reverse_complement()), (kmer, True))
    A.make_automaton()

    found = set()
    rows = []
    for record_id, seq in records.items():
        for pos, (kmer, is_rc) in A.iter(str(seq)):
            if kmer not in found:
                rows.append((record_id, kmer, pos - len(kmer) + 1, is_rc))
                found.add(kmer)

    df = pd.DataFrame(rows, columns=['contig_id', '#kmer', 'kmer_position', 'reverse_complement'])

    if len(df) < len(kmers):
        print(f'WARNING: not all kmers found in genome ({len(df)} / {len(kmers)})')
    elif len(df) > len(kmers):
        print('WARNING: kmers found more than once')
    else:
        print(f'  {len(df)} kmers mapped (all unique)')

    dict_len = {cid: len(seq) for cid, seq in records.items()}
    df['contig_length'] = df['contig_id'].map(dict_len)
    df['terminal_kmer'] = (
        (df['kmer_position'] < terminal_dist) |
        ((df['contig_length'] - df['kmer_position']) < terminal_dist)
    )
    df['label'] = df['terminal_kmer'].map({True: 'terminal', False: 'internal'})

    n_terminal = int(df['terminal_kmer'].sum())
    print(f'  Terminal kmers: {n_terminal} / {len(df)}')

    return df, dict_len


def add_block_ids(df, min_block_size=30, presence_col='list_scrub_id'):
    df = df.sort_values(['contig_id', 'kmer_position']).copy()
    presence = df[presence_col].fillna('')

    pos_break = df.groupby('contig_id')['kmer_position'].diff().ne(1)
    presence_break = presence.ne(presence.shift())
    df['block_id'] = (pos_break | presence_break).cumsum()

    df['block_size'] = df.groupby('block_id')['block_id'].transform('size')
    df = df[df['block_size'] >= min_block_size].copy()
    df['block_id'] = pd.factorize(df['block_id'])[0] + 1

    df_gpd = df[['block_id', 'block_size', 'presence_list']].drop_duplicates(subset=['block_id'])
    return df, df_gpd


def restore_int_cols(df):
    """concat with NA rows upcasts ints to float -> back to nullable Int64."""
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            v = df[c].dropna()
            if len(v) and (v % 1 == 0).all():
                df[c] = df[c].astype('Int64')
    return df


def parse_presence(s):
    return [int(x) for x in str(s).split(',')] if pd.notna(s) else []


# --------------------------------------------------------------------------- #
# Lineage call
# --------------------------------------------------------------------------- #

def call_lineage(query_kmers, lineage_db, lineage_col=LINEAGE_COL, tax_col=TAX_COL,
                 metric='n_hits', top_n=5):
    """Group lineage-DB hits of the query kmers by lineage and pick the best one.

    query_kmers : pl.DataFrame with a single '#kmer' column (unique)
    Returns (lineage_call, gtdb_tax, df_calls) where df_calls holds the per-lineage
    scores for every lineage with >= 1 hit.
    """
    lf = pl.scan_parquet(lineage_db)
    cols = lf.collect_schema().names()
    missing = {'#kmer', lineage_col, tax_col} - set(cols)
    if missing:
        raise ValueError(f"lineage_db missing columns {sorted(missing)}, found: {cols}")

    tax_expr = pl.col(tax_col).first().alias('gtdb_tax')

    # hits per lineage among the query kmers
    df_hits = (lf.join(query_kmers.lazy(), on='#kmer', how='semi')
                 .group_by(lineage_col)
                 .agg(pl.len().alias('n_hits'), tax_expr)
                 .collect(engine='streaming'))

    if df_hits.height == 0:
        print('WARNING: no query kmers found in lineage_db — no lineage call')
        return None, None, df_hits

    # total kmers per hit lineage -> fraction of the lineage recovered
    df_tot = (lf.join(df_hits.lazy().select(lineage_col), on=lineage_col, how='semi')
                .group_by(lineage_col)
                .agg(pl.len().alias('n_lineage_kmers'))
                .collect(engine='streaming'))

    n_query = query_kmers.height
    df_calls = (df_hits.join(df_tot, on=lineage_col, how='left')
                .with_columns(
                    (pl.col('n_hits') / pl.col('n_lineage_kmers')).alias('frac_lineage_hit'),
                    (pl.col('n_hits') / n_query).alias('frac_query_hit'))
                .sort([metric, 'n_hits'], descending=True))

    print(f'Lineage hits (top {top_n} of {df_calls.height}):')
    print(df_calls.head(top_n))

    best = df_calls.row(0, named=True)
    if df_calls.height > 1:
        second = df_calls.row(1, named=True)
        ratio = second[metric] / best[metric] if best[metric] else float('nan')
        if ratio > 0.5:
            print(f"WARNING: ambiguous lineage call — runner-up {second[lineage_col]} "
                  f"scores {ratio:.2f}x of best")

    print(f"Lineage call: {best[lineage_col]}  ({best['n_hits']} hits, "
          f"{best['frac_lineage_hit']:.3f} of lineage kmers)")
    print(f"GTDB tax:     {best['gtdb_tax']}")
    return best[lineage_col], best['gtdb_tax'], df_calls


def get_lineage_kmers(lineage_db, lineage, lineage_col=LINEAGE_COL):
    """All unique kmers of the called lineage with their DB block id and block size."""
    return (pl.scan_parquet(lineage_db)
              .filter(pl.col(lineage_col) == lineage)
              .select('#kmer', 'kmer_block_id', 'block_n_kmers')
              .sort('kmer_block_id')
              .unique(subset='#kmer', keep='first', maintain_order=True)
              .collect(engine='streaming'))


def main():
    parser = argparse.ArgumentParser(
        description='Select strain-informative kmers, add lineage kmers, build kmer blocks.')
    parser.add_argument('--reference', required=True,
                        help='Genome FASTA (.fna or .fna.gz)')
    parser.add_argument('--input_dir', required=True,
                        help='directory of scrub output files, containing summary, global counts and presence file')
    parser.add_argument('--output_dir', default='.')
    parser.add_argument('--basename', default=None,
                        help='Output basename (default: derived from --reference)')
    parser.add_argument('--terminal_dist', type=int, default=300,
                        help='Distance from contig ends to flag terminal kmers')
    parser.add_argument('--min_block_size', type=int, default=20)
    parser.add_argument('--percentile', type=float, default=0.01,
                        help='percentile union on pan genome and metagenome counts for rare kmer selection')
    parser.add_argument('--lineage_db', type=str, required=True,
                        help='path to .parquet lineage file in db')
    parser.add_argument('--lineage_metric', choices=['n_hits', 'frac_lineage_hit'], default='n_hits', help='score used to pick the lineage call')
    parser.add_argument('--no_map_lineage', action='store_true',
                        help='skip mapping lineage kmers to the reference (positions left NA)')
    args = parser.parse_args()

    basename = args.basename if args.basename else strain_name_from_path(args.reference)
    os.makedirs(args.output_dir, exist_ok=True)

    # read dataframes
    df_global = pl.read_csv(os.path.join(args.input_dir, '*.global_counts.tsv.zst'),
                            separator='\t')
    df_summary = pl.read_csv(os.path.join(args.input_dir, '*.summary.tsv'),
                             separator='\t')
    df_presence = pl.read_csv(os.path.join(args.input_dir, '*.presence.tsv.zst'),
                              separator='\t')

    # ---- lineage call on all kmers of the strain ----
    query_kmers = df_global.select('#kmer').unique()
    lineage_call, gtdb_tax, df_calls = call_lineage(
        query_kmers, args.lineage_db, metric=args.lineage_metric)
    df_calls.write_csv(os.path.join(args.output_dir, f'{basename}.lineage_call.tsv'),
                       separator='\t')

    # ---- rare (target) kmers ----
    df_global_pd = df_global.to_pandas()
    df_presence_pd = df_presence.to_pandas()

    # lineage kmers fetched up front: needed to purge non-strain-specific targets
    if lineage_call is not None:
        df_lin_kmers = get_lineage_kmers(args.lineage_db, lineage_call).to_pandas()
    else:
        df_lin_kmers = pd.DataFrame(columns=['#kmer', 'kmer_block_id', 'block_n_kmers'])

    df_rare = get_lowest_percentile(df_global_pd, percentile=args.percentile, drug_scrub='count_hard')

    # rare kmers shared with the lineage are by definition not strain specific -> drop.
    # Done before mapping/blocking so blocks only span strain-specific kmers.
    in_lineage = df_rare['#kmer'].isin(df_lin_kmers['#kmer'])
    print(f'dropped {int(in_lineage.sum())} / {len(df_rare)} rare kmers shared with lineage')
    df_rare = df_rare[~in_lineage].copy()
    print(df_rare.sort_values('pangenome_count'))

    genome = load_genome(args.reference)
    df_locations, _ = build_mapped_kmers_ahocorasick(genome, df_rare['#kmer'],
                                                     terminal_dist=args.terminal_dist)
    if len(df_locations) == 0:
        print('No kmers mapped to reference!')

    df_merge = pd.merge(df_locations, df_global_pd, on='#kmer', how='left')
    df_merge2 = pd.merge(df_merge, df_presence_pd, on='#kmer', how='left')

    # convert scrub ids to lists; empty list where pangenome and metagenome_count == 0, else drop
    missing = df_merge2['list_scrub_id'].isna()
    has_hits = df_merge2[['pangenome_count', 'metagenome_count']].gt(0).any(axis=1)

    # capped presence list -> the empty entry is an artefact, not a real absence
    n_capped = int((missing & has_hits).sum())
    df_merge2 = df_merge2[~(missing & has_hits)].copy()
    df_merge2['presence_list'] = df_merge2['list_scrub_id'].apply(parse_presence)

    print(f'dropped {n_capped} k-mers with capped presence lists')
    df_merge2 = df_merge2.loc[df_merge2['terminal_kmer'] == False].copy()

    df_target, df_gpd = add_block_ids(df_merge2, min_block_size=args.min_block_size)
    df_target['kmer_type'] = 'target'

    # ---- lineage kmers (full set for the called lineage) ----
    if lineage_call is not None:
        n_in_sweep = int(df_lin_kmers['#kmer'].isin(df_global_pd['#kmer']).sum())
        print(f'Lineage kmers: {len(df_lin_kmers)}; '
              f'{n_in_sweep} present in initial sweep, '
              f'{len(df_lin_kmers) - n_in_sweep} added from lineage_db')

        if not args.no_map_lineage and len(df_lin_kmers) > 0:
            df_lin_loc, _ = build_mapped_kmers_ahocorasick(genome, df_lin_kmers['#kmer'],
                                                           terminal_dist=args.terminal_dist)
            df_lin = pd.merge(df_lin_kmers, df_lin_loc, on='#kmer', how='left')
        else:
            df_lin = df_lin_kmers.copy()

        df_lin = pd.merge(df_lin, df_global_pd, on='#kmer', how='left')
        df_lin = pd.merge(df_lin, df_presence_pd, on='#kmer', how='left')
        df_lin['presence_list'] = df_lin['list_scrub_id'].apply(parse_presence)
        df_lin['kmer_type'] = 'lineage'

        # block ids/sizes from the DB; offset ids past the target blocks so the two
        # never collide in block_id. Original DB id kept in db_block_id.
        block_offset = int(df_target['block_id'].max()) if len(df_target) else 0
        df_lin['db_block_id'] = df_lin['kmer_block_id']
        df_lin['block_id'] = pd.factorize(df_lin['kmer_block_id'], sort=True)[0] + 1 + block_offset
        df_lin['block_size'] = df_lin['block_n_kmers']
        df_lin = df_lin.drop(columns=['kmer_block_id', 'block_n_kmers'])
        print(f"Lineage blocks: {df_lin['block_id'].nunique()} "
              f"(block_id {block_offset + 1}..{int(df_lin['block_id'].max())})")

        df_gpd_lin = (df_lin[['block_id', 'block_size', 'db_block_id']]
                      .drop_duplicates(subset=['block_id'])
                      .assign(kmer_type='lineage'))
        df_gpd = pd.concat([df_gpd.assign(kmer_type='target'), df_gpd_lin],
                           ignore_index=True, sort=False)

        df_out = pd.concat([df_target, df_lin], ignore_index=True, sort=False)
    else:
        df_out = df_target
        df_gpd = df_gpd.assign(kmer_type='target')

    df_out['lineage_call'] = lineage_call
    df_out['gtdb_tax'] = gtdb_tax

    # '#kmer' first: kmer_strain_detect advances with strtok('\t'), which collapses
    # empty fields -> any empty cell before the kmer column would shift it.
    front = ['#kmer', 'kmer_type', 'lineage_call', 'gtdb_tax']
    df_out = df_out[front + [c for c in df_out.columns if c not in front]]

    df_out = restore_int_cols(df_out)
    print(df_out['kmer_type'].value_counts())
    df_out.to_csv(os.path.join(args.output_dir, f'{basename}.rare_kmers.mapped.tsv'),
                  sep='\t', index=None, na_rep='NA')

    df_gpd = restore_int_cols(df_gpd)
    df_gpd = df_gpd[['block_id', 'kmer_type', 'block_size'] +
                    [c for c in df_gpd.columns if c not in ('block_id', 'kmer_type', 'block_size')]]
    df_gpd.to_csv(os.path.join(args.output_dir, f'{basename}.kmer_blocks.tsv'),
                  sep='\t', index=None, na_rep='NA')


if __name__ == '__main__':
    main()