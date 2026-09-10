#!/usr/bin/env python3
"""Select strain-informative kmers from a scrub database and build kmer pairs.

Flow:
  1. global counts -> drop repeated reference kmers and drug-scrub hits
  2. split into informative singletons (zero pangenome + metagenome counts)
     and non-informative ones
  3. map singletons onto the genome, drop terminal kmers, reduce to a
     non-overlapping set, pair them
  4. if that set is smaller than --max_kmers, pull non-informative kmers from
     the presence index, pair them, and reduce to a set that is independent of
     each other AND of the singletons already committed
  5. keep only pairs whose two members both survived, export their locations

Usage:
    python kmer_scrub_filter_pairs.py \
        --genome strain.fna.gz \
        --counts_global counts_global.tsv \
        --counts_individual kmer_presence.tsv.zst \
        --counts_summary summary.tsv \
        --basename my_strain --output_dir results/
"""
import argparse
import gc
import glob
import gzip
import os
import random
from collections import defaultdict

import ahocorasick
import numpy as np
import pandas as pd
import plotly.express as px
import polars as pl
from Bio import SeqIO
from Bio.Seq import Seq

from kmer_pairs import (
    create_all_pairs,
    kmer_pairs_from_presence,
    write_empty_pairs,
)


# ── genome IO ───────────────────────────────────────────────────────────

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


def build_mapped_kmers_ahocorasick(records, kmers, terminal_dist):
    """Locate each kmer (forward or reverse complement) on the genome.
    Only the first hit per kmer is kept, so pass single-count kmers."""
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

    df = pd.DataFrame(
        rows, columns=['contig_id', '#kmer', 'kmer_position', 'reverse_complement'])

    if len(df) < len(kmers):
        print(f'  WARNING: {len(kmers) - len(df)} kmers not found in genome')
    else:
        print(f'  {len(df)} kmers mapped')

    dict_len = {cid: len(seq) for cid, seq in records.items()}
    df['contig_length'] = df['contig_id'].map(dict_len)
    df['terminal_kmer'] = (
        (df['kmer_position'] < terminal_dist) |
        ((df['contig_length'] - df['kmer_position']) < terminal_dist)
    )
    df['label'] = df['terminal_kmer'].map({True: 'terminal', False: 'internal'})
    print(f'  Terminal kmers: {int(df["terminal_kmer"].sum())} / {len(df)}')

    return df, dict_len


# ── overlap graph and independent-set selection ─────────────────────────

def find_overlap_kmer_fast(df, max=0.8, chunk_size=10_000):
    """Map each kmer to the kmers whose genomic footprint it overlaps.
    Returns a one-directional dict: symmetrise before using it as a graph."""
    dict_overlap = {}
    same_thresh = 31 * max
    cross_lo = 31 * (1 - max)
    cross_hi = 62 * max

    for contig_id, contig_df in df.groupby('contig_id'):
        kmers = contig_df['#kmer'].values
        positions = contig_df['kmer_position'].values
        is_rc = contig_df['reverse_complement'].values.astype(bool)
        n = len(kmers)
        overlap_lists = [[] for _ in range(n)]

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            dist = positions[None, :] - positions[start:end, None]
            rc_i = is_rc[start:end, None]
            rc_j = is_rc[None, :]

            same_strand = (rc_i == rc_j) & (np.abs(dist) < same_thresh)
            fwd_rc = (~rc_i) & rc_j & (dist > cross_lo) & (dist < cross_hi)
            rc_fwd = rc_i & (~rc_j) & (dist > -cross_hi) & (dist < -cross_lo)

            overlap_matrix = same_strand | fwd_rc | rc_fwd
            for local_i in range(end - start):
                overlap_matrix[local_i, start + local_i] = False
            for local_i in range(end - start):
                overlap_lists[start + local_i] = kmers[overlap_matrix[local_i]].tolist()

        for i in range(n):
            dict_overlap[kmers[i]] = overlap_lists[i]

    return dict_overlap


def max_independent_kmers_greedy_heap(dict_overlap):
    """Greedy minimum-degree independent set, heap-backed."""
    import heapq

    original_degree = {k: len(v) for k, v in dict_overlap.items()}
    adj = {k: set(v) for k, v in dict_overlap.items()}
    for k, nbrs in list(adj.items()):
        for nb in nbrs:
            adj.setdefault(nb, set()).add(k)

    degree = {k: len(v) for k, v in adj.items()}
    counter = 0
    heap = []
    for k in dict_overlap.keys():
        heapq.heappush(heap, (degree[k], original_degree[k], counter, k))
        counter += 1

    selected, excluded = [], set()
    while heap:
        d, od, _, node = heapq.heappop(heap)
        if node in excluded:
            continue
        if d != degree[node]:
            heapq.heappush(heap, (degree[node], original_degree[node], counter, node))
            counter += 1
            continue

        selected.append(node)
        excluded.add(node)
        for nb in adj[node]:
            if nb in excluded:
                continue
            excluded.add(nb)
            for nb2 in adj[nb]:
                if nb2 not in excluded:
                    degree[nb2] -= 1
                    heapq.heappush(heap, (degree[nb2], original_degree[nb2], counter, nb2))
                    counter += 1
    return selected


def select_independent_with_fixed(df_locations, fixed_kmers, candidate_kmers, overlap):
    """Independent set over candidate_kmers that also avoids fixed_kmers.

    The singletons are selected and committed before the pair kmers exist, so
    running the greedy on the pair kmers alone would happily pick one sitting
    on top of a selected singleton. Build the graph over the union, drop every
    candidate adjacent to a committed kmer, then reduce what's left.
    """
    fixed_kmers = set(fixed_kmers)
    candidate_kmers = set(candidate_kmers) - fixed_kmers

    sub = df_locations.loc[df_locations['#kmer'].isin(fixed_kmers | candidate_kmers)]
    dict_overlap = find_overlap_kmer_fast(sub, max=overlap)

    adj = defaultdict(set)
    for k, nbrs in dict_overlap.items():
        for nb in nbrs:
            adj[k].add(nb)
            adj[nb].add(k)

    blocked = set()
    for k in fixed_kmers:
        blocked |= adj.get(k, set())

    free = candidate_kmers - blocked
    sub_overlap = {k: [nb for nb in adj.get(k, ()) if nb in free] for k in free}

    print(f'  {len(candidate_kmers) - len(free):,} pair kmers dropped for '
          f'overlapping a selected singleton, {len(free):,} still eligible')
    return max_independent_kmers_greedy_heap(sub_overlap)


# ── presence index helpers ──────────────────────────────────────────────

def exclusion_list(summary_tsv, similarity_t=None):
    """scrub_ids whose hits should not count: samples excluded from the global
    columns, or below a coverage cutoff if one is given."""
    df = pl.read_csv(summary_tsv, separator='\t')
    if similarity_t is not None:
        df_t = df.filter(pl.col('coverage_pct') < similarity_t)
    else:
        df_t = df.filter(pl.col('is_in_global') == False)
    return df_t.get_column('scrub_id').cast(pl.UInt32).to_list()


def _pl_to_pandas(df):
    """polars -> pandas without going through pyarrow."""
    return pd.DataFrame({c: df[c].to_list() for c in df.columns})


def annotate_kmers(df_export, df_global_counts, presence_tsv, li_t):
    """Attach the global scrub counts and the presence list to the export.

    list_scrub_id is the li_t-masked list — the same one the pipeline uses to
    judge informativeness — re-joined with commas, and n_presence is its
    length. An empty list means the kmer was hit by nothing, or was dropped by
    the -P cap in the C writer; the count columns tell those apart, since a
    capped kmer has non-zero pangenome/metagenome counts.
    """
    used = pl.Series(sorted(df_export['#kmer'].unique())).implode()

    presence = (
        pl.scan_csv(presence_tsv, separator='\t')
        .filter(pl.col('#kmer').is_in(used))
        .with_columns(
            pl.col('list_scrub_id')
              .str.split(',')
              .cast(pl.List(pl.UInt32))
              .list.set_difference(li_t)
              .alias('ids')
        )
        .with_columns(
            pl.col('ids').list.len().alias('n_presence'),
            pl.col('ids').list.sort().cast(pl.List(pl.String))
              .list.join(',').alias('list_scrub_id'),
        )
        .select(['#kmer', 'list_scrub_id', 'n_presence'])
        .collect(engine='streaming')
    )

    ann = (df_global_counts
           .filter(pl.col('#kmer').is_in(used))
           .join(presence, on='#kmer', how='left'))

    out = df_export.merge(_pl_to_pandas(ann), on='#kmer', how='left')
    out['n_presence'] = out['n_presence'].fillna(0).astype(int)
    out['list_scrub_id'] = out['list_scrub_id'].fillna('')
    return out


# ── plots ───────────────────────────────────────────────────────────────

def plot_scrub_counts(df_global_counts, basename, output_dir):
    """Waterfall of how many kmers carry each total scrub count.

    Written twice: the full range, and a zoom on the first 20 counts where
    the informative tail actually lives.
    """
    df_gl = df_global_counts.to_pandas().drop(columns=['reference_count']).set_index('#kmer')
    count_hist = (df_gl.sum(axis=1).value_counts()
                  .sort_index()
                  .rename_axis('total_count')
                  .reset_index(name='n_kmers'))

    zero = count_hist.loc[count_hist['total_count'] == 0, 'n_kmers']

    for x_max, suffix in ((5000, ''), (20, '.zoom')):
        # subset rather than just setting range_x, so the log y-axis
        # autoscales to what is visible instead of to the whole tail
        sub = count_hist.loc[count_hist['total_count'] <= x_max]
        fig = px.scatter(sub, x='total_count', y='n_kmers',
                         log_y=True, template='simple_white',
                         range_x=[-0.1, x_max], title=basename)
        if len(zero):
            fig.add_hline(y=zero.iloc[0], line_width=3,
                          line_dash='dash', line_color='grey')
        fig.write_image(
            os.path.join(output_dir, f'{basename}.scrub_counts{suffix}.svg'))


def plot_coverage_histogram(summary_tsv, basename, output_dir):
    df_hist = pd.read_csv(summary_tsv, sep='\t')
    fig = px.histogram(df_hist, x='coverage_pct', log_y=True,
                       color='is_in_global', histfunc='count',
                       template='simple_white', range_x=[0, 1],
                       title=f'{basename} — coverage_pct distribution')
    fig.update_layout(width=800, height=500)
    fig.write_image(os.path.join(output_dir, f'{basename}.histogram_scrub_db.svg'))


# ── pair selection and export ───────────────────────────────────────────

def finish_pair_workflow(args, basename, records, selected_singletons,
                         pair_kmers, df_locations_singletons,
                         df_global_counts):
    """Reduce the pair kmers to a set independent of each other and of the
    committed singletons, keep only pairs whose two members both survive, and
    export the merged locations for strain_detect."""
    pair_glob = os.path.join(args.output_dir,
                             f'{basename}.inform_kmer_pairs.part*.parquet')
    filtered_path = os.path.join(args.output_dir,
                                 f'{basename}.inform_kmer_pairs.pairs.parquet')

    df_locations_all = df_locations_singletons
    selected_pair_kmers = set()

    if pair_kmers:
        print(f'Locating pair kmers on genome ({len(pair_kmers):,})')
        df_locations_pairs, _ = build_mapped_kmers_ahocorasick(
            records, pair_kmers, terminal_dist=args.terminal_dist)
        print('dropping terminal kmers')
        df_locations_pairs = df_locations_pairs.loc[
            df_locations_pairs['terminal_kmer'] == False]

        df_locations_all = pd.concat(
            [df_locations_singletons, df_locations_pairs], ignore_index=True)

        print('finding overlapping kmers for pair selection')
        selected_pair_kmers = set(select_independent_with_fixed(
            df_locations_all, selected_singletons,
            set(df_locations_pairs['#kmer']), args.kmer_overlap))
        print(f'Selected pair kmers: {len(selected_pair_kmers):,}')

    # keep only pairs where BOTH members survived selection
    pair_parts = sorted(glob.glob(pair_glob))
    if pair_parts and selected_pair_kmers:
        sel = pl.Series(sorted(selected_pair_kmers))
        (
            pl.scan_parquet(pair_parts, low_memory=True)
            .filter(pl.col('kmerA').is_in(sel) & pl.col('kmerB').is_in(sel))
            .sink_parquet(filtered_path, compression='zstd')
        )
    else:
        print('No informative pairs for this strain — writing empty pairs parquet',
              flush=True)
        write_empty_pairs(filtered_path)

    # A selected kmer whose partner was dropped now appears in no pair at all.
    # Read back what survived so the exported locations match the pairs file.
    used_pair_kmers = set()
    if os.path.exists(filtered_path) and os.path.getsize(filtered_path) > 0:
        for col in ('kmerA', 'kmerB'):
            used_pair_kmers.update(
                pl.read_parquet(filtered_path, columns=[col])
                  .get_column(col).unique().to_list())
            gc.collect()
    dropped = len(selected_pair_kmers) - len(used_pair_kmers)
    if dropped > 0:
        print(f'  {dropped:,} selected pair kmers ended up in no surviving pair '
              f'and are excluded from the export')

    if pair_parts:
        if os.path.exists(filtered_path) and os.path.getsize(filtered_path) > 0:
            for p in pair_parts:
                os.remove(p)
            print(f'Wrote {filtered_path} and removed {len(pair_parts)} part files',
                  flush=True)
        else:
            print('WARNING: final parquet missing or empty — keeping part files',
                  flush=True)

    # merged export for strain_detect
    used = set(selected_singletons) | used_pair_kmers
    df_export = df_locations_all.loc[df_locations_all['#kmer'].isin(used)].copy()
    df_export['origin'] = np.where(
        df_export['#kmer'].isin(selected_singletons), 'singleton', 'pair')
    df_export = df_export.sort_values(['contig_id', 'kmer_position'])

    li_t = exclusion_list(args.counts_summary, args.similarity_t)
    print(f'Annotating export with scrub counts and presence lists '
          f'({len(li_t):,} samples excluded from the lists)')
    df_export = annotate_kmers(df_export, df_global_counts,
                               args.counts_individual, li_t)

    out_path = os.path.join(args.output_dir, f'{basename}.rare_kmers_mapped.tsv.gz')
    df_export.to_csv(out_path, sep='\t', index=False, compression='gzip')

    n_s = int((df_export['origin'] == 'singleton').sum())
    n_p = int((df_export['origin'] == 'pair').sum())
    print(f'Total kmers for strain_detect: {len(df_export):,} '
          f'({n_s:,} singleton, {n_p:,} pair)')
    print(f'Wrote {out_path}', flush=True)
    return df_export


# ── main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Select strain-informative kmers and build kmer pairs.')
    parser.add_argument('--genome', required=True,
                        help='Genome FASTA (.fna or .fna.gz)')
    parser.add_argument('--counts_global', required=True,
                        help='Global kmer counts TSV from kmer_scrub_count_individual')
    parser.add_argument('--counts_individual', required=True,
                        help='Inverted presence index (#kmer, list_scrub_id)')
    parser.add_argument('--counts_summary', required=True,
                        help='Per-sample summary TSV')
    parser.add_argument('--output_dir', default='.')
    parser.add_argument('--basename', default=None,
                        help='Output basename (default: derived from --genome)')
    parser.add_argument('--threads', type=int, default=20)
    parser.add_argument('--max_kmers', type=int, default=20000,
                        help='Target kmer count; below this, pairs are also '
                             'built from non-informative kmers')
    parser.add_argument('--kmer_overlap', type=float, default=0.8,
                        help='Max fraction of overlap allowed between two '
                             'selected kmers (0-1)')
    parser.add_argument('--terminal_dist', type=int, default=300,
                        help='Distance from contig ends to flag terminal kmers')
    parser.add_argument('--similarity_t', type=float, default=None,
                        help='Optional coverage cutoff for excluding samples; '
                             'default uses is_in_global from the summary')
    args = parser.parse_args()

    basename = args.basename if args.basename else strain_name_from_path(args.genome)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── global counts ──────────────────────────────────────────────────
    df_global_counts = pl.read_csv(
        args.counts_global, separator='\t',
        schema_overrides={'reference_count': pl.UInt32,
                          'pangenome_count': pl.UInt32,
                          'metagenome_count': pl.UInt32,
                          'drug_count': pl.UInt32})
    print(f'Total kmers: {len(df_global_counts):,}')

    plot_scrub_counts(df_global_counts, basename, args.output_dir)
    plot_coverage_histogram(args.counts_summary, basename, args.output_dir)

    print('Removing kmers with count >1 in reference genome')
    df_global_counts = df_global_counts.filter(pl.col('reference_count') == 1)
    print(f'  remaining: {len(df_global_counts):,}')

    if 'drug_count' in df_global_counts.columns:
        print('Removing kmers present in drug scrub')
        df_no_drugs = df_global_counts.filter(pl.col('drug_count') == 0)
    else:
        print('No drug scrub performed')
        df_no_drugs = df_global_counts
    print(f'  remaining: {len(df_no_drugs):,}')

    is_informative = (pl.col('metagenome_count') == 0) & (pl.col('pangenome_count') == 0)
    df_inform_singletons = df_no_drugs.filter(is_informative)
    df_non_inform_singletons = df_no_drugs.filter(~is_informative)
    print(f'Informative singletons: {len(df_inform_singletons):,} | '
          f'non-informative: {len(df_non_inform_singletons):,}')

    print(f'Loading genome: {args.genome}')
    records = load_genome(args.genome)

    # ── singletons: map, reduce to non-overlapping, pair ───────────────
    singleton_kmers = set(df_inform_singletons['#kmer'].to_list())
    print(f'Locating unique kmers on genome ({len(singleton_kmers):,})')
    df_locations_singletons, _ = build_mapped_kmers_ahocorasick(
        records, singleton_kmers, terminal_dist=args.terminal_dist)
    print('dropping terminal kmers')
    df_locations_singletons = df_locations_singletons.loc[
        df_locations_singletons['terminal_kmer'] == False]

    print('finding overlapping kmers for kmer selection')
    dict_overlap = find_overlap_kmer_fast(df_locations_singletons, max=args.kmer_overlap)
    selected = max_independent_kmers_greedy_heap(dict_overlap)
    print(f'Selected {len(selected):,} kmers from unique at overlap {args.kmer_overlap}')

    # create_all_pairs subsamples internally with an unseeded random.sample, so
    # cap here instead: the exported kmers must be the ones that got paired.
    if len(selected) > args.max_kmers:
        print(f'Capping {len(selected):,} singletons to --max_kmers '
              f'({args.max_kmers:,})')
        selected = sorted(random.Random(42).sample(sorted(selected), args.max_kmers))

    print('creating pairs of selected kmers')
    # create_all_pairs joins output_dir with `basename` directly, so this has
    # to be the full filename including the extension.
    create_all_pairs(selected,
                     output_dir=args.output_dir,
                     basename=f'{basename}.inform_kmer_pairs.singletons.parquet',
                     batch_size=1_000_000,
                     max_kmers=args.max_kmers)

    # ── top up from non-informative kmers if the set is thin ───────────
    pair_kmers = set()
    if len(selected) < args.max_kmers:
        print('creating pairs from non informative kmers')
        print(f'max kmers for generation: {args.max_kmers} *3 ')
        kmer_pairs_from_presence(
            args.counts_individual, args.counts_summary,
            args.output_dir,
            basename=basename,
            df_keep=df_non_inform_singletons,
            presence_t=10,
            similarity_t=args.similarity_t,
            n_workers=args.threads,
            max_for_pairs=args.max_kmers * 3)

        pair_glob = os.path.join(args.output_dir,
                                 f'{basename}.inform_kmer_pairs.part*.parquet')
        for path in sorted(glob.glob(pair_glob)):
            for col in ('kmerA', 'kmerB'):
                pair_kmers.update(
                    pl.read_parquet(path, columns=[col])
                      .get_column(col).unique().to_list())
            gc.collect()
        print(f'Pair kmers: {len(pair_kmers):,}')
    else:
        print(f'{len(selected):,} singletons already meet --max_kmers '
              f'({args.max_kmers:,}); skipping non-informative pairs')

    finish_pair_workflow(args, basename, records, set(selected),
                         pair_kmers, df_locations_singletons, df_no_drugs)


if __name__ == '__main__':
    main()