import pandas as pd
import os
import argparse
import numpy as np


def get_global_sample_coverage(df_hits, df_counts):

    df_counts = df_counts[['#kmer', 'pangenome_count', 'metagenome_count']].set_index('#kmer').sum(axis=1)
    df_counts = df_counts.reset_index(name='count')
    df = df_hits.set_index('#kmer')

    # subset hits for target kmers (kmers without hits count as 0)
    df = df.reindex(df_counts['#kmer']).fillna(0)

    # get full kmer coverages based on columns
    df_cov_target = pd.DataFrame({'all_kmers_hit': (df > 0).sum(axis=0),
                                  'all_kmers_total': len(df),
                                  'coverage_all_kmers': round((df > 0).sum(axis=0) / len(df), 4),
                                  'depth_all_kmers': round(df.mean(axis=0), 2),
                                  })

    df_counts_uni = df_counts.loc[df_counts['count'] == 0].copy()
    rare_count = '<= 0'
    if len(df_counts_uni) < 1000:
        df_counts_uni = df_counts.loc[df_counts['count'] <= 1].copy()
        rare_count = '<= 1'
    df_counts_uni['rare_count'] = rare_count

    df_unique = df.loc[df.index.isin(df_counts_uni['#kmer'])]

    df_cov_target_uni = pd.DataFrame({'rare_kmers_hit': (df_unique > 0).sum(axis=0),
                                      'rare_kmers_total': len(df_unique),
                                      'coverage_rare_kmers': round((df_unique > 0).sum(axis=0) / len(df_unique), 4),
                                      'depth_rare_kmers': round(df_unique.mean(axis=0), 2),
                                      'rare_count': rare_count,
                                      })
    df_cov_target = pd.concat([df_cov_target, df_cov_target_uni], axis=1)

    return df_cov_target


def get_global_lineage_coverage(df_hits, df_counts):

    df = df_hits.set_index('#kmer')

    # subset hits for lineage kmers (kmers without hits count as 0)
    df = df.reindex(df_counts['#kmer']).fillna(0)

    df_cov_lineage = pd.DataFrame({'lineage_kmers_hit': (df > 0).sum(axis=0),
                                   'lineage_kmers_total': len(df),
                                   'coverage_lineage_kmers': round((df > 0).sum(axis=0) / len(df), 4),
                                   'depth_lineage_kmers': round(df.mean(axis=0), 2),
                                   })
    return df_cov_lineage


def call_blocks(df_counts, df_hits):
    df_counts = df_counts[['#kmer', 'block_id', 'list_scrub_id', 'block_size']].copy()
    # subset hits for df_counts
    df = pd.merge(df_hits, df_counts, on='#kmer', how='right')

    samples = [c for c in df_hits.columns if c != '#kmer']
    df[samples] = df[samples].fillna(0)

    keys = ['block_id', 'list_scrub_id', 'block_size']
    grp_keys = [df[k] for k in keys]

    block_total = df.groupby(keys).size()
    block_hit = (df[samples] > 0).groupby(grp_keys).sum()
    block_coverage = block_hit.div(block_total, axis=0).round(4)
    block_depth = df[samples].groupby(grp_keys).mean().round(2)

    df_gpd = pd.concat({'block_hit': block_hit,
                        'block_coverage': block_coverage,
                        'block_depth': block_depth}, axis=1)
    df_gpd = df_gpd.stack(level=-1, future_stack=True).rename_axis(keys + ['sample']).reset_index()
    df_gpd['block_total'] = df_gpd.set_index(keys).index.map(block_total)

    return df_gpd


def get_block_pair_coverage(df_called_blocks, cov_t=0.5):
    # blocks x samples coverage matrix
    df_cov = (df_called_blocks
              .pivot(index='block_id', columns='sample', values='block_coverage')
              .fillna(0))
    blocks = df_cov.index.to_numpy()
    C = df_cov.to_numpy()                                   # B x S

    # block -> scrub-id list, aligned to df_cov row order
    scrub = (df_called_blocks[['block_id', 'list_scrub_id']]
             .drop_duplicates('block_id')
             .set_index('block_id')['list_scrub_id']
             .reindex(blocks)
             .astype(str)
             .str.split(','))

    # incidence matrix (blocks x scrub ids) -> overlap counts between blocks
    ids = sorted(set().union(*scrub))
    col = {s: i for i, s in enumerate(ids)}
    M = np.zeros((len(blocks), len(ids)), dtype=np.int32)
    for i, lst in enumerate(scrub):
        M[i, [col[s] for s in lst]] = 1
    overlap = M @ M.T

    # disjoint pairs = zero overlap, upper triangle only
    ia, ib = np.nonzero(np.triu(overlap == 0, k=1))
    P = len(ia)
    print(f"{P} total disjoint pairs from {len(blocks)} blocks")

    cA, cB = C[ia], C[ib]                                   # P x S
    prod = cA * cB

    def pmean(x):
        return x.mean(axis=0) if P else np.zeros(C.shape[1])

    df_pair_cov = pd.DataFrame({
        'sample':          df_cov.columns,
        'blocks_total':    len(blocks),
        'blocks_observed': (C >= cov_t).sum(axis=0),
        'pairs_total':     P,
        'pairs_observed':  ((cA >= cov_t) & (cB >= cov_t)).sum(axis=0),
        'pairs_nonzero':   (prod > 0).sum(axis=0),
        'pair_cov_prod':   pmean(prod).round(4),
        'pair_cov_geo':    pmean(np.sqrt(prod)).round(4),
        'pair_cov_min':    pmean(np.minimum(cA, cB)).round(4),
    })
    df_pair_cov['pairs_coverage'] = (df_pair_cov['pairs_observed'] / P).round(4) if P else 0.0
    df_pair_cov['cov_t'] = cov_t
    return df_pair_cov


def main():
    parser = argparse.ArgumentParser(description='Calculating target/lineage coverage, block coverage and disjoint block pair coverage')
    parser.add_argument('--counts', help='a target_strain .rare_kmers.mapped.tsv.gz file (kmer_blocks)')
    parser.add_argument('--hits', help='a target_strain .kmer_hits.tsv.gz')
    parser.add_argument('--output_dir', help='directory where output is saved')
    parser.add_argument('--cov_t', type=float, default=0.5, help='minimal block coverage for a block to count as observed')
    parser.add_argument('--write_blocks', action='store_true', help='also write per block / sample coverage table')
    args = parser.parse_args()

    hits_path = args.hits
    counts_path = args.counts
    outpath = args.output_dir
    cov_t = args.cov_t

    os.makedirs(outpath, exist_ok=True)

    df_hits = pd.read_csv(hits_path, sep='\t')
    total_evaluated = df_hits.loc[df_hits['#kmer'] == 'total_evaluated'].drop(columns='#kmer').T.reset_index()
    total_evaluated.columns = ['sample', 'total_evaluated']
    print(total_evaluated)

    df_hits = df_hits.loc[df_hits['#kmer'] != 'total_evaluated'].copy()
    df_counts = pd.read_csv(counts_path, sep='\t')

    # split into targets and lineage
    df_target_cnt = df_counts.loc[df_counts['kmer_type'] == 'target'].copy()
    df_lineage_cnt = df_counts.loc[df_counts['kmer_type'] == 'lineage'].copy()

    # global target coverage (all + rare kmers)
    df_cov_target = get_global_sample_coverage(df_hits=df_hits, df_counts=df_target_cnt)
    df_cov_target = df_cov_target.reset_index(names='sample')

    # global lineage coverage
    df_cov_lineage = get_global_lineage_coverage(df_hits=df_hits, df_counts=df_lineage_cnt)
    df_cov_lineage = df_cov_lineage.reset_index(names='sample')

    df_cov_depth = pd.merge(df_cov_target, df_cov_lineage, on='sample', how='left')

    # block coverage and disjoint block pair coverage
    df_gpd = call_blocks(df_counts=df_target_cnt, df_hits=df_hits)
    df_pair_cov = get_block_pair_coverage(df_called_blocks=df_gpd, cov_t=cov_t)
    print(df_pair_cov)

    df_cov_depth = pd.merge(df_cov_depth, df_pair_cov, on='sample', how='left')
    df_cov_depth = pd.merge(df_cov_depth, total_evaluated, on='sample', how='left')

    df_cov_depth.to_csv(os.path.join(outpath, "coverage_depth.tsv"), sep='\t')
    if args.write_blocks:
        df_gpd.to_csv(os.path.join(outpath, "block_coverage.tsv.gz"), sep='\t', index=False)
    print(f'Saved coverage dataframe to \n {outpath}')


if __name__ == '__main__':
    main()