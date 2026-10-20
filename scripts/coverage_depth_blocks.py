import pandas as pd
import os
import json
import numpy as np
import scipy.sparse as sp
import argparse

def get_global_sample_coverage(df_hits, df_counts):

    df_counts = df_counts[['#kmer', 'pangenome_count', 'metagenome_count']].set_index('#kmer').sum(axis=1)
    df_counts = df_counts.reset_index(name= 'count')
    df = df_hits.set_index('#kmer')

    #subset hits for df_counts (align by kmer; kmers without a hits row count as 0)
    df = df.reindex(df_counts['#kmer'], fill_value=0)

    # get full kmer coverages based on columns
    df_cov_target = pd.DataFrame({'all_kmers_hit': (df > 0).sum(axis=0),
                                'all_kmers_total': len(df),
                                'coverage_all_kmers': round((df > 0).sum(axis=0)/len(df),4),
                                'depth_all_kmers': round(df.mean(axis=0),2),
                            })


    df_counts_uni = df_counts.loc[df_counts['count']==0].copy()
    rare_count = '<= 0'
    if len(df_counts_uni) < 1000:
        df_counts_uni = df_counts.loc[df_counts['count'] <= 1].copy()
        rare_count = '<= 1'
    df_counts_uni['rare_count'] = rare_count

    df_unique = df_hits.loc[df_hits['#kmer'].isin(df_counts_uni['#kmer'])].set_index('#kmer')

    df_cov_target_uni = pd.DataFrame({'rare_kmers_hit': (df_unique > 0).sum(axis=0),
                                'rare_kmers_total': len(df_unique),
                                'coverage_rare_kmers': round((df_unique > 0).sum(axis=0)/len(df_unique),4),
                                'depth_rare_kmers': round(df_unique.mean(axis=0),2),
                                'rare_count': rare_count,
                            })
    df_cov_target = pd.concat([df_cov_target, df_cov_target_uni], axis = 1)    

    return df_cov_target

def get_global_lineage_coverage(df_hits, df_counts):

    df_counts = df_counts[['#kmer', 'pangenome_count', 'metagenome_count']].set_index('#kmer').sum(axis=1)
    df_counts = df_counts.reset_index(name= 'count')
    df = df_hits.set_index('#kmer')
    #subset hits for df_counts (align by kmer; kmers without a hits row count as 0)
    df = df.reindex(df_counts['#kmer'], fill_value=0)

    # get full kmer coverages based on columns
    df_cov_target = pd.DataFrame({'lineage_kmers_hit': (df > 0).sum(axis=0),
                                'lineage_kmers_total': len(df),
                                'coverage_lineage_kmers': round((df > 0).sum(axis=0)/len(df),4),
                                'depth_lineage_kmers': round(df.mean(axis=0),2),
                            })
    return df_cov_target




def call_blocks(df_counts, df_hits):
    df_counts = df_counts[['#kmer','block_id', 'list_scrub_id','block_size']].copy()
    # empty scrub lists come in as NaN -> ''
    df_counts['list_scrub_id'] = df_counts['list_scrub_id'].fillna('').astype(str)

    samples = [c for c in df_hits.columns if c != '#kmer']

    # per-block metadata and kmer totals come from the counts table
    block_meta = (df_counts.drop_duplicates('block_id')
                  .set_index('block_id')[['list_scrub_id', 'block_size']])
    block_meta['list_scrub_id'] = block_meta['list_scrub_id'].astype('category')
    block_total = df_counts.groupby('block_id').size()

    # inner merge: kmers without a hits row are zeros, accounted for via block_total
    df = pd.merge(df_hits, df_counts[['#kmer', 'block_id']], on='#kmer', how='inner')

    block_hit = ((df[samples] > 0).groupby(df['block_id']).sum()
                 .reindex(block_total.index, fill_value=0))
    block_sum = (df[samples].groupby(df['block_id']).sum()
                 .reindex(block_total.index, fill_value=0))
    del df

    block_coverage = block_hit.div(block_total, axis=0).round(4)
    block_depth = block_sum.div(block_total, axis=0).round(2)

    df_gpd = pd.concat({'block_hit': block_hit,
                        'block_coverage': block_coverage,
                        'block_depth': block_depth}, axis=1)
    df_gpd = df_gpd.stack(level=-1, future_stack=True).rename_axis(['block_id', 'sample']).reset_index()
    df_gpd['sample'] = df_gpd['sample'].astype('category')
    df_gpd = df_gpd.join(block_meta, on='block_id')
    df_gpd['block_total'] = df_gpd['block_id'].map(block_total)

    return df_gpd[['block_id', 'list_scrub_id', 'block_size', 'sample',
                   'block_hit', 'block_coverage', 'block_depth', 'block_total']]

def get_block_pair_coverage(df_called_blocks, cov_t=0.5):
    # blocks x samples coverage matrix
    df_cov = (df_called_blocks
              .pivot(index='block_id', columns='sample', values='block_coverage')
              .fillna(0))
    df_cov.columns = df_cov.columns.astype(str)
    C = df_cov.to_numpy(dtype=np.float64)                   # B x S
    B = C.shape[0]

    # block -> scrub ids, aligned to df_cov row order; empty -> no entries -> disjoint with all
    scrub = (df_called_blocks.drop_duplicates('block_id')
             .set_index('block_id')['list_scrub_id']
             .reindex(df_cov.index).astype(object).fillna('').astype(str))
    ex = (pd.Series(scrub.to_numpy(), index=np.arange(B))
          .str.split(',').explode().str.strip())
    ex = ex[ex.notna() & (ex != '')]
    cols, ids = pd.factorize(ex.to_numpy())

    # sparse incidence (blocks x scrub ids) -> sparse upper-triangle overlap
    M = sp.csr_matrix((np.ones(len(cols), np.float32), (ex.index.to_numpy(), cols)),
                      shape=(B, max(len(ids), 1)))
    O = sp.triu(M @ M.T, k=1, format='csr')
    O.eliminate_zeros()
    O.data[:] = 1.0

    # disjoint pairs = all pairs - overlapping pairs
    P = B * (B - 1) // 2 - O.nnz
    print(f"{P} total disjoint pairs from {B} blocks")

    # sum over disjoint pairs of F[a]*F[b], without materialising pairs x samples
    def pair_sum(F):
        tot = F.sum(axis=0)
        s_all = (tot * tot - (F * F).sum(axis=0)) / 2.0
        s_ov = ((O @ F) * F).sum(axis=0)
        return np.clip(s_all - s_ov, 0, None)

    def pmean(x):
        return x / P if P else np.zeros(C.shape[1])

    pairs_observed = np.rint(pair_sum((C >= cov_t).astype(np.float64))).astype(np.int64)

    df_pair_cov = pd.DataFrame({
        'sample':          df_cov.columns,
        'blocks_total':    B,
        'blocks_observed': (C >= cov_t).sum(axis=0),
        'pairs_total':     P,
        'pairs_observed':  pairs_observed,
        'pair_cov_prod':   pmean(pair_sum(C)),
        'pair_cov_geo':    pmean(pair_sum(np.sqrt(C))),
    })
    df_pair_cov['pairs_coverage'] = pmean(pairs_observed.astype(np.float64))
    return df_pair_cov


def main():
    parser = argparse.ArgumentParser(description='Calculating target/lineage coverage, block coverage and disjoint block pair coverage')
    parser.add_argument('--counts', help='a target_strain .rare_kmers.mapped.tsv.gz file (kmer_blocks)', required= True)
    parser.add_argument('--hits', help='a target_strain .kmer_hits.tsv.gz', required= True)
    parser.add_argument('--output_dir', help='directory where output is saved', required= True)
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

    df_cov_depth.to_csv(os.path.join(outpath, "coverage_depth.tsv"), sep='\t', index = False)
    if args.write_blocks:
        df_gpd.to_csv(os.path.join(outpath, "block_coverage.tsv.gz"), sep='\t', index=False)
    print(f'Saved coverage dataframe to \n {outpath}')


if __name__ == '__main__':
    main()