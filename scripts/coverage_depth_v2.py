import pandas as pd
import os
import argparse
import numpy as np
import json

def get_sample_coverage(df_hits, df_counts):

    df_counts = df_counts.drop(columns=['reference_count', 'drug_count']).set_index('#kmer').sum(axis=1)
    df_counts = df_counts.reset_index(name= 'count')
    df = df_hits.set_index('#kmer')

    # get full kmer coverages based on columns
    df_cov_depth = pd.DataFrame({'all_kmers_hit': (df > 0).sum(axis=0),
                                'all_kmers_total': len(df),
                                'coverage_all_kmers': round((df > 0).sum(axis=0)/len(df),4),
                                'depth_all_kmers': round(df.mean(axis=0),2),
                            })


    df_counts_uni = df_counts.loc[df_counts['count']==0].copy()
    rare_count = '=0'
    if len(df_counts_uni) < 1000:
        df_counts_uni = df_counts.loc[df_counts['count'] <= 1].copy()
        rare_count = '<=1'
    df_counts_uni['rare_count'] = rare_count

    df_unique = df_hits.loc[df_hits['#kmer'].isin(df_counts_uni['#kmer'])].set_index('#kmer')

    df_cov_depth_uni = pd.DataFrame({'rare_kmers_hit': (df_unique > 0).sum(axis=0),
                                'rare_kmers_total': len(df_unique),
                                'coverage_rare_kmers': round((df_unique > 0).sum(axis=0)/len(df_unique),4),
                                'depth_rare_kmers': round(df_unique.mean(axis=0),2),
                                'rare_count': rare_count,
                            })
    df_cov_depth = pd.concat([df_cov_depth, df_cov_depth_uni], axis = 1)    

    return df_cov_depth

def lw_inverse(coverage, model):
    """Depth that gives this coverage under coverage = a(1 - exp(-b*depth))."""
    frac = np.clip(np.asarray(coverage, dtype=float) / model['a'], 0, 1 - 1e-9)
    return -np.log1p(-frac) / model['b']


def lw_resid_sd(depth, model):
    sd = np.interp(np.log10(np.clip(depth, 1e-3, None)),
                   model['resid_sd_log10_depth'], model['resid_sd'])
    return np.maximum(sd, model['sd_floor'])


def scrub_sample_exp_depth(df_hits, dict_max_depth):
    df_hits = df_hits.set_index('#kmer')
    max_depth_s = pd.Series(dict_max_depth).reindex(df_hits.columns, fill_value=np.inf)

    df_keep = df_hits.where(df_hits.le(max_depth_s, axis=1))
    n_total = df_keep.notna().sum(axis=0)   # kmers kept per sample

    df_cov_depth_filt = pd.DataFrame({
        'filtered_kmers_hit': (df_keep > 0).sum(axis=0),
        'filtered_kmers_total': n_total,
        'coverage_filtered_kmers': ((df_keep > 0).sum(axis=0) / n_total).round(4),
        'depth_filtered_kmers': df_keep.mean(axis=0).round(2),
    })
    return df_cov_depth_filt


def main():
    # to do, basename + strain name addition
    parser = argparse.ArgumentParser(description='Calcualting coverage and depth with kmer locations')
    parser.add_argument('--counts', help='a global_counts.tsv.zst file')
    parser.add_argument('--hits', help='a target_strain .kmer_hits.tsv.gz')
    parser.add_argument('--output_dir', help='directory where output is saved')
    parser.add_argument('--model_path', help='path to lw model')
    parser.add_argument('--z', type = float, default = 1.0, help='standard deviations allowed for maximal depth calculation')
    args = parser.parse_args()

    hits_path = args.hits
    counts_path = args.counts
    outpath = args.output_dir
    model_path = args.model_path
    z = args.z


    df_hits = pd.read_csv(hits_path, sep = '\t')
    total_evaluated = df_hits.loc[df_hits['#kmer']=='total_evaluated'].drop(columns='#kmer').T.reset_index()#.to_dict()
    total_evaluated.columns = ['sample', 'total_evaluated']
    print(total_evaluated)

    df_hits = df_hits.loc[df_hits['#kmer'] !='total_evaluated'].copy()
    df_counts = pd.read_csv(counts_path, sep = '\t')

    # get sample coverage for all kemrs and unique/rare kmers
    df_cov_depth = get_sample_coverage(df_hits=df_hits, df_counts= df_counts)

    print(df_cov_depth)
    df_cov_depth = df_cov_depth.reset_index(names = 'sample')
    
    # scrub data
    with open(model_path) as mp:
        lw = json.load(mp)

    df_cov_depth['exp_depth'] = lw_inverse(df_cov_depth['coverage_rare_kmers'], lw)
    df_cov_depth['cov_high'] = (df_cov_depth['coverage_rare_kmers'] + z * lw_resid_sd(df_cov_depth['exp_depth'], lw))
    df_cov_depth['saturated'] = df_cov_depth['cov_high'] >= lw['a']
    df_cov_depth['max_depth'] = np.ceil(np.where(df_cov_depth['saturated'],
                                                np.inf,
                                                lw_inverse(df_cov_depth['cov_high'], lw)))

    dict_max_depth = df_cov_depth.set_index('sample')['max_depth'].to_dict()



    df_cov_depth_filt = scrub_sample_exp_depth(df_hits=df_hits, dict_max_depth=dict_max_depth)
    df_cov_depth.set_index('sample', inplace=True)
    df_cov_depth = pd.concat([df_cov_depth, df_cov_depth_filt], axis=1)
    
    
    df_cov_depth.reset_index().to_csv(os.path.join(outpath,"coverage_depth.tsv"), sep = '\t')
    print(f'Saved coverage dataframe to \n {outpath}')



if __name__ == '__main__':
    main()