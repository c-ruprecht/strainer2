import pandas as pd
import plotly.express as px
import numpy as np
import os
import polars as pl
import glob
import gc
import argparse
from kmer_pairs import get_pair_hits_streaming
import warnings
warnings.filterwarnings("ignore", module="kaleido")


def read_kmer_hits(path_to_kmer_hits, path_to_genome_map):
    #gets total reads 
    df = pd.read_csv(path_to_kmer_hits, sep = '\t')
    df_total_reads = df.loc[df['#kmer']=='total_evaluated'].copy().set_index('#kmer').T
    #df_total_reads = df_total_reads.reset_index()
    dict_total_reads = df_total_reads['total_evaluated'].to_dict()
    # get kmerhits and map locations
    df_kmer_hits = df.loc[df['#kmer']!= 'total_evaluated'].copy()
    df_locations = pd.read_csv(path_to_genome_map, sep = '\t')

    df_merge = pd.merge(df_kmer_hits, df_locations, on = ['#kmer'], how = 'left')
    df_merge = df_merge.set_index(df_locations.columns.to_list()).stack()
    df_merge = df_merge.reset_index()
    df_merge = df_merge.rename(columns={'level_8': 'sample', 0: 'count'})
    df_merge['strain'] = str(path_to_genome_map).split('/')[-1].split('.rare_kmers_mapped.')[0]
    return df_merge, dict_total_reads

def visualize_count_map(df_hits_stack, df_coverage, outdir, min_coverage = 0.02):
    #visualize coverage for strains with more coverage of unique kmers than threshhold
    for sample in df_coverage.loc[df_coverage['coverage_kmer_single']>min_coverage]['sample'].unique():
        os.makedirs(outdir+f'/plots/{sample}', exist_ok=True)
        print(sample)
        df_sample = df_hits_stack.loc[df_hits_stack['sample']==sample]
        fig = px.box(df_sample,
                     x= 'contig_id',
                     y = 'count_per10B_kmers',
                     template = 'simple_white',
                     color = 'sample',
                     title = df_sample['strain'].unique()[0]
                     )
        fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        fig.write_image(outdir + f'/plots/{sample}/contig-box_{sample}.svg')
        fig = px.histogram(df_sample,
                x = 'count',
                #y = 'count',
                facet_col = 'sample',
                facet_col_wrap=4,
                template= 'simple_white',
                width = 1000)
        
        fig.write_image(outdir + f'/plots/{sample}/histogram-counts_{sample}.svg')
        
        os.makedirs(outdir+f'/plots/{sample}/contig_coverage_plots/', exist_ok=True)
        for contig, df_contig in df_sample.groupby('contig_id'):#speeds up for bad assemblies with a lot of contigs
            fig = px.scatter(df_contig,
                            x = 'kmer_position',
                            y = 'count',
                            color = 'contig_id',
                            facet_col = 'sample',
                            facet_col_wrap=4,
                            template= 'simple_white',
                            width = 1000)
            fig.write_image(outdir + f'/plots/{sample}/contig_coverage_plots/{contig}.svg')


    return

def main():
    parser = argparse.ArgumentParser(description='Calcualting coverage and depth with kmer locations')
    parser.add_argument('--hits', help='a target_strain .kmer_hits.tsv.gz')
    parser.add_argument('--inform_kmers')
    parser.add_argument('--output_dir', help='directory where output is saved')
    args = parser.parse_args()

    output_dir = args.output_dir
    #print(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Get informative kmer coverage
    df_samples = pl.read_csv(args.hits, separator="\t")
    
    map_glob = os.path.join(args.inform_kmers, "*.rare_kmers_mapped.*")
    matches = glob.glob(map_glob)
    if matches:
        df_mapped = pd.read_csv(matches[0], sep='\t')
        origin_counts = df_mapped['origin'].value_counts().to_dict()
        print(origin_counts)
    
    #get toal evaulated kmers
    row = df_samples.filter(pl.col("#kmer") == "total_evaluated").drop("#kmer")
    dict_total_reads = row.row(0, named=True)


    df_samples = df_samples.filter(pl.col("#kmer") != "total_evaluated")

    # drop this to check if missing samples reappear
    #pair_cols = ["#kmer"] + [col for col in df_samples.columns if col != "#kmer" and df_samples[col].sum() > 1]
    #df_samples = df_samples.select(pair_cols)
    

    ### Pairs
    print('getting pair coverage')
    #df_kmer_pairs = pl.read_parquet(os.path.join(args.inform_kmers, f"*.inform_kmer_pairs.parquet"))
    #df_cov_p = get_pair_hits(df_samples, df_kmer_pairs)
    pair_glob = os.path.join(args.inform_kmers, "*.inform_kmer_pairs.*.parquet")
    if glob.glob(pair_glob):
        df_cov_p = get_pair_hits_streaming(df_samples, pair_glob)
    
    print('getting only pair coverage')
    
    pair_glob = os.path.join(args.inform_kmers, "*.inform_kmer_pairs.pairs.parquet")
    if glob.glob(pair_glob):
        df_cov_pp = get_pair_hits_streaming(df_samples, pair_glob)
        # add pair_ prefix to all columns but sample
    print('getting only singleton coverage')
    
    pair_glob = os.path.join(args.inform_kmers, "*.inform_kmer_pairs.singletons.parquet")
    if glob.glob(pair_glob):
        df_cov_sp = get_pair_hits_streaming(df_samples, pair_glob)
        # add singleton_ prefix to all columns but sample

    # add prefix to non sample columns
    def add_prefix(df, prefix):
        return df.rename({c: f"{prefix}{c}" for c in df.columns if c != "sample"})
    
    df_cov_inform = (add_prefix(df_cov_p,  "combined_")
                    .join(add_prefix(df_cov_pp, "pairs_"),      on="sample", how="left")
                    .join(add_prefix(df_cov_sp, "singletons_"), on="sample", how="left")
                    .to_pandas()
            )
    
    #merge in total reads
    df_cov_inform['total_kmers_evaluated'] = df_cov_inform['sample'].map(dict_total_reads)
    #renames:
    drop_cols = ['pairs_strain', 'pairs_total_unique_kmers',
                  'pairs_observed_unique_kmers',	'pairs_unique_kmer_coverage','pairs_unique_kmer_count_mean',
                  'pairs_unique_kmer_count_std', 'singletons_strain','singletons_total_unique_kmers',
                  'singletons_observed_unique_kmers',	'singletons_unique_kmer_coverage',	'singletons_unique_kmer_count_mean'	,
                  'singletons_unique_kmer_count_std',
                  'combined_unique_kmer_count_std']
    
    df_cov_inform = df_cov_inform.rename(columns = {'combined_strain': 'strain',
                                                    'combined_total_unique_kmers': 'total_unique_kmers',
                                                    'combined_observed_unique_kmers': 'observed_unique_kmers',
                                                    'combined_unique_kmer_coverage': 'unique_kmer_coverage',
                                                    'combined_unique_kmer_count_mean':'unique_kmer_count_mean'})
    print(df_cov_inform)
    df_cov_inform.drop(columns = drop_cols, inplace = True)
    print(df_cov_inform.columns)

    df_cov_inform['singletons_individual_kmers_total'] = origin_counts['singleton']
    df_cov_inform['pairs_individual_kmers_total'] = origin_counts['pair']

       
    sort_cols = ['strain', 'sample', 'total_kmers_evaluated',
                 'total_unique_kmers','singletons_individual_kmers_total', 'pairs_individual_kmers_total',
                 'observed_unique_kmers',
                 'unique_kmer_count_mean','unique_kmer_coverage', 'combined_pairs_coverage', 'singletons_pairs_coverage', 'pairs_pairs_coverage',
                  'combined_pairs_total', 'singletons_pairs_total','pairs_pairs_total',
                 'combined_pairs_observed', 'singletons_pairs_observed', 'pairs_pairs_observed',
                 'combined_pairs_count_mean-min', 'combined_pairs_count_mean-mean', 'combined_pairs_count_mean-max',
                 'singletons_pairs_count_mean-min', 'singletons_pairs_count_mean-mean', 'singletons_pairs_count_mean-max', 
                 'pairs_pairs_count_mean-min', 'pairs_pairs_count_mean-mean','pairs_pairs_count_mean-max'
                 ]
    df_cov_inform[sort_cols].sort_values(['combined_pairs_coverage'], ascending= False).to_csv(output_dir+'/coverage_depth.tsv', index = False, sep = '\t')




if __name__ == '__main__':
    main()
