# strainer2
kmer-based software for detecting bacterial strain genomes inside metagenomes

The Strainer2 software is an updated version of the [Strainer workflow](https://bitbucket.org/faithj02/strainer-metagenomics/src/master/README.md)
that identifies rare kmers in a bacterial strain genome and searches for those rare kmers in one or more metagenomes to determine if a strain
is present or absent in a metagenome.

The program can be used to track bacterial strains from defined live biotherapeutic products, sequenced culture isolates from fecal transplant donors and recipients, or (less tested) MAGs generated from metagenomes of microbial therapeutic recipients or other individuals that might have strain overlap.

If you use Strainer2, please cite:
[**Precise quantification of bacterial strains after fecal microbiota transplantation delineates long-term engraftment and explains outcomes.**](https://doi.org/10.1038/s41564-021-00966-0). Varun Aggarwala, Ilaria Mogno, Zhihua Li, Chao Yang, Graham Britton, Alice Chen-Liaw, Josephine Mitcham, Gerold Bongers, Dirk Gevers, Jose Clemente, Jean-Frederic Colombel, Ari Grinspan, and Jeremiah Faith. Nature Microbiology 2021.

## What's new in version 2
* computationally intensive programs are written in C
* all software features can be controlled through batch files or command line parameters to facilitate automation (e.g., snakemake)


## Installation
in the source directory is a simple Makefile. 

# make 
- git pull
- cd src && make

# create conda environment
- all packages needed to run python based filtering of kmers

conda env create -f strainer2_environment.yml

# database 
- In order to identify strain specific kmers we built and dereplicated a dataset of high quality genomes
- Space requirements genomes_t98: 181756 files, 227.5 GiB     
- download dir holds temporary files
- dest is where database will be built
- cleanup deletes raw files

python scripts/scrubdb_hf.py pull --repo cruprecht/scrubdb-derep-hq-comp99-cont05 --download-dir /path/to/tmp --dest /path/to/db --cleanup



### External libraries and C headers used
* zlib (must be installed on your system, but very likely it is)
* kseq.h (excellent fasta/fastq parser distributed with this software) [original site]()

### How to install
From inside the src directory type:
```
make
```

and it should generate the executables: kmer_scrub_count, strain_detect, and (not necessary for Strainer2 but a helpful tool) genome_compare.

There are a two python files in the scripts directory that are also needed.


## Overview of the algorithm
1. for all kmers in a strain, count how often each kmer occurs in a large set of genomes and metagenomes where the strain is not expected (i.e., unrelated individuals)
2. use the kmer counts from above to keep only the rare kmers and remove the other kmers. the more you scrub the lower the sensitivity and higher the precision. The default (empirically determined) is to keep 1% of the kmers
3. take the scrubbed kmer set and track the kmers in the metagenomes
4. determine if a strain is present or absent based on the frequency of the rare kmers in the metagenome

Note that the software examples and applications are currently set up for differentiating human gut strains in the human gut metagenome. The main ideas should apply to other body sites, but the learning of informative kmers would require metagenomes and genomes from the target sites.

# Little Helpers
## genome_compare

## genome_compare_presence

./genome_compare_presence -B /Volumes/metrica/scratch/derep_animalis_kmer_high_qual-comp99-cont05/kmer_primary/cl_Bifidobacterium_animalis__95b681/genomes -k 51 -o /Volumes/metrica/scratch/animalisk51

## tax_genome.py

Assigns a scrub-DB lineage (and its GTDB taxonomy) to one or more strain genomes.

```bash
python scripts/tax_genome.py \
    --genome_dir /path/to/genomes \
    --lineage_db /path/to/lineage_kmer_blocks.parquet \
    --output /path/to/outdir \
    --threads 16
```

Input is one of `--genome`, `--genome_dir` or `--genome_list` (one path per
line); FASTA may be gzipped. `--output` takes a TSV path or a directory, in
which case it writes `lineage_calls.tsv` there.

### Output
One row per genome:

| Column | Meaning |
|---|---|
| `genome` | genome file name without its extension |
| `lineage_id` | best-scoring lineage, or `not_found` |
| `gtdb_tax` | that lineage's GTDB taxonomy, or `not_found` |
| `n_labels` | distinct GTDB labels among the lineage's representatives |
| `lineage_purity` | share of representatives carrying the label the lineage is named after |
| `other_labels` | the other labels with their counts, trimmed to the part where the names disagree |
| `n_hits` | query k-mers found in that lineage |
| `n_lineage_kmers` | k-mers the lineage has in the database |
| `frac_lineage_hit` | `n_hits / n_lineage_kmers` |

`lineage_purity` near 1 means the lineage is one species and the call names it.
Low purity means the lineage is a complex and the call means "something in this
group"; `other_labels` says what else is in it. For example
`Collinsella_aerofaciens_M` has 93 labels at purity 0.06, so it is a
*Collinsella* complex rather than that species.

`not_found` means no query k-mer hit any lineage. This usually reflects how the
database was built rather than a problem with the genome: a lineage only exists
where the dereplicated set holds enough related genomes to define one. For
example *Streptococcus sanguinis_A* has only 3 representatives, which split too
unevenly to form a lineage, so it contributes no k-mers and any *S. sanguinis_A*
genome comes back `not_found`. `n_hits` is 0 and the two lineage columns stay
empty.

A runner-up scoring above half the best triggers an ambiguity warning in the
log; the TSV keeps only the best call.

## Description of programs and order of execution

### timing
Each strain is independently tracked, so the sequential steps for each strain genome can be run in parallel to each other. The number of parallel jobs will be the number of strains to be tracked. However, there are many other opportunities for splitting up further if desired. The current version finishes in 1-4 days depending on the number of metagenomes. Most of the time is spent during the initial kmer analysis or in the strain_detect step if there are many metagenomes. If the work flow was broken down not by strain but by strain and by metagenome (and the pieces combined back together at the end), the run time should run in a few hours instead of a few days. Lastly, the examples are watered down to finish in a few minutes. However, they are not sufficient for actual strain tracking (you would have many false positives if you do not study enough metagenomes/genomes to properly rank rare from common kmers).


### programs
Each program if run with no parameters will provide a Usage statement of all the parameters available.


* `kmer_scrub_count`
	* input: takes a strain genome (the one to be tracked) in fasta format as a parameter -r, a file with a list of metagenomes -B, and a file with a list of genomes -A (all in fasta format with or without gzip compression) and a name of a progress file -p. The program counts the frequency of the strain genome kmers inside the set of genomes and metagenomes. It updates the progress file when it completes a genome (to track progress) and prints the final kmer counts separately for the metagenome and genome. Optionally the program will accept an additional -C file that contains a set of genomes that are also strains from the same drug or FMT donor. These co-occuring strains can be quantified separately, as it might be useful to not have overlapping informative kmers used to track multiple strains in the same drug or FMT.
	* output: the complete list of the strains' kmers and the frequency of each kmer in the genome list, metagenome list, and (optionally) drug/FMT strain list.
* `kmer_scrub_filter.py`
	* input: the output file of `kmer_scrub_count` 
	* output: a file of informative (i.e., scrubbed) kmers using the threshold of --min_fraction the parameter --independent will scrub the pangenome and metagenome independently for a more stringent criterion; however this can lead to very few kmers if the intersection between these two is small. The joint scrubbing is the default.
	* joint scrubbing algorithm: to make sure differences in the number of genomes/metagenomes scrubbed aren't a main influence on the kmers that are scrubbed the kmers for each are converted into the frequencies (i.e., metagenome_kmer_count / sum_metagenome_kmer_counts; genome_kmer_count / sum_genome_kmer_count); then we go through all kmers in the genome and rank them by their MAX frequency between metagenome and genome (note that metagenome frequencies are more strongly distributed so they get they dominate the first kmers but then do not contribute much after that because the genome kmers are more evenly distributed; some interesting biology there...); the kmers are then removed from most frequent to least frequent until there are --min_fraction remaining
* `strain_detect`
	* input: takes a reference genome file -r, the informative kmer file from `kmer_scrub_filter`, and -B file with multiple metagenomics files. can handle paired-end, single-end, or paired end interleaved. you need to specify the file type in the batch file or you can run individually through the commandline.
	* output: a table where each line is an individual informative kmer found in a specific metagenome
* `coverage_depth.py`
	* input: the informative kmer matches from strain_detect
	* output: the proportion of a strains informative kmers covered and the average depth at which the kmers are covered which can be used to decide if a strain is present or absent in a sample




## Example Workflows

Example batch scripts (bash) or Snakemake files are available in the tests directory.


### Batch script (bash) workflow
from inside the test directory type:
```
sh example.sh
```


### Snakemake workflow
from inside the test directory type:

```
snakemake -s Snakemake.strain_detect -c all
```


