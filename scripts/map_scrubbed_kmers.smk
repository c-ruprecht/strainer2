"""
Map each strain's genome against its matching scrubbed_kmers file.

Strains:     /metrica/scratch/strainer_dev/strains/strains/*.fna.gz
Kmers:       /metrica/scratch/strainer_dev/in_vitro/scrubbed_kmers/*.scrubbed_kmers.gz
Output:      /metrica/scratch/strainer_dev/in_vitro/kmer_mapped/

Run with:
    snakemake -s scripts/map_scrubbed_kmers.smk --cores <N>
"""

import os
import re

STRAINS_DIR   = "/metrica/scratch/strainer_dev/strains/strains"
KMERS_DIR     = "/metrica/scratch/strainer_dev/in_vitro/scrubbed_kmers"
OUTPUT_DIR    = "/metrica/scratch/strainer_dev/in_vitro/kmer_mapped"
MAP_SCRIPT    = os.path.join(workflow.basedir, "map_scrubbed_kmers.py")
CONDA_RUN     = "conda run -n strainer2_env"

os.makedirs(OUTPUT_DIR, exist_ok=True)

_genomes = {re.sub(r"\.fna\.gz$", "", f)
            for f in os.listdir(STRAINS_DIR)
            if f.endswith(".fna.gz")}

_kmer_strains = {re.sub(r"\.scrubbed_kmers\.gz$", "", f)
                 for f in os.listdir(KMERS_DIR)
                 if f.endswith(".scrubbed_kmers.gz")}

STRAINS = sorted(_genomes & _kmer_strains)


rule all:
    input:
        expand(os.path.join(OUTPUT_DIR, "{strain}.kmer_map.html"), strain=STRAINS)


rule map_kmers:
    input:
        genome = os.path.join(STRAINS_DIR, "{strain}.fna.gz"),
        kmers  = os.path.join(KMERS_DIR,   "{strain}.scrubbed_kmers.gz"),
    output:
        os.path.join(OUTPUT_DIR, "{strain}.kmer_map.html"),
    params:
        conda_run = CONDA_RUN,
        script    = MAP_SCRIPT,
    shell:
        "{params.conda_run} python {params.script} {input.genome} {input.kmers} {output}"
