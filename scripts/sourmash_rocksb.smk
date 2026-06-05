# ============================================================
# Snakemake: FASTA -> sourmash RocksDB (batched, 1000 per batch)
# Usage: snakemake -j <cores> --scheduler greedy \
#          --config fasta_dir=<path>
# ============================================================


from pathlib import Path
import glob
import math

BASE_DIR  = "/metrica/databases/genomes/mgnify_uhgg_v2.0.2"
FASTA_DIR = config.get("fasta_dir", f"{BASE_DIR}/genomes_fasta")
ROCKSDB   = f"{BASE_DIR}/uhgg_v2.0.2_k31_s1000.rocksdb"
SIG_DIR   = f"{BASE_DIR}/sigs"
K         = config.get("k", 31)
SCALED    = config.get("scaled", 1000)
BATCH_SIZE = config.get("batch_size", 1000)

# ── collect files ─────────────────────────────────────────────
FASTAS = sorted(
    glob.glob(f"{FASTA_DIR}/**/*.fasta.gz", recursive=True) +
    glob.glob(f"{FASTA_DIR}/**/*.fna.gz",   recursive=True)
)

if not FASTAS:
    raise ValueError(f"No FASTA files found in {FASTA_DIR}")

SAMPLES = [Path(f).name.replace(".fasta.gz", "").replace(".fna.gz", "")
           for f in FASTAS]

FASTA_MAP = {Path(f).name.replace(".fasta.gz","").replace(".fna.gz",""): f
             for f in FASTAS}

# ── batches ───────────────────────────────────────────────────
N_BATCHES = math.ceil(len(SAMPLES) / BATCH_SIZE)
BATCH_IDS = [str(i).zfill(6) for i in range(N_BATCHES)]

def batch_samples(batch_id):
    idx = int(batch_id)
    return SAMPLES[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE]

def batch_fastas(batch_id):
    return [FASTA_MAP[s] for s in batch_samples(batch_id)]

def batch_sigs(batch_id):
    return [f"{SIG_DIR}/{s}.sig.gz" for s in batch_samples(batch_id)]

# ── rules ─────────────────────────────────────────────────────

rule all:
    input: ROCKSDB

# Write one manysketch CSV per batch
rule batch_manifest:
    output:
        csv = f"{BASE_DIR}/batches/{{batch_id}}/manifest.csv"
    params:
        batch_id = lambda wc: wc.batch_id
    run:
        samples = batch_samples(params.batch_id)
        fastas  = batch_fastas(params.batch_id)
        Path(output.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(output.csv, "w") as f:
            f.write("name,genome_filename,protein_filename\n")
            for sample, fasta in zip(samples, fastas):
                f.write(f"{sample},{fasta},\n")

# Sketch one batch -> one zip per batch
rule sketch_batch:
    input:
        csv  = f"{BASE_DIR}/batches/{{batch_id}}/manifest.csv"
    output:
        zip  = f"{BASE_DIR}/batches/{{batch_id}}/sigs.zip"
    params:
        k      = K,
        scaled = SCALED,
        jobs   = config.get("jobs", 4)   # threads per batch job
    shell:
        """
        sourmash sketch dna \
            --from-file <(tail -n +2 {input.csv} | cut -d',' -f2) \
            -p k={params.k},scaled={params.scaled} \
            --name-from-first \
            -o {output.zip}
        """

rule index:
    input:
        zips = expand(f"{BASE_DIR}/batches/{{batch_id}}/sigs.zip",
                      batch_id=BATCH_IDS)
    output:
        db = directory(ROCKSDB)
    params:
        k    = K,
        jobs = config.get("jobs", 32)
    shell:
        """
        sourmash scripts index \
            {input.zips} \
            -o {output.db} \
            -k {params.k} \
            --cores {params.jobs}
        """