#!/usr/bin/env bash
# ============================================================
# MGnify UHGG v2.0.2 → download all genomes + sourmash RocksDB
# Uses genomes-all_metadata.tsv as download manifest
# Run on metrica; BASE_DIR should be on your NVMe
# ============================================================
set -euo pipefail

BASE_DIR="/metrica/databases/genomes/mgnify_uhgg_v2.0.2/"
FTP_BASE="https://ftp.ebi.ac.uk"
SKETCH_DIR="${BASE_DIR}/sketches"
DB_PATH="${BASE_DIR}/uhgg_v2.0.2_k31_s1000.rocksdb"
METADATA_URL="https://ftp.ebi.ac.uk/pub/databases/metagenomics/mgnify_genomes/human-gut/v2.0.2/genomes-all_metadata.tsv"
N_DOWNLOAD=32    # parallel wget jobs
N_SKETCH=32     # sourmash sketch threads (all cores on 5950X)


# ── Step 4: Build manysketch input CSV ───────────────────────
echo "[4/5] Building manysketch manifest CSV..."

# Header direkt schreiben
echo "name,genome_filename,protein_filename" > "${BASE_DIR}/manysketch_manifest.csv"

# Dateien finden, Pfad säubern und in die CSV schreiben
find "${BASE_DIR}/genomes_fasta" -name "*.fasta.gz" | sort | while read -r filepath; do
    # Extrahiert nur den Dateinamen (z.B. MGYG000058069.fasta.gz)
    fname=$(basename "$filepath")
    # Schneidet .fasta.gz ab, um die reine Accession zu erhalten (z.B. MGYG000058069)
    acc="${fname%.fasta.gz}"
    # Schreibt: Accession, absoluter_Pfad, (leeres Protein-Feld)
    echo "${acc},${filepath}," >> "${BASE_DIR}/manysketch_manifest.csv"
done

NFOUND=$(tail -n +2 "${BASE_DIR}/manysketch_manifest.csv" | wc -l)
echo "  Found ${NFOUND} genomes in manifest"

# Prepend header
sed -i '1s/^/name,genome_filename,protein_filename\n/' "${BASE_DIR}/manysketch_manifest.csv"

NFOUND=$(tail -n +2 "${BASE_DIR}/manysketch_manifest.csv" | wc -l)
echo "  Found ${NFOUND} genomes in manifest"

# ── Step 5: manysketch ────────────────────────────────────────
echo "[5/5] Running manysketch (k=31, scaled=1000)..."

sourmash scripts manysketch \
  "${BASE_DIR}/manysketch_manifest.csv" \
  -p "dna,k=31,scaled=1000" \
  -o "${BASE_DIR}/uhgg_v2.0.2_k31_s1000.zip" \
  --cores "${N_SKETCH}" \
  --force

# ── Step 6: Build RocksDB ─────────────────────────────────────
echo "[6/5] Building RocksDB index..."

sourmash scripts index \
  "${BASE_DIR}/uhgg_v2.0.2_k31_s1000.zip" \
  -o "${DB_PATH}" \
  --cores "${N_SKETCH}"

echo "Done! RocksDB at: ${DB_PATH}"