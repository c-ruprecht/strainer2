#!/usr/bin/env bash
# ============================================================
# MGnify UHGG v2.0.2 -> download all genomes + sourmash RocksDB
# Uses genomes-all_metadata.tsv as download manifest.
# RESUMABLE: computes missing + corrupt files locally (no per-URL
# network round-trips) and only fetches what's actually needed.
# Safe to re-run as many times as you like.
# Run on metrica; BASE_DIR should be on your NVMe.
# ============================================================
set -euo pipefail

BASE_DIR="/metrica/databases/genomes/mgnify_uhgg_v2.0.2"
FTP_BASE="https://ftp.ebi.ac.uk"
SKETCH_DIR="${BASE_DIR}/sketches"
GENOMES_DIR="${BASE_DIR}/genomes"
DB_PATH="${BASE_DIR}/uhgg_v2.0.2_k31_s1000.rocksdb"
ZIP_PATH="${BASE_DIR}/uhgg_v2.0.2_k31_s1000.zip"
MANIFEST_CSV="${BASE_DIR}/manysketch_manifest.csv"
METADATA_URL="https://ftp.ebi.ac.uk/pub/databases/metagenomics/mgnify_genomes/human-gut/v2.0.2/genomes-all_metadata.tsv"
URL_LIST="${BASE_DIR}/all_ftp_urls.txt"

N_DOWNLOAD=12    # parallel wget jobs
N_SKETCH=32      # sourmash threads (all cores on the 5950X)
N_TEST=32        # parallel gzip -t integrity tests

WORK="${BASE_DIR}/.resume"   # scratch for the diff lists
mkdir -p "${GENOMES_DIR}" "${SKETCH_DIR}" "${WORK}"

# ── Step 1: Metadata manifest (skip if present) ──────────────
echo "[1/6] Fetching metadata manifest..."
wget -q -c "${METADATA_URL}" -O "${BASE_DIR}/genomes-all_metadata.tsv"

# ── Step 2: Build the full URL list (skip if present) ────────
echo "[2/6] Parsing FTP paths from metadata..."
if [[ ! -s "${URL_LIST}" ]]; then
  FTP_COL=$(head -1 "${BASE_DIR}/genomes-all_metadata.tsv" \
    | tr '\t' '\n' \
    | grep -in 'ftp\|download' \
    | head -1 \
    | cut -d: -f1)
  echo "  FTP column detected: ${FTP_COL}"

  tail -n +2 "${BASE_DIR}/genomes-all_metadata.tsv" \
    | cut -f"${FTP_COL}" \
    | sed "s|^/pub|${FTP_BASE}/pub|" \
    > "${URL_LIST}"
else
  echo "  Reusing existing ${URL_LIST}"
fi

TOTAL=$(wc -l < "${URL_LIST}")
echo "  Total genomes in manifest: ${TOTAL}"

# ── Step 3: Compute what's missing or corrupt (all local) ────
echo "[3/6] Diffing against files on disk..."

# expected basenames
sed 's|.*/||' "${URL_LIST}" | sort -u > "${WORK}/expected.txt"



# missing = expected - present
comm -23 "${WORK}/expected.txt" "${WORK}/present.txt" > "${WORK}/need.txt"
N_NEED=$(wc -l < "${WORK}/need.txt")
echo "  Need to download: ${N_NEED}"

# map needed basenames back to full URLs
awk -F'/' 'NR==FNR{need[$1]=1; next} ($NF in need)' \
  "${WORK}/need.txt" "${URL_LIST}" > "${WORK}/need_urls.txt"

# ── Step 4: Download only what's needed ──────────────────────
if [[ "${N_NEED}" -gt 0 ]]; then
  echo "[4/6] Downloading ${N_NEED} files (${N_DOWNLOAD} parallel)..."
  cat "${WORK}/need_urls.txt" \
    | xargs -P "${N_DOWNLOAD}" -I{} wget --quiet --continue --tries=3 \
        --no-host-directories --cut-dirs=6 \
        --directory-prefix="${GENOMES_DIR}" "{}"
  echo "  Download pass complete."

  # re-verify: anything still in need.txt after a full pass is likely
  # server-side missing (404) rather than just unfinished
  find "${GENOMES_DIR}" -maxdepth 1 -name '*.gff.gz' -printf '%f\n' \
    | sort -u > "${WORK}/present.txt"
  comm -23 "${WORK}/expected.txt" "${WORK}/present.txt" > "${WORK}/still_missing.txt"
  N_STILL=$(wc -l < "${WORK}/still_missing.txt")
  if [[ "${N_STILL}" -gt 0 ]]; then
    echo "  WARNING: ${N_STILL} files still missing after download."
    echo "  See ${WORK}/still_missing.txt — sample one URL by hand to check for 404s."
  fi
else
  echo "[4/6] Nothing to download — all files present and valid."
fi

# ── Step 5: Build manysketch manifest CSV ────────────────────
echo "[5/6] Building manysketch manifest CSV..."
# NOTE: this sketches *.gff.gz (GFF annotation files). For DNA k-mer
# sketches you almost certainly want the genomic FASTA (.fna/.fa).
# Confirm the manifest points at FASTA before trusting the index.
find "${GENOMES_DIR}" -maxdepth 1 -name '*.gff.gz' | sort \
  | awk -F'/' '{
      fname = $NF
      acc = fname
      sub(/\.gff\.gz$/, "", acc)
      print acc "," $0 ","
    }' \
  > "${MANIFEST_CSV}.tmp"

{ echo "name,genome_filename,protein_filename"; cat "${MANIFEST_CSV}.tmp"; } > "${MANIFEST_CSV}"
rm -f "${MANIFEST_CSV}.tmp"

NFOUND=$(tail -n +2 "${MANIFEST_CSV}" | wc -l)
echo "  Genomes in manifest: ${NFOUND}"

# ── Step 6: manysketch + RocksDB ─────────────────────────────
echo "[6/6] Running manysketch (k=31, scaled=1000)..."
sourmash scripts manysketch \
  "${MANIFEST_CSV}" \
  -p "dna,k=31,scaled=1000" \
  -o "${ZIP_PATH}" \
  --cores "${N_SKETCH}" \
  --force

echo "  Building RocksDB index..."
sourmash scripts index \
  "${ZIP_PATH}" \
  -o "${DB_PATH}" \
  --cores "${N_SKETCH}"

echo "Done! RocksDB at: ${DB_PATH}"