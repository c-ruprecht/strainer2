#!/usr/bin/env bash
# ============================================================
# MGnify UHGG v2.0.2 -> validate FASTAs -> sourmash RocksDB
#
# Step order:
#   1. (optional) full gzip CRC integrity check
#   2. content validation: header + DNA-alphabet (catches protein / empty / no-seq)
#   3. gate: stop and show rejects unless they are accepted
#   4. manysketch  (NO --force, so failures surface)
#   5. verify the zip with `unzip -t` before indexing
#   6. build RocksDB
#
# Env toggles:
#   RUN_GZIP_CHECK=1   also run `gzip -t` on every file (slow, full CRC)
#   FORCE_CONTINUE=1   sketch the VALID subset even if some files were rejected
#   K=31 SCALED=1000 JOBS=$(nproc)
# ============================================================
set -euo pipefail

BASE_DIR="/metrica/databases/genomes/mgnify_uhgg_v2.0.2"
FASTA_DIR="${BASE_DIR}/genomes_fasta"
MANIFEST="${BASE_DIR}/manysketch_manifest.csv"
REJECTS="${BASE_DIR}/fasta_rejects.tsv"
ZIP="${BASE_DIR}/uhgg_v2.0.2_k31_s1000.zip"
DB_PATH="${BASE_DIR}/uhgg_v2.0.2_k31_s1000.rocksdb"

K="${K:-31}"
SCALED="${SCALED:-1000}"
JOBS="${JOBS:-$(nproc)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo " UHGG v2.0.2  ->  sourmash RocksDB (k=${K}, scaled=${SCALED})"
echo " fasta dir : ${FASTA_DIR}"
echo " jobs      : ${JOBS}"
echo "============================================================"

[[ -d "$FASTA_DIR" ]] || { echo "ERROR: ${FASTA_DIR} not found"; exit 1; }

# ── Step 1: optional full gzip integrity ────────────────────
if [[ "${RUN_GZIP_CHECK:-0}" == "1" ]]; then
    echo "[1/6] gzip -t integrity check (full CRC)..."
    bad="$(find "$FASTA_DIR" -name '*.fasta.gz' -print0 \
           | xargs -0 -P "$JOBS" -I{} sh -c 'gzip -t "{}" 2>/dev/null || printf "%s\n" "{}"')"
    if [[ -n "$bad" ]]; then
        echo "  CORRUPT gzip files:"; printf '%s\n' "$bad"
        exit 1
    fi
    echo "  all gzips OK"
else
    echo "[1/6] skipping full gzip -t (set RUN_GZIP_CHECK=1 to enable)"
fi

# ── Step 2: content validation ──────────────────────────────
echo "[2/6] Validating FASTA content..."
set +e
python3 "${SCRIPT_DIR}/validate_fastas.py" \
    --fasta-dir "$FASTA_DIR" \
    --manifest  "$MANIFEST" \
    --rejects   "$REJECTS" \
    --jobs      "$JOBS"
vstatus=$?
set -e
if [[ $vstatus -eq 3 ]]; then
    echo "ERROR: zero valid genomes. Inspect ${REJECTS}."
    exit 1
fi

# ── Step 3: reject gate ─────────────────────────────────────
reject_n=$(( $(wc -l < "$REJECTS") - 1 ))   # minus header
valid_n=$((  $(wc -l < "$MANIFEST") - 1 ))
echo "[3/6] valid=${valid_n}  rejected=${reject_n}"

if [[ "$reject_n" -gt 0 ]]; then
    echo "  reject reasons:"
    tail -n +2 "$REJECTS" | cut -f2 | sort | uniq -c | sort -rn | sed 's/^/    /'
    echo "  sample rejects:"
    tail -n +2 "$REJECTS" | head -5 | sed 's/^/    /'
    if [[ "${FORCE_CONTINUE:-0}" != "1" ]]; then
        echo
        echo "  Stopping. Review ${REJECTS}."
        echo "  If these are expected, re-run with FORCE_CONTINUE=1 to sketch the"
        echo "  ${valid_n} valid genomes only."
        exit 2
    fi
    echo "  FORCE_CONTINUE=1 -> sketching the ${valid_n} valid genomes only."
fi

# ── Step 4: manysketch (no --force) ─────────────────────────
echo "[4/6] manysketch -> ${ZIP}"
sourmash scripts manysketch \
    "$MANIFEST" \
    -p "dna,k=${K},scaled=${SCALED}" \
    -o "$ZIP" \
    --cores "$JOBS"

# ── Step 5: verify the zip BEFORE indexing ──────────────────
echo "[5/6] verifying ${ZIP}..."
unzip -t "$ZIP" >/dev/null || { echo "ERROR: ${ZIP} failed unzip -t"; exit 1; }
sourmash sig summarize "$ZIP" >/dev/null || { echo "ERROR: sourmash cannot read ${ZIP}"; exit 1; }
echo "  zip OK"

# ── Step 6: build RocksDB ───────────────────────────────────
echo "[6/6] index -> ${DB_PATH}"
sourmash scripts index \
    "$ZIP" \
    -o "$DB_PATH" \
    --cores "$JOBS"

echo "Done. RocksDB at: ${DB_PATH}"
