#ifndef METAGENOME_DB_H
#define METAGENOME_DB_H

#include <stdint.h>
#include <stdio.h>
#include <stddef.h>

/* ════════════════════════════════════════════════════════════
 * metagenome_db — external-sort + streaming-merge kmer index
 *
 * The old design held a global uint64->id_list hash table in RAM and
 * OOM'd past ~50 metagenomes. This rewrite never holds the whole index
 * in memory:
 *
 *   Phase 1 (parallel): each worker reads ONE sample, extracts canonical
 *     kmers, sorts + dedups them, and writes a sorted binary temp file.
 *     RAM is bounded by one sample's records, not the whole corpus.
 *
 *   Phase 2 (serial): a k-way min-heap merge streams the per-sample temp
 *     files in global (kmer, sample_id) order, accumulating one ID list
 *     per kmer and flushing it as a TSV row the moment the kmer changes.
 *     RAM is bounded by the heap + one kmer's ID list.
 * ════════════════════════════════════════════════════════════ */

/* ── On-disk + in-memory record ───────────────────────────────
 * Temp files are a flat array of these, 12 bytes each, no padding:
 *   [uint64_t kmer][uint32_t sample_id]
 * The packing is the contract the merge relies on for fread/fwrite,
 * so the struct MUST be exactly 12 bytes.
 *
 * Because it is packed, never take the address of a member
 * (&rec.kmer) — read/write whole members only. Targets here are
 * x86-64, where the resulting unaligned scalar loads are fine.
 * ──────────────────────────────────────────────────────────── */
#pragma pack(push, 1)
typedef struct {
    uint64_t kmer;        /* 2-bit encoded canonical kmer */
    uint32_t sample_id;   /* 1-based sample id            */
} kmer_record;
#pragma pack(pop)

_Static_assert(sizeof(kmer_record) == 12,
               "kmer_record must be exactly 12 bytes on disk");

/* ── Phase 1 ──────────────────────────────────────────────────
 * Read one FASTA/FASTQ file (gzip or plain), extract every canonical
 * k-mer, sort by (kmer, sample_id), dedup, and write the sorted binary
 * temp file <tmpdir>/<sample_id>.tmp.bin.
 *
 * Thread-safe: touches only its own temp file plus stack/heap locals.
 * The per-file stderr stats line is serialized internally.
 *
 *   returns  0  success — out_path holds the temp-file path
 *   returns  1  file produced zero usable kmers (no temp file; out_path "")
 *   returns -1  hard error (open/alloc/write failure)
 * ──────────────────────────────────────────────────────────── */
int mdb_sort_sample(const char *filepath, uint32_t sample_id,
                    int kmer_len, const char *tmpdir,
                    char *out_path, size_t out_path_sz);

/* ── Phase 2 ──────────────────────────────────────────────────
 * k-way merge of all per-sample temp files into one or more zstd TSV
 * outputs. Each output is described by an mdb_output: the merged k-mer
 * stream is fanned out to every output, and each row is kept only if its
 * id-list count falls in the half-open range [min_count, max_count):
 *
 *   min_count 0, max_count 0  : keep all        — the "full" index
 *   min_count 0, max_count x  : keep count < x  — the "rare/prev" index
 *   min_count x, max_count 0  : keep count >= x — the "common" list
 *
 * kmer_only selects the row schema:
 *   0 : "#kmer\tlist_scrub_id"  ->  ACGT...\t1,5,23,...
 *   1 : "#kmer"                 ->  ACGT...
 *
 * The common list is emitted as bare k-mers by design: at 60k samples the
 * high-prevalence k-mers each carry up to tens of thousands of ids, so a
 * full id-list would balloon the file — a later merge across scrubs only
 * needs to recognize these k-mers as common so they aren't mistaken for
 * rare.
 *
 * The merge uses a min-heap over file cursors. When the number of inputs
 * exceeds `fanout`, it merges in multiple passes through intermediate
 * binary temp files in `tmpdir`, so it never needs more than `fanout`
 * file descriptors open at once — this is what lets it scale to 60k
 * samples past the open-fd ulimit. Filtering happens only on the final
 * pass, so intermediate passes keep every record.
 *
 * Consumes its inputs: every temp file it reads (and every intermediate
 * it creates) is unlinked once fully consumed. The caller still owns and
 * must free the `temp_paths` strings themselves.
 *
 *   returns 0 on success, -1 on error.
 * ──────────────────────────────────────────────────────────── */
typedef struct {
    const char *path;        /* output .tsv.zst; created by the merge      */
    uint32_t    min_count;   /* keep iff id-count >= min_count              */
    uint32_t    max_count;   /* keep iff id-count <  max_count; 0 = no cap  */
    int         kmer_only;   /* 1: emit bare "kmer"; 0: emit kmer + id list */
} mdb_output;

int mdb_merge(char *const *temp_paths, size_t n_files,
              int kmer_len, const char *tmpdir, int fanout,
              const mdb_output *outputs, size_t n_outputs);

/* Append one "sample_id<TAB>filepath" line to an open manifest stream. */
void mdb_write_manifest_line(FILE *manifest_fp, uint32_t sample_id,
                             const char *filepath);

#endif /* METAGENOME_DB_H */
