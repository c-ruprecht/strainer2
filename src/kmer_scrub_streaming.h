#ifndef KMER_SCRUB_STREAMING_H
#define KMER_SCRUB_STREAMING_H

#include <stdio.h>
#include <stdint.h>
#include "BIO_hash.h"

/*
	kmer_scrub_streaming (inverted-index variant)

	Per-sample k-mer counting that produces an INVERTED index instead of
	a long-format per-sample TSV. The big output file has one row per
	k-mer, with the list of strain (scrub) IDs that hit that k-mer.

	Architecture
	------------
	Workers do the k-mer counting hot loop and scratch sweep in parallel.
	After finishing a sample, each worker hands ONE queue record to the
	single writer thread containing the sample's metadata and a list of
	hash-bucket indices (the kmers it hit). The writer (serial) assigns
	the scrub_id, writes the summary row, and appends the scrub_id to
	each bucket's per-bucket id-list.

	The inverted presence file is materialized in one serial sweep at
	end-of-run via presence_writer_flush() with zstd compression.

	Memory model
	------------
	Hash value vector: (4 + num_threads) columns.
	  [0..3]        global counts (ref, pan, meta, drug)
	  [4 .. 4+T-1]  per-thread scratch columns (column 4+tid owned by tid)

	Side-array (id_lists) parallel to the hash carries a small dynamic
	u32 array per bucket, allocated lazily on first scrub_id append.

	Outputs
	-------
	1) stdout — global k-mer counts (driver-printed; this lib is silent
	   on stdout).

	2) presence file (optional, via presence_writer_open()):
	   zstd-compressed TSV with one row per k-mer that had any sample hit:
	     #kmer<TAB>list_scrub_id
	   list_scrub_id is comma-separated u32.

	3) summary file (optional, via summary_writer_open()):
	     scrub_id<TAB>sample_type<TAB>sample_id<TAB>n_unique_kmers
	             <TAB>coverage_pct<TAB>is_in_global
	   scrub_id is sequential u32 starting at 0, assigned in
	   sample-completion order. The same id appears in the presence file.

	Coverage threshold
	------------------
	If coverage_pct > cov_threshold, the sample's counts are NOT folded
	into the global column (is_in_global=False). The sample STILL gets a
	scrub_id and its kmers are still added to the presence index. Default
	1.0 disables the gate.
*/

/* Opaque contexts. */
typedef struct presence_writer_s presence_writer;
typedef struct summary_writer_s  summary_writer;
typedef struct seen_registry_s   seen_registry;

/* ── Presence writer ──────────────────────────────────────────────────
   Spawns a single writer thread that consumes per-sample queue records
   from the public entry-point's workers. Handles scrub_id assignment,
   summary writes (if attached), and per-bucket scrub_id appends.
   Call sequence:
     w = presence_writer_open(path, queue_capacity);
     ... GEN_per_sample_kmer_counts_dual(...) one or more times ...
     presence_writer_close(w);            (joins writer thread, drains queue)
     presence_writer_flush(w, h);         (writes the zstd output)
     presence_writer_print_diagnostics(w);
     presence_writer_destroy(w);          (frees memory)
   ──────────────────────────────────────────────────────────────────── */
presence_writer *presence_writer_open(const char *path,
                                      size_t queue_capacity);
void             presence_writer_close(presence_writer *w);
void             presence_writer_flush(presence_writer *w, BIO_hash h);
void             presence_writer_print_diagnostics(presence_writer *w);
void             presence_writer_destroy(presence_writer *w);

/* ── Summary writer (plain TSV) ─────────────────────────────────────── */
summary_writer *summary_writer_open(const char *path,
                                    unsigned long total_reference_kmers);
void            summary_writer_close(summary_writer *s);

/* ── Cross-call duplicate-detection registry ────────────────────────── */
seen_registry *seen_registry_new(void);
void           seen_registry_free(seen_registry *r);

/* ── Public entry point ─────────────────────────────────────────────────
   `writer` may be NULL — kmer hits are then discarded; only globals and
   (if `summary` is non-NULL) the summary row are produced. When NULL,
   summary rows still get a scrub_id but it's allocated by an internal
   atomic counter rather than the writer.
*/
void GEN_per_sample_kmer_counts_dual(const char *list_path,
                                     const char *sample_type,
                                     unsigned int global_col,
                                     int seed,
                                     BIO_hash h,
                                     int num_threads,
                                     presence_writer *writer,
                                     summary_writer  *summary,
                                     seen_registry   *seen,
                                     double cov_threshold,
                                     unsigned long total_reference_kmers,
                                     FILE *progress,
                                     const char *skip_file);

#endif /* KMER_SCRUB_STREAMING_H */
