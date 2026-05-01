#ifndef KMER_SCRUB_STREAMING_H
#define KMER_SCRUB_STREAMING_H

#include <stdio.h>
#include "BIO_hash.h"

/*
	kmer_scrub_streaming

	Per-sample k-mer counting that streams output to a zstd-compressed TSV
	instead of accumulating per-sample columns in the hash.

	Memory model
	------------
	The hash value vector has (4 + num_threads) columns:
	  [0..3]        global counts (ref, pan, meta, drug)
	  [4 .. 4+T-1]  per-thread scratch columns; column (4+tid) is owned
	                exclusively by worker thread tid

	Outputs
	-------
	Big per-sample file (zstd-compressed long-format TSV):
	  kmer<TAB>sample_type<TAB>sample_id<TAB>count

	Optional summary file (plain TSV, one row per sample):
	  sample_type<TAB>sample_id<TAB>n_unique_kmers<TAB>coverage_pct
	          <TAB>is_in_global<TAB>is_in_individual
	  coverage_pct = n_unique_kmers / total_reference_kmers
	  is_in_global     = "True" if the sample's counts were folded into
	                     the global column (global_col).
	  is_in_individual = "True" if per-sample rows were written to the
	                     long-format file. Always "False" if writer==NULL.

	Coverage band gating (cov_low, cov_high)
	----------------------------------------
	Two independent gates, both based on the sample's coverage_pct:

	  coverage_pct  <  cov_low  -> in_global=True,  in_individual=False
	    (sample contributes to the background/global column but is too
	     sparse to be worth carrying as per-sample rows)

	  cov_low <= coverage_pct <= cov_high
	                            -> in_global=True,  in_individual=True
	    (the "good band")

	  coverage_pct  >  cov_high -> in_global=False, in_individual=False
	    (sample looks too much like the reference itself — likely a
	     near-duplicate that would skew the global background — drop
	     entirely)

	  Defaults that disable both gates: cov_low=0.0, cov_high=1.0.
	  (Coverage is bounded by [0, 1] by definition.)

	  total_reference_kmers is the denominator used to compute
	  coverage_pct; it must match the value passed to summary_writer_open.
*/

/* Opaque contexts. */
typedef struct streaming_writer_s streaming_writer;
typedef struct summary_writer_s   summary_writer;
typedef struct seen_registry_s    seen_registry;

/* ── Big per-sample writer (zstd) ───────────────────────────────────── */
streaming_writer *streaming_writer_open(const char *path, size_t queue_capacity);
void              streaming_writer_close(streaming_writer *w);

/* ── Summary writer (plain TSV) ─────────────────────────────────────── */
summary_writer *summary_writer_open(const char *path,
                                    unsigned long total_reference_kmers);
void            summary_writer_close(summary_writer *s);

/* ── Cross-call duplicate-detection registry ────────────────────────── */
seen_registry *seen_registry_new(void);
void           seen_registry_free(seen_registry *r);

/* ── Public entry point ─────────────────────────────────────────────────
   `writer` may be NULL — per-sample long-format rows are then discarded
   regardless of cov_low (in_individual is always False in that case).
   `summary` may also be NULL (no summary file written).
*/
void GEN_per_sample_kmer_counts_dual(const char *list_path,
                                     const char *sample_type,
                                     unsigned int global_col,
                                     int seed,
                                     BIO_hash h,
                                     int num_threads,
                                     streaming_writer *writer,
                                     summary_writer  *summary,
                                     seen_registry   *seen,
                                     double cov_low,
                                     double cov_high,
                                     unsigned long total_reference_kmers,
                                     FILE *progress,
                                     const char *skip_file);

#endif /* KMER_SCRUB_STREAMING_H */
