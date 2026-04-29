#ifndef KMER_SCRUB_STREAMING_H
#define KMER_SCRUB_STREAMING_H

#include <zlib.h>
#include <stdio.h>
#include "BIO_hash.h"

/*
	kmer_scrub_streaming

	Per-sample k-mer counting that streams output to a single gzipped TSV
	file instead of accumulating per-sample columns in the hash.

	Memory model
	------------
	The hash value vector has (4 + num_threads) columns:
	  [0..3]                  global counts (ref, pan, meta, drug) — owned
	                          by the existing GEN_all_kmer_counts* code in
	                          the dual-pass version, OR by the dual-counter
	                          in the single-pass version
	  [4 .. 4+T-1]            per-thread scratch columns; column (4+tid) is
	                          owned exclusively by worker thread tid

	Output format (long-form TSV, gzipped):
	  kmer<TAB>sample_type<TAB>sample_id<TAB>count
*/

/* Opaque writer context. */
typedef struct streaming_writer_s streaming_writer;

/* Open a gzipped output file and start the writer thread.
   queue_capacity controls backpressure on workers; pass 0 for the default. */
streaming_writer *streaming_writer_open(const char *path, size_t queue_capacity);

/* Block until all queued rows are written, then close. */
void streaming_writer_close(streaming_writer *w);

/*
	Single-pass per-sample counting.

	For every file in list_path, a worker thread:
	  - opens the file ONCE
	  - for each k-mer hit, atomically increments BOTH
	      counts[global_col]  (1=pan / 2=meta / 3=drug)
	      counts[4 + worker_tid]   (the worker's scratch column)
	  - sweeps the hash, emits non-zero rows for its scratch column to
	    the writer, and zeros the column

	If skip_file is non-NULL, samples whose path string-matches it are
	skipped entirely (no global increment, no rows). Used for -C to
	avoid double-counting the reference.

	num_threads must equal the T used when allocating the hash with
	(4 + T) columns.
*/
void GEN_per_sample_kmer_counts_dual(const char *list_path,
                                     const char *sample_type,
                                     unsigned int global_col,
                                     int seed,
                                     BIO_hash h,
                                     int num_threads,
                                     streaming_writer *writer,
                                     FILE *progress,
                                     const char *skip_file);

#endif /* KMER_SCRUB_STREAMING_H */
