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
	                          by the existing GEN_all_kmer_counts* code
	  [4 .. 4+T-1]            per-thread scratch columns; column (4+tid) is
	                          owned exclusively by worker thread tid

	Each worker processes whole samples (one file) at a time:
	  1) k-mer-counts the file into its own scratch column
	  2) sweeps the hash, emits non-zero rows for that sample to a writer
	     queue, and zeros its column in the same pass
	  3) picks up the next sample

	A single dedicated writer thread owns the gzFile and drains the row
	queue, so there is no mutex on gzwrite and no contention on the output
	file descriptor.

	Output format (long-form TSV, gzipped):
	  kmer<TAB>sample_type<TAB>sample_id<TAB>count
	One header line is written by streaming_writer_open(); call this once
	before the first call to GEN_per_sample_kmer_counts_streaming.
*/

/* Opaque writer context. Workers don't access it directly — they push rows
   via streaming_emit_row which the writer thread consumes. */
typedef struct streaming_writer_s streaming_writer;

/* Open a gzipped output file and start the writer thread.
   Returns a writer handle, or NULL on error. queue_capacity controls
   backpressure on workers; pass 0 for the default (1<<20). */
streaming_writer *streaming_writer_open(const char *path, size_t queue_capacity);

/* Block until all queued rows are written, then close the file and join
   the writer thread. Frees the writer context. */
void streaming_writer_close(streaming_writer *w);

/*
	Process every file in list_path, one sample per line, in parallel.

	Each sample is assigned to a worker thread which:
	  - counts k-mers into column (4 + tid)
	  - emits non-zero rows via the writer with the given sample_type
	    ("ge", "me", or "dr")
	  - zeros its column

	If skip_file is non-NULL, samples whose path string-matches it are
	skipped entirely (used for -C to avoid double-counting the reference).

	If progress is non-NULL, each sample's path and start time are written
	to it as the worker picks it up — same format as the existing
	GEN_all_kmer_counts progress log.

	num_threads must match the T used when seeding the hash with
	(4 + T) columns. The first call also publishes T into the streaming
	module's internal state.
*/
void GEN_per_sample_kmer_counts_streaming(const char *list_path,
                                          const char *sample_type,
                                          int seed,
                                          BIO_hash h,
                                          int num_threads,
                                          streaming_writer *writer,
                                          FILE *progress,
                                          const char *skip_file);

#endif /* KMER_SCRUB_STREAMING_H */
