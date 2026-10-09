#include "kmer_scrub_streaming.h"
#include "BIO_sequence.h"   /* must precede genome_compare.h: defines BIO_sequences */
#include "BIO_hash.h"
#include "genome_compare.h"
#include "kseq.h"
#include "kmdb.h"

#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <stdint.h>
#include <inttypes.h>
#include <errno.h>
#include <zlib.h>
#include <zstd.h>

KSEQ_INIT(gzFile, gzread)

#define GLOBAL_COLS 4

/* ── zstd stream writer (declared in kmer_scrub_streaming.h) ────────── */

struct zstd_out_s {
	FILE         *fp;
	ZSTD_CStream *cs;
	void         *out_buf;
	size_t        out_cap;
};

zstd_out_t *zstd_out_open(const char *path, int level)
{
	zstd_out_t *z = calloc(1, sizeof(*z));
	if (!z) return NULL;
	z->fp = fopen(path, "wb");
	if (!z->fp) { free(z); return NULL; }
	z->cs = ZSTD_createCStream();
	if (!z->cs) { fclose(z->fp); free(z); return NULL; }
	size_t init_rc = ZSTD_initCStream(z->cs, level);
	if (ZSTD_isError(init_rc)) {
		ZSTD_freeCStream(z->cs); fclose(z->fp); free(z);
		return NULL;
	}
	z->out_cap = ZSTD_CStreamOutSize();
	z->out_buf = malloc(z->out_cap);
	if (!z->out_buf) {
		ZSTD_freeCStream(z->cs); fclose(z->fp); free(z);
		return NULL;
	}
	return z;
}

int zstd_out_write(zstd_out_t *z, const void *data, size_t len)
{
	ZSTD_inBuffer in = { data, len, 0 };
	while (in.pos < in.size) {
		ZSTD_outBuffer out = { z->out_buf, z->out_cap, 0 };
		size_t rc = ZSTD_compressStream(z->cs, &out, &in);
		if (ZSTD_isError(rc)) return -1;
		if (out.pos > 0 &&
		    fwrite(z->out_buf, 1, out.pos, z->fp) != out.pos)
			return -1;
	}
	return 0;
}

int zstd_out_close(zstd_out_t *z)
{
	if (!z) return 0;
	int err = 0;
	for (;;) {
		ZSTD_outBuffer out = { z->out_buf, z->out_cap, 0 };
		size_t rem = ZSTD_endStream(z->cs, &out);
		if (ZSTD_isError(rem)) { err = -1; break; }
		if (out.pos > 0 &&
		    fwrite(z->out_buf, 1, out.pos, z->fp) != out.pos) {
			err = -1; break;
		}
		if (rem == 0) break;
	}
	ZSTD_freeCStream(z->cs);
	if (fclose(z->fp) != 0) err = -1;
	free(z->out_buf);
	free(z);
	return err;
}

/* ── Per-bucket dynamic id list ──────────────────────────────────────── */

/* Sentinel stored in id_list_t.cap to mark a bucket as "saturated": its
   presence exceeded presence_max, so its ids have been freed and it must
   never be re-grown or written. Chosen to be a value cap can never take
   legitimately (real caps double from 4 and would need 16 GB to reach). */
#define ID_LIST_SATURATED 0xFFFFFFFFu

typedef struct {
	uint32_t *ids;
	uint32_t  size;
	uint32_t  cap;
} id_list_t;

static void id_list_append(id_list_t *l, uint32_t id)
{
	if (l->size == l->cap) {
		uint32_t newcap = l->cap == 0 ? 4 : l->cap * 2;
		uint32_t *newids = realloc(l->ids, newcap * sizeof(uint32_t));
		if (!newids) { perror("realloc id_list"); exit(EXIT_FAILURE); }
		l->ids = newids;
		l->cap = newcap;
	}
	l->ids[l->size++] = id;
}

/* ── Per-sample queue record ─────────────────────────────────────────── */

typedef struct sample_record_s {
	char         *sample_id;     /* owned heap copy */
	char          sample_type[3];
	unsigned long n_unique_kmers;
	double        coverage_pct;
	int           is_in_global;
	uint32_t     *bucket_indices;  /* owned; bucket idx of each hit kmer */
	uint32_t      n_kmers;
} sample_record_t;

/* ── Summary writer ──────────────────────────────────────────────────── */

struct summary_writer_s {
	FILE           *fp;
	unsigned long   total_ref_kmers;
	pthread_mutex_t mtx;  /* used only when writer is NULL */
};

summary_writer *summary_writer_open(const char *path,
                                    unsigned long total_reference_kmers)
{
	if (!path) return NULL;
	summary_writer *s = calloc(1, sizeof(*s));
	if (!s) return NULL;
	s->fp = fopen(path, "w");
	if (!s->fp) {
		fprintf(stderr, "summary_writer_open: fopen %s failed: %s\n",
		        path, strerror(errno));
		free(s);
		return NULL;
	}
	s->total_ref_kmers = total_reference_kmers;
	pthread_mutex_init(&s->mtx, NULL);
	fprintf(s->fp,
	        "scrub_id\tsample_type\tsample_id\tn_unique_kmers"
	        "\tcoverage_pct\tis_in_global\n");
	return s;
}

void summary_writer_close(summary_writer *s)
{
	if (!s) return;
	fclose(s->fp);
	pthread_mutex_destroy(&s->mtx);
	free(s);
}

static void summary_write_row(summary_writer *s,
                              uint32_t scrub_id,
                              const char *sample_type,
                              const char *sample_id,
                              unsigned long n_unique_kmers,
                              double coverage_pct,
                              int is_in_global,
                              int needs_lock)
{
	if (!s) return;
	if (needs_lock) pthread_mutex_lock(&s->mtx);
	fprintf(s->fp, "%u\t%s\t%s\t%lu\t%.6f\t%s\n",
	        scrub_id, sample_type, sample_id,
	        n_unique_kmers, coverage_pct,
	        is_in_global ? "True" : "False");
	fflush(s->fp);
	if (needs_lock) pthread_mutex_unlock(&s->mtx);
}

/* ── Presence writer ──────────────────────────────────────────────────── */

struct presence_writer_s {
	const char  *out_path;        /* not duped; caller must keep valid */

	sample_record_t **queue;
	size_t            cap;
	size_t            head, tail, size;
	int               shutdown;
	pthread_mutex_t   mtx;
	pthread_cond_t    not_empty;
	pthread_cond_t    not_full;
	pthread_t         writer_tid;
	int               thread_started;

	uint32_t          next_scrub_id;
	summary_writer   *summary;

	id_list_t        *id_lists;
	unsigned int      n_buckets;

	uint32_t          presence_max;   /* drop k-mers seen in > this many
	                                     samples; 0 = unlimited */

	/* Diagnostic counters */
	uint64_t  diag_writer_waits;
	uint64_t  diag_worker_waits;
	uint64_t  diag_writer_wait_ns;
	uint64_t  diag_worker_wait_ns;
	size_t    diag_max_queue_depth;
	uint64_t  diag_samples_processed;
	uint64_t  diag_kmer_appends;
	uint64_t  diag_kmers_dropped;     /* distinct k-mers cut by presence cap */
};

static inline uint64_t now_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void *presence_writer_main(void *arg);

presence_writer *presence_writer_open(const char *path,
                                      size_t queue_capacity,
                                      uint32_t presence_max)
{
	if (queue_capacity == 0) queue_capacity = 256;

	presence_writer *w = calloc(1, sizeof(*w));
	if (!w) return NULL;
	w->out_path = path;
	w->presence_max = presence_max;
	w->cap = queue_capacity;
	w->queue = calloc(queue_capacity, sizeof(sample_record_t *));
	if (!w->queue) { free(w); return NULL; }
	pthread_mutex_init(&w->mtx, NULL);
	pthread_cond_init(&w->not_empty, NULL);
	pthread_cond_init(&w->not_full, NULL);

	if (pthread_create(&w->writer_tid, NULL, presence_writer_main, w) != 0) {
		fprintf(stderr, "presence_writer_open: pthread_create failed\n");
		pthread_mutex_destroy(&w->mtx);
		pthread_cond_destroy(&w->not_empty);
		pthread_cond_destroy(&w->not_full);
		free(w->queue);
		free(w);
		return NULL;
	}
	w->thread_started = 1;
	return w;
}

/* Called from the public entry point on first use to wire up the writer
   to its companion summary file and to allocate the per-bucket side
   array. Idempotent. */
static void presence_writer_attach_summary(presence_writer *w,
                                           summary_writer *s)
{
	if (!w) return;
	pthread_mutex_lock(&w->mtx);
	if (w->summary == NULL) w->summary = s;
	pthread_mutex_unlock(&w->mtx);
}

static void presence_writer_attach_hash(presence_writer *w, BIO_hash h)
{
	if (!w) return;
	pthread_mutex_lock(&w->mtx);
	if (w->id_lists == NULL) {
		w->n_buckets = (unsigned int)h->M;
		w->id_lists  = calloc(w->n_buckets, sizeof(id_list_t));
		if (!w->id_lists) {
			fprintf(stderr,
			        "presence_writer_attach_hash: calloc(%u) failed\n",
			        w->n_buckets);
			exit(EXIT_FAILURE);
		}
	}
	pthread_mutex_unlock(&w->mtx);
}

static void writer_push_sample(presence_writer *w, sample_record_t *rec)
{
	uint64_t t0 = 0;
	int waited = 0;
	pthread_mutex_lock(&w->mtx);
	if (w->size == w->cap) { waited = 1; t0 = now_ns(); }
	while (w->size == w->cap)
		pthread_cond_wait(&w->not_full, &w->mtx);
	if (waited) {
		w->diag_worker_waits++;
		w->diag_worker_wait_ns += (now_ns() - t0);
	}
	w->queue[w->tail] = rec;
	w->tail = (w->tail + 1) % w->cap;
	w->size++;
	if (w->size > w->diag_max_queue_depth)
		w->diag_max_queue_depth = w->size;
	pthread_cond_signal(&w->not_empty);
	pthread_mutex_unlock(&w->mtx);
}

static void *presence_writer_main(void *arg)
{
	presence_writer *w = (presence_writer *)arg;

	for (;;) {
		uint64_t t0 = 0;
		int waited = 0;
		pthread_mutex_lock(&w->mtx);
		if (w->size == 0 && !w->shutdown) { waited = 1; t0 = now_ns(); }
		while (w->size == 0 && !w->shutdown)
			pthread_cond_wait(&w->not_empty, &w->mtx);
		if (waited) {
			w->diag_writer_waits++;
			w->diag_writer_wait_ns += (now_ns() - t0);
		}
		if (w->size == 0 && w->shutdown) {
			pthread_mutex_unlock(&w->mtx);
			break;
		}
		sample_record_t *rec = w->queue[w->head];
		w->head = (w->head + 1) % w->cap;
		w->size--;
		pthread_cond_broadcast(&w->not_full);
		pthread_mutex_unlock(&w->mtx);

		/* ── Process this sample serially ─────────────────────────── */
		uint32_t scrub_id = w->next_scrub_id++;

		summary_write_row(w->summary, scrub_id,
		                  rec->sample_type, rec->sample_id,
		                  rec->n_unique_kmers, rec->coverage_pct,
		                  rec->is_in_global, /*needs_lock=*/0);

		if (w->id_lists != NULL && rec->bucket_indices != NULL) {
			for (uint32_t k = 0; k < rec->n_kmers; k++) {
				uint32_t idx = rec->bucket_indices[k];
				if (idx >= w->n_buckets) continue;

				id_list_t *l = &w->id_lists[idx];

				/* Presence cap. Each sample contributes a given bucket at
				   most once, so l->size == number of distinct samples seen
				   so far. Keep k-mers with presence <= presence_max; the
				   moment a further sample would push presence past the cap,
				   drop the list entirely: free its ids and mark it
				   saturated so it is never re-grown or written. */
				if (l->cap == ID_LIST_SATURATED)
					continue;                 /* already dropped */
				if (w->presence_max != 0 && l->size >= w->presence_max) {
					free(l->ids);
					l->ids  = NULL;
					l->size = 0;
					l->cap  = ID_LIST_SATURATED;
					w->diag_kmers_dropped++;
					continue;
				}

				id_list_append(l, scrub_id);
				w->diag_kmer_appends++;
			}
		}

		w->diag_samples_processed++;

		free(rec->sample_id);
		free(rec->bucket_indices);
		free(rec);
	}
	return NULL;
}

void presence_writer_close(presence_writer *w)
{
	if (!w) return;
	if (!w->thread_started) return;

	pthread_mutex_lock(&w->mtx);
	w->shutdown = 1;
	pthread_cond_broadcast(&w->not_empty);
	pthread_mutex_unlock(&w->mtx);

	pthread_join(w->writer_tid, NULL);
	w->thread_started = 0;
}

void presence_writer_flush(presence_writer *w, BIO_hash h)
{
	if (!w || !w->out_path) return;
	if (w->thread_started) {
		fprintf(stderr,
		        "presence_writer_flush: must call presence_writer_close first\n");
		return;
	}
	if (!w->id_lists) {
		fprintf(stderr, "presence_writer_flush: no id_lists (no samples?)\n");
		return;
	}

	zstd_out_t *z = zstd_out_open(w->out_path, /*level*/ 9);
	if (!z) {
		fprintf(stderr, "presence_writer_flush: cannot open %s: %s\n",
		        w->out_path, strerror(errno));
		return;
	}

	const char *header = "#kmer\tlist_scrub_id\n";
	zstd_out_write(z, header, strlen(header));

	/* Buffer for one row. Worst case: kmer string + tab + N_max ids x 11
	   bytes (uint32) + commas + \n. We grow on demand. */
	size_t row_cap = 256 * 1024;
	char  *rowbuf  = malloc(row_cap);
	if (!rowbuf) { perror("malloc rowbuf"); exit(EXIT_FAILURE); }

	uint64_t emitted = 0;
	for (unsigned int i = 0; i < w->n_buckets; i++) {
		id_list_t *l = &w->id_lists[i];
		if (l->size == 0 || l->cap == ID_LIST_SATURATED) continue;
		const char *key = h->data[i].key;
		if (!key) continue;

		size_t need = strlen(key) + 2 + (size_t)l->size * 12 + 2;
		if (need > row_cap) {
			while (row_cap < need) row_cap *= 2;
			char *nbuf = realloc(rowbuf, row_cap);
			if (!nbuf) { perror("realloc rowbuf"); exit(EXIT_FAILURE); }
			rowbuf = nbuf;
		}

		int n = snprintf(rowbuf, row_cap, "%s\t", key);
		for (uint32_t j = 0; j < l->size; j++) {
			n += snprintf(rowbuf + n, row_cap - (size_t)n,
			              j == 0 ? "%u" : ",%u", l->ids[j]);
		}
		rowbuf[n++] = '\n';
		zstd_out_write(z, rowbuf, (size_t)n);
		emitted++;
	}

	free(rowbuf);
	zstd_out_close(z);
	fprintf(stderr,
	        "presence_writer_flush: emitted %" PRIu64 " kmer rows to %s\n",
	        emitted, w->out_path);
}

void presence_writer_print_diagnostics(presence_writer *w)
{
	if (!w) return;
	fprintf(stderr,
	        "─── presence writer diagnostics ─────────────────────────────\n"
	        "  samples processed:   %" PRIu64 "\n"
	        "  scrub_id appends:    %" PRIu64 "\n"
	        "  kmers dropped (cap):  %" PRIu64 " (presence_max = %u%s)\n"
	        "  worker queue waits:  %" PRIu64
	        " (total %.3fs blocked on full queue)\n"
	        "                       ↑ if >>0: WRITER is the bottleneck\n"
	        "  writer queue waits:  %" PRIu64
	        " (total %.3fs blocked on empty queue)\n"
	        "                       ↑ if >>0: WORKERS are the bottleneck\n"
	        "  max queue depth:     %zu / %zu\n"
	        "─────────────────────────────────────────────────────────────\n",
	        w->diag_samples_processed,
	        w->diag_kmer_appends,
	        w->diag_kmers_dropped,
	        w->presence_max,
	        w->presence_max == 0 ? ", disabled" : "",
	        w->diag_worker_waits,
	        (double)w->diag_worker_wait_ns / 1e9,
	        w->diag_writer_waits,
	        (double)w->diag_writer_wait_ns / 1e9,
	        w->diag_max_queue_depth,
	        w->cap);
}

void presence_writer_destroy(presence_writer *w)
{
	if (!w) return;
	if (w->thread_started) presence_writer_close(w);
	if (w->id_lists) {
		for (unsigned int i = 0; i < w->n_buckets; i++)
			free(w->id_lists[i].ids);
		free(w->id_lists);
	}
	pthread_mutex_destroy(&w->mtx);
	pthread_cond_destroy(&w->not_empty);
	pthread_cond_destroy(&w->not_full);
	free(w->queue);
	free(w);
}

/* ── Duplicate-sample registry ───────────────────────────────────────── */

struct seen_registry_s {
	char           **slots;
	size_t           cap;
	size_t           size;
	pthread_mutex_t  mtx;
};

static unsigned long sr_hash(const char *s)
{
	unsigned long h = 1469598103934665603UL;
	for (; *s; s++) { h ^= (unsigned char)*s; h *= 1099511628211UL; }
	return h;
}

static void sr_insert_into(char **slots, size_t cap, char *key)
{
	size_t i = sr_hash(key) % cap;
	while (slots[i] != NULL) i = (i + 1) % cap;
	slots[i] = key;
}

static void sr_grow(seen_registry *r)
{
	size_t new_cap = r->cap * 2;
	char **new_slots = calloc(new_cap, sizeof(char *));
	if (!new_slots) { perror("calloc"); exit(EXIT_FAILURE); }
	for (size_t i = 0; i < r->cap; i++)
		if (r->slots[i]) sr_insert_into(new_slots, new_cap, r->slots[i]);
	free(r->slots);
	r->slots = new_slots;
	r->cap = new_cap;
}

seen_registry *seen_registry_new(void)
{
	seen_registry *r = calloc(1, sizeof(*r));
	if (!r) return NULL;
	r->cap = 256;
	r->slots = calloc(r->cap, sizeof(char *));
	if (!r->slots) { free(r); return NULL; }
	pthread_mutex_init(&r->mtx, NULL);
	return r;
}

void seen_registry_free(seen_registry *r)
{
	if (!r) return;
	for (size_t i = 0; i < r->cap; i++) free(r->slots[i]);
	free(r->slots);
	pthread_mutex_destroy(&r->mtx);
	free(r);
}

static int seen_registry_check_and_add(seen_registry *r,
                                       const char *type,
                                       const char *id)
{
	if (!r) return 0;
	char key[512];
	int n = snprintf(key, sizeof key, "%s:%s", type, id);
	if (n < 0 || (size_t)n >= sizeof key) {
		fprintf(stderr, "seen_registry: key too long for %s:%s\n", type, id);
		return 0;
	}
	pthread_mutex_lock(&r->mtx);
	if (r->size * 2 >= r->cap) sr_grow(r);
	size_t i = sr_hash(key) % r->cap;
	while (r->slots[i] != NULL) {
		if (strcmp(r->slots[i], key) == 0) {
			pthread_mutex_unlock(&r->mtx);
			return 1;
		}
		i = (i + 1) % r->cap;
	}
	r->slots[i] = strdup(key);
	if (!r->slots[i]) { perror("strdup"); exit(EXIT_FAILURE); }
	r->size++;
	pthread_mutex_unlock(&r->mtx);
	return 0;
}

/* ── Helpers ─────────────────────────────────────────────────────────── */

static char *basename_no_ext(const char *path)
{
	const char *base = strrchr(path, '/');
	base = base ? base + 1 : path;
	char *out = strdup(base);
	if (!out) { perror("strdup"); exit(EXIT_FAILURE); }
	for (int pass = 0; pass < 2; pass++) {
		char *dot = strrchr(out, '.');
		if (!dot) break;
		const char *ext = dot + 1;
		if (strcmp(ext, "gz")    == 0 ||
		    strcmp(ext, "bz2")   == 0 ||
		    strcmp(ext, "xz")    == 0 ||
		    strcmp(ext, "zst")   == 0 ||
		    strcmp(ext, "fa")    == 0 ||
		    strcmp(ext, "fna")   == 0 ||
		    strcmp(ext, "ffn")   == 0 ||
		    strcmp(ext, "fasta") == 0 ||
		    strcmp(ext, "fastq") == 0 ||
		    strcmp(ext, "fq")    == 0 ||
		    strcmp(ext, "kmdb")  == 0)
			*dot = '\0';
		else
			break;
	}
	return out;
}

extern int   contains_N(char *str);
extern char *orient_string(char *seed_seq, char *seedStrRevComp, int seed);

/* Hot loop: scratch-only counting. */
static void calculate_kmer_count_scratch(const char *file,
                                         const int seed,
                                         BIO_hash h,
                                         unsigned int scratch_col)
{
	gzFile fp = gzopen(file, "r");
	if (fp == NULL) {
		fprintf(stderr,
		        "could not read file %s in calculate_kmer_count_scratch()\n",
		        file);
		return;
	}
	kseq_t *seq = kseq_init(fp);

	char *seedStrRevComp = (char *)malloc(sizeof(char) * (seed + 1));
	char *orientStr;
	unsigned int *count = NULL;
	char temp_nuc;
	char *seed_seq;
	int has_N;

	while (kseq_read(seq) >= 0) {
		if ((int)seq->seq.l < seed) continue;
		BIO_stringToUpper(seq->seq.s);
		seed_seq = seq->seq.s;
		has_N = contains_N(seed_seq);
		for (unsigned int i = 0; i < seq->seq.l - seed + 1; i++) {
			temp_nuc = seed_seq[seed];
			seed_seq[seed] = '\0';
			orientStr = orient_string(seed_seq, seedStrRevComp, seed);
			if (!has_N || !contains_N(orientStr)) {
				count = (unsigned int *)BIO_searchHash(h, orientStr);
				if (count != NULL)
					__sync_fetch_and_add(&count[scratch_col], 1);
			}
			seed_seq[seed] = temp_nuc;
			seed_seq++;
		}
	}
	kseq_destroy(seq);
	gzclose(fp);
	free(seedStrRevComp);
}

/* ── .kmdb samples: merge-join against the sorted reference k-mers ───────

   A .kmdb holds sorted canonical 2-bit k-mers with counts. The reference
   hash keys are converted once into a sorted (kmer, bucket) array; each
   database block is then merge-joined against it and every hit adds the
   stored count to the bucket's scratch column, exactly as the FASTQ scan
   would add one per occurrence. Blocks are independent, so one sample is
   split across several helper threads. Keys in a .kmdb are unique, so
   each bucket is touched at most once per sample. */

typedef struct {
	uint64_t kmer;
	uint32_t bucket;
} ref_kmer_t;

static int ref_kmer_cmp(const void *a, const void *b)
{
	uint64_t x = ((const ref_kmer_t *)a)->kmer, y = ((const ref_kmer_t *)b)->kmer;
	return (x > y) - (x < y);
}

/* Returns a sorted array of every hash key as a canonical 2-bit k-mer.
   Keys that are not pure ACGT of length `seed` cannot occur in a .kmdb and
   are left out (counted in *n_skipped). */
static ref_kmer_t *build_ref_kmers(BIO_hash h, int seed, size_t *n_out,
                                   size_t *n_skipped)
{
	size_t n = 0, skipped = 0;
	for (unsigned int i = 0; i < h->M; i++)
		if (h->data[i].DATA != NULL) n++;
	ref_kmer_t *r = malloc((n ? n : 1) * sizeof(*r));
	if (!r) { perror("malloc ref kmers"); exit(EXIT_FAILURE); }
	size_t j = 0;
	for (unsigned int i = 0; i < h->M; i++) {
		if (h->data[i].DATA == NULL) continue;
		const char *key = h->data[i].key;
		uint64_t x;
		if (strlen(key) != (size_t)seed || kmdb_encode(key, seed, &x) != 0) {
			skipped++;
			continue;
		}
		r[j].kmer   = kmdb_canonical(x, seed);
		r[j].bucket = i;
		j++;
	}
	qsort(r, j, sizeof(*r), ref_kmer_cmp);
	*n_out = j;
	*n_skipped = skipped;
	return r;
}

typedef struct {
	const kmdb_t     *db;
	const ref_kmer_t *ref;
	size_t            n_ref;
	BIO_hash          h;
	unsigned int      scratch_col;
	uint64_t          next_block;   /* atomic */
	int               error;        /* atomic */
} kmdb_scrub_t;

static size_t ref_lower_bound(const ref_kmer_t *r, size_t n, uint64_t x)
{
	size_t lo = 0, hi = n;
	while (lo < hi) {
		size_t m = lo + (hi - lo) / 2;
		if (r[m].kmer < x) lo = m + 1; else hi = m;
	}
	return lo;
}

static void *kmdb_scrub_helper(void *arg)
{
	kmdb_scrub_t *s = (kmdb_scrub_t *)arg;
	kmdb_block_buf buf;
	kmdb_block_buf_init(&buf);

	for (;;) {
		uint64_t b = __atomic_fetch_add(&s->next_block, 1, __ATOMIC_RELAXED);
		if (b >= s->db->n_blocks) break;
		const kmdb_block_info *bi = &s->db->blocks[b];

		size_t j = ref_lower_bound(s->ref, s->n_ref, bi->first_kmer);
		if (j == s->n_ref || s->ref[j].kmer > bi->last_kmer)
			continue;                      /* no reference k-mer in range */

		long n = kmdb_read_block(s->db, b, &buf);
		if (n < 0) { __atomic_store_n(&s->error, 1, __ATOMIC_RELAXED); break; }

		long i = 0;
		while (i < n && j < s->n_ref) {
			uint64_t x = buf.kmers[i], y = s->ref[j].kmer;
			if (x < y)      i++;
			else if (x > y) j++;
			else {
				unsigned int *counts =
					(unsigned int *)s->h->data[s->ref[j].bucket].DATA;
				__sync_fetch_and_add(&counts[s->scratch_col], buf.counts[i]);
				i++; j++;
			}
		}
	}
	kmdb_block_buf_free(&buf);
	return NULL;
}

/* Returns 0 on success, -1 if the database could not be read fully. */
static int calculate_kmer_count_scratch_kmdb(const kmdb_t *db,
                                             const ref_kmer_t *ref,
                                             size_t n_ref,
                                             BIO_hash h,
                                             unsigned int scratch_col,
                                             int n_helpers)
{
	kmdb_scrub_t s;
	memset(&s, 0, sizeof s);
	s.db = db; s.ref = ref; s.n_ref = n_ref; s.h = h; s.scratch_col = scratch_col;

	if (n_helpers < 1) n_helpers = 1;
	if ((uint64_t)n_helpers > db->n_blocks) n_helpers = (int)(db->n_blocks ? db->n_blocks : 1);
	if (n_helpers == 1) {
		kmdb_scrub_helper(&s);
	} else {
		pthread_t *th = malloc(sizeof(pthread_t) * n_helpers);
		for (int i = 0; i < n_helpers; i++)
			pthread_create(&th[i], NULL, kmdb_scrub_helper, &s);
		for (int i = 0; i < n_helpers; i++)
			pthread_join(th[i], NULL);
		free(th);
	}
	return s.error ? -1 : 0;
}

/* A path that is itself one sample rather than a list of samples. */
static int is_sample_path(const char *path)
{
	static const char *ext[] = {
		".kmdb", ".fa", ".fna", ".fasta", ".ffn", ".fas", ".fq", ".fastq",
		".fa.gz", ".fna.gz", ".fasta.gz", ".ffn.gz", ".fas.gz", ".fq.gz",
		".fastq.gz", NULL };
	size_t n = strlen(path);
	for (int i = 0; ext[i]; i++) {
		size_t m = strlen(ext[i]);
		if (n > m && strcasecmp(path + n - m, ext[i]) == 0) return 1;
	}
	return 0;
}

/* ── Worker pool ──────────────────────────────────────────────────────── */

typedef struct {
	char *filepath;
} sample_job_t;

typedef struct {
	sample_job_t   **jobs;
	int              jhead, jtail, jsize, jcap;
	int              shutdown;
	pthread_mutex_t  jmtx;
	pthread_cond_t   jhas_work;

	int              tid_counter;

	/* Internal scrub_id allocator used only when writer == NULL.
	   When writer is non-NULL the writer thread allocates ids serially. */
	uint32_t         fallback_next_id;

	int              seed;
	BIO_hash         h;
	int              num_threads;
	presence_writer *writer;
	summary_writer  *summary;
	seen_registry   *seen;
	const char      *sample_type;
	unsigned int     global_col;
	double           cov_threshold;
	unsigned long    total_ref_kmers;
	const char      *skip_file;
	FILE            *progress;
	pthread_mutex_t  progress_mtx;

	/* .kmdb samples */
	int              n_jobs;
	ref_kmer_t      *ref_kmers;     /* NULL unless a job is a .kmdb */
	size_t           n_ref_kmers;
} worker_pool_t;

/* Single-pass: collect bucket indices of hit kmers, optionally fold
   into global, zero scratch. n_unique is the count from the cheap
   pre-sweep (count_unique_in_column) and is exact for reference-seeded
   buckets, which is what we collect here. Returns malloc'd uint32 array
   (size n_unique) or NULL if n_unique == 0. */
static unsigned long count_unique_in_column(BIO_hash h,
                                            unsigned int scratch_col)
{
	unsigned long n = 0;
	for (unsigned int i = 0; i < h->M; i++) {
		unsigned int *counts = (unsigned int *)h->data[i].DATA;
		if (counts == NULL) continue;
		if (counts[0] > 0 && counts[scratch_col] > 0) n++;
	}
	return n;
}

static uint32_t *collect_zero_and_maybe_merge(BIO_hash h,
                                              unsigned int scratch_col,
                                              unsigned int global_col,
                                              int merge_into_global,
                                              unsigned long n_unique)
{
	uint32_t *out = NULL;
	if (n_unique > 0) {
		out = malloc((size_t)n_unique * sizeof(uint32_t));
		if (!out) { perror("malloc kmer indices"); exit(EXIT_FAILURE); }
	}
	uint32_t k = 0;
	for (unsigned int i = 0; i < h->M; i++) {
		unsigned int *counts = (unsigned int *)h->data[i].DATA;
		if (counts == NULL) continue;
		unsigned int c = counts[scratch_col];
		if (c == 0) continue;

		if (counts[0] > 0 && k < (uint32_t)n_unique)
			out[k++] = (uint32_t)i;

		if (merge_into_global)
			__sync_fetch_and_add(&counts[global_col], c);
		counts[scratch_col] = 0;
	}
	/* k should equal n_unique. If hash mutated mid-flight (it can't here)
	   we'd notice. Defensive: pad with 0s if short, but realistically k==n_unique. */
	return out;
}

static void *worker_main(void *arg)
{
	worker_pool_t *p = (worker_pool_t *)arg;

	int tid = __sync_fetch_and_add(&p->tid_counter, 1);
	unsigned int scratch_col = (unsigned int)(GLOBAL_COLS + tid);

	for (;;) {
		pthread_mutex_lock(&p->jmtx);
		while (p->jsize == 0 && !p->shutdown)
			pthread_cond_wait(&p->jhas_work, &p->jmtx);
		if (p->jsize == 0 && p->shutdown) {
			pthread_mutex_unlock(&p->jmtx);
			break;
		}
		sample_job_t *job = p->jobs[p->jhead];
		p->jhead = (p->jhead + 1) % p->jcap;
		p->jsize--;
		pthread_mutex_unlock(&p->jmtx);

		const char *path = job->filepath;

		if (p->skip_file != NULL && strcmp(path, p->skip_file) == 0) {
			fprintf(stderr, "skipping %s (identical match to reference)\n",
			        path);
			free(job->filepath); free(job);
			continue;
		}

		kmdb_t *db = NULL;
		char *id = NULL;
		if (kmdb_is_kmdb_path(path)) {
			/* fatal rather than skipped: a silently missing background
			   sample would leave k-mers looking more specific than they are */
			db = kmdb_open(path);
			if (!db) {
				fprintf(stderr, "error: %s is not a readable .kmdb\n", path);
				exit(EXIT_FAILURE);
			}
			if ((int)db->k != p->seed) {
				fprintf(stderr, "error: %s was built with k=%u, the scrub "
				        "uses k=%d\n", path, db->k, p->seed);
				exit(EXIT_FAILURE);
			}
			id = kmdb_meta_get(db, "sample_id");
		}
		if (!id) id = basename_no_ext(path);
		if (seen_registry_check_and_add(p->seen, p->sample_type, id)) {
			fprintf(stderr,
			        "skipping %s: sample_id '%s' (type=%s) already processed\n",
			        path, id, p->sample_type);
			kmdb_close(db);
			free(id); free(job->filepath); free(job);
			continue;
		}

		if (p->progress) {
			pthread_mutex_lock(&p->progress_mtx);
			time_t ltime = time(NULL);
			fprintf(p->progress, "%s\t%s", path, asctime(localtime(&ltime)));
			fflush(p->progress);
			pthread_mutex_unlock(&p->progress_mtx);
		}

		if (db) {
			/* split this sample's blocks over our share of the threads */
			int busy = p->n_jobs < p->num_threads ? p->n_jobs : p->num_threads;
			int helpers = busy > 0 ? p->num_threads / busy : 1;
			if (calculate_kmer_count_scratch_kmdb(db, p->ref_kmers,
			                                      p->n_ref_kmers, p->h,
			                                      scratch_col, helpers) != 0) {
				fprintf(stderr, "error: %s could not be read completely; "
				        "stopping rather than writing partial counts\n", path);
				exit(EXIT_FAILURE);
			}
			kmdb_close(db);
		} else {
			calculate_kmer_count_scratch(path, p->seed, p->h, scratch_col);
		}

		unsigned long n_unique = count_unique_in_column(p->h, scratch_col);
		double coverage = p->total_ref_kmers > 0
		    ? (double)n_unique / (double)p->total_ref_kmers
		    : 0.0;
		int is_in_global = (coverage <= p->cov_threshold);
		if (!is_in_global) {
			fprintf(stderr,
			        "excluding %s from global %s column: coverage=%.6f > T=%.6f\n",
			        id, p->sample_type, coverage, p->cov_threshold);
		}

		/* A sample excluded from the global columns is excluded from the
		   presence index too, so the two artifacts agree on what counts.
		   Passing 0 here leaves idxs NULL: the sweep still zeroes the
		   scratch column, but no bucket indices are collected, nothing is
		   queued to the writer, and the sample never occupies a slot in
		   any k-mer's presence list. That last part matters — an excluded
		   sample that hits everything otherwise burns one of the -P slots
		   on every k-mer and saturates lists that hold real signal. */
		unsigned long n_collect = is_in_global ? n_unique : 0;
		uint32_t *idxs = collect_zero_and_maybe_merge(
		    p->h, scratch_col, p->global_col, is_in_global, n_collect);

		if (p->writer) {
			/* Hand off to writer; writer assigns scrub_id and writes summary. */
			sample_record_t *rec = malloc(sizeof(*rec));
			if (!rec) { perror("malloc rec"); exit(EXIT_FAILURE); }
			rec->sample_id = id;  /* ownership transferred */
			strncpy(rec->sample_type, p->sample_type,
			        sizeof(rec->sample_type) - 1);
			rec->sample_type[sizeof(rec->sample_type) - 1] = '\0';
			rec->n_unique_kmers = n_unique;
			rec->coverage_pct   = coverage;
			rec->is_in_global   = is_in_global;
			rec->bucket_indices = idxs;       /* NULL when excluded */
			rec->n_kmers        = (uint32_t)n_collect;
			writer_push_sample(p->writer, rec);
			/* DO NOT free id or idxs — owned by writer now. */
		} else {
			/* No writer: emit summary inline with a fallback scrub_id.
			   The id will not be referenced from any presence file (which
			   is also absent in this mode), but we keep the column for
			   schema consistency. */
			uint32_t scrub_id = __sync_fetch_and_add(&p->fallback_next_id, 1);
			summary_write_row(p->summary, scrub_id,
			                  p->sample_type, id,
			                  n_unique, coverage, is_in_global,
			                  /*needs_lock=*/1);
			free(idxs);
			free(id);
		}

		free(job->filepath);
		free(job);
	}
	return NULL;
}

/* ── Public entry point ──────────────────────────────────────────────── */

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
                                     const char *skip_file)
{
	if (writer) {
		presence_writer_attach_summary(writer, summary);
		presence_writer_attach_hash(writer, h);
	}

	/* A single sample file (.kmdb, FASTQ, FASTA) can be given directly
	   instead of a list. */
	FILE *fp;
	char *single = NULL;
	if (is_sample_path(list_path)) {
		single = strdup(list_path);
		fp = single ? fmemopen(single, strlen(single), "r") : NULL;
	} else {
		fp = fopen(list_path, "r");
	}
	if (!fp) {
		fprintf(stderr,
		        "GEN_per_sample_kmer_counts_dual: cannot open %s: %s\n",
		        list_path, strerror(errno));
		exit(EXIT_FAILURE);
	}

	int nlines = 0;
	int any_kmdb = 0;
	{
		char *line = NULL; size_t cap = 0;
		while (getline(&line, &cap, fp) != -1) {
			char *q = line;
			while (*q == ' ' || *q == '\t') q++;
			if (*q != '\n' && *q != '\0') nlines++;
			char *e = q + strlen(q);
			while (e > q && (e[-1] == '\n' || e[-1] == '\r' ||
			                 e[-1] == ' '  || e[-1] == '\t')) *--e = '\0';
			if (kmdb_is_kmdb_path(q)) any_kmdb = 1;
		}
		free(line);
		rewind(fp);
	}
	if (nlines == 0) { fclose(fp); free(single); return; }

	worker_pool_t p;
	memset(&p, 0, sizeof p);
	p.jcap            = nlines + 1;
	p.jobs            = malloc(sizeof(sample_job_t *) * p.jcap);
	p.seed            = seed;
	p.h               = h;
	p.num_threads     = num_threads;
	p.writer          = writer;
	p.summary         = summary;
	p.seen            = seen;
	p.sample_type     = sample_type;
	p.global_col      = global_col;
	p.cov_threshold   = cov_threshold;
	p.total_ref_kmers = total_reference_kmers;
	p.skip_file       = skip_file;
	p.progress        = progress;
	p.n_jobs          = nlines;
	if (any_kmdb) {
		size_t skipped = 0;
		p.ref_kmers = build_ref_kmers(h, seed, &p.n_ref_kmers, &skipped);
		fprintf(stderr, "%s: %zu reference k-mers indexed for .kmdb samples",
		        sample_type, p.n_ref_kmers);
		if (skipped)
			fprintf(stderr, " (%zu with non-ACGT bases can never match a .kmdb)",
			        skipped);
		fprintf(stderr, "\n");
	}
	pthread_mutex_init(&p.jmtx, NULL);
	pthread_cond_init(&p.jhas_work, NULL);
	pthread_mutex_init(&p.progress_mtx, NULL);

	{
		char *line = NULL; size_t cap = 0; ssize_t n;
		while ((n = getline(&line, &cap, fp)) != -1) {
			while (n > 0 && (line[n-1] == '\n' || line[n-1] == '\r' ||
			                 line[n-1] == ' '  || line[n-1] == '\t'))
				line[--n] = '\0';
			if (n == 0) continue;
			sample_job_t *job = malloc(sizeof(*job));
			job->filepath = strdup(line);
			p.jobs[p.jtail] = job;
			p.jtail = (p.jtail + 1) % p.jcap;
			p.jsize++;
		}
		free(line);
	}
	fclose(fp);
	free(single);

	pthread_t *threads = malloc(sizeof(pthread_t) * num_threads);
	for (int i = 0; i < num_threads; i++)
		pthread_create(&threads[i], NULL, worker_main, &p);

	pthread_mutex_lock(&p.jmtx);
	p.shutdown = 1;
	pthread_cond_broadcast(&p.jhas_work);
	pthread_mutex_unlock(&p.jmtx);

	for (int i = 0; i < num_threads; i++)
		pthread_join(threads[i], NULL);

	free(threads);
	free(p.jobs);
	free(p.ref_kmers);
	pthread_mutex_destroy(&p.jmtx);
	pthread_cond_destroy(&p.jhas_work);
	pthread_mutex_destroy(&p.progress_mtx);
}