#include "kmer_scrub_streaming.h"
#include "BIO_sequence.h"   /* must precede genome_compare.h: defines BIO_sequences */
#include "BIO_hash.h"
#include "genome_compare.h"
#include "kseq.h"

#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <errno.h>
#include <zlib.h>

KSEQ_INIT(gzFile, gzread)

#define GLOBAL_COLS 4

/* ── Bounded row queue + writer thread ────────────────────────────────── */

typedef struct sample_id_s {
	char           *id;
	char            type[3];
	int             refcount;
	pthread_mutex_t mtx;
} sample_id_t;

typedef struct {
	const char  *kmer;
	sample_id_t *sid;
	uint32_t     count;
} row_t;

struct streaming_writer_s {
	gzFile           gz;
	row_t           *queue;
	size_t           cap;
	size_t           head, tail, size;
	int              shutdown;
	pthread_mutex_t  mtx;
	pthread_cond_t   not_empty;
	pthread_cond_t   not_full;
	pthread_t        writer_tid;
};

static void sample_id_unref(sample_id_t *sid);

static void *writer_main(void *arg)
{
	streaming_writer *w = (streaming_writer *)arg;

	for (;;) {
		pthread_mutex_lock(&w->mtx);
		while (w->size == 0 && !w->shutdown)
			pthread_cond_wait(&w->not_empty, &w->mtx);
		if (w->size == 0 && w->shutdown) {
			pthread_mutex_unlock(&w->mtx);
			break;
		}
		row_t batch[256];
		size_t n = 0;
		while (n < (sizeof batch / sizeof batch[0]) && w->size > 0) {
			batch[n++] = w->queue[w->head];
			w->head = (w->head + 1) % w->cap;
			w->size--;
		}
		pthread_cond_broadcast(&w->not_full);
		pthread_mutex_unlock(&w->mtx);

		for (size_t i = 0; i < n; i++) {
			gzprintf(w->gz, "%s\t%s\t%s\t%u\n",
			         batch[i].kmer,
			         batch[i].sid->type,
			         batch[i].sid->id,
			         batch[i].count);
			sample_id_unref(batch[i].sid);
		}
	}
	return NULL;
}

streaming_writer *streaming_writer_open(const char *path, size_t queue_capacity)
{
	if (queue_capacity == 0) queue_capacity = 1u << 20;

	streaming_writer *w = calloc(1, sizeof(*w));
	if (!w) return NULL;

	w->gz = gzopen(path, "wb6");  /* level 6: ~3x faster than 9, ~5% bigger */
	if (!w->gz) {
		fprintf(stderr, "streaming_writer_open: gzopen %s failed: %s\n",
		        path, strerror(errno));
		free(w);
		return NULL;
	}
	gzbuffer(w->gz, 1u << 20);

	w->queue = calloc(queue_capacity, sizeof(row_t));
	if (!w->queue) {
		gzclose(w->gz);
		free(w);
		return NULL;
	}
	w->cap = queue_capacity;
	pthread_mutex_init(&w->mtx, NULL);
	pthread_cond_init(&w->not_empty, NULL);
	pthread_cond_init(&w->not_full, NULL);

	gzprintf(w->gz, "kmer\tsample_type\tsample_id\tcount\n");

	if (pthread_create(&w->writer_tid, NULL, writer_main, w) != 0) {
		fprintf(stderr, "streaming_writer_open: pthread_create failed\n");
		gzclose(w->gz);
		free(w->queue);
		free(w);
		return NULL;
	}
	return w;
}

void streaming_writer_close(streaming_writer *w)
{
	if (!w) return;

	pthread_mutex_lock(&w->mtx);
	w->shutdown = 1;
	pthread_cond_broadcast(&w->not_empty);
	pthread_mutex_unlock(&w->mtx);

	pthread_join(w->writer_tid, NULL);

	gzclose(w->gz);
	pthread_mutex_destroy(&w->mtx);
	pthread_cond_destroy(&w->not_empty);
	pthread_cond_destroy(&w->not_full);
	free(w->queue);
	free(w);
}

static void writer_push(streaming_writer *w, row_t r)
{
	pthread_mutex_lock(&w->mtx);
	while (w->size == w->cap)
		pthread_cond_wait(&w->not_full, &w->mtx);
	w->queue[w->tail] = r;
	w->tail = (w->tail + 1) % w->cap;
	w->size++;
	pthread_cond_signal(&w->not_empty);
	pthread_mutex_unlock(&w->mtx);
}

/* ── Sample ID ref-counting ───────────────────────────────────────────── */

static sample_id_t *sample_id_new(const char *id, const char *type)
{
	sample_id_t *s = malloc(sizeof(*s));
	if (!s) { perror("malloc"); exit(EXIT_FAILURE); }
	s->id = strdup(id);
	if (!s->id) { perror("strdup"); exit(EXIT_FAILURE); }
	strncpy(s->type, type, sizeof(s->type) - 1);
	s->type[sizeof(s->type) - 1] = '\0';
	s->refcount = 1;
	pthread_mutex_init(&s->mtx, NULL);
	return s;
}

static void sample_id_ref(sample_id_t *s)
{
	pthread_mutex_lock(&s->mtx);
	s->refcount++;
	pthread_mutex_unlock(&s->mtx);
}

static void sample_id_unref(sample_id_t *s)
{
	pthread_mutex_lock(&s->mtx);
	int rc = --s->refcount;
	pthread_mutex_unlock(&s->mtx);
	if (rc == 0) {
		pthread_mutex_destroy(&s->mtx);
		free(s->id);
		free(s);
	}
}

/* ── Helpers ──────────────────────────────────────────────────────────── */

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
		    strcmp(ext, "fa")    == 0 ||
		    strcmp(ext, "fna")   == 0 ||
		    strcmp(ext, "ffn")   == 0 ||
		    strcmp(ext, "fasta") == 0 ||
		    strcmp(ext, "fastq") == 0 ||
		    strcmp(ext, "fq")    == 0)
			*dot = '\0';
		else
			break;
	}
	return out;
}

/* Forward-declared: defined in genome_compare.c but not in its header */
extern int   contains_N(char *str);
extern char *orient_string(char *seed_seq, char *seedStrRevComp, int seed);

/*
	Single-pass dual-counter for one file.

	Reads the file once. For every k-mer hit in the reference hash:
	  - atomically increments counts[global_col]   (pangenome / metagenome / drug)
	  - atomically increments counts[scratch_col]  (this worker's scratch)

	Both increments are atomic because this is the same column-write pattern
	the existing GEN_calculate_kmer_count uses; other threads may be writing
	to *different* columns of the same value vector concurrently. Different
	memory addresses, no contention, but use the atomic for safety and
	consistency with existing semantics.
*/
static void calculate_kmer_count_dual(const char *file,
                                      const int seed,
                                      BIO_hash h,
                                      unsigned int global_col,
                                      unsigned int scratch_col)
{
	gzFile fp = gzopen(file, "r");
	if (fp == NULL) {
		fprintf(stderr, "could not read file %s in calculate_kmer_count_dual()\n",
		        file);
		return;  /* don't crash the whole job for one bad file */
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
				if (count != NULL) {
					__sync_fetch_and_add(&count[global_col],  1);
					__sync_fetch_and_add(&count[scratch_col], 1);
				}
			}

			seed_seq[seed] = temp_nuc;
			seed_seq++;
		}
	}

	kseq_destroy(seq);
	gzclose(fp);
	free(seedStrRevComp);
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

	int              seed;
	BIO_hash         h;
	int              num_threads;
	streaming_writer *writer;
	const char      *sample_type;
	unsigned int     global_col;
	const char      *skip_file;
	FILE            *progress;
	pthread_mutex_t  progress_mtx;
} worker_pool_t;

/*
	After a sample is fully counted, sweep the hash:
	  - emit one row per non-zero scratch entry to the writer
	  - zero the scratch column

	Different workers run this concurrently for *different* scratch columns,
	so there's no contention on the writes. Reads of h->data[i].DATA and
	the kmer key are safe because the hash topology is read-only after
	seeding.
*/
static void emit_and_zero_column(BIO_hash h,
                                 unsigned int col,
                                 sample_id_t *sid,
                                 streaming_writer *writer)
{
	for (unsigned int i = 0; i < h->M; i++) {
		unsigned int *counts = (unsigned int *)h->data[i].DATA;
		if (counts == NULL) continue;
		unsigned int c = counts[col];
		if (c == 0) continue;

		sample_id_ref(sid);
		row_t r;
		r.kmer  = h->data[i].key;
		r.sid   = sid;
		r.count = c;
		writer_push(writer, r);

		counts[col] = 0;
	}
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
			fprintf(stderr, "skipping %s (identical match)\n", path);
			free(job->filepath);
			free(job);
			continue;
		}

		if (p->progress) {
			pthread_mutex_lock(&p->progress_mtx);
			time_t ltime = time(NULL);
			fprintf(p->progress, "%s\t%s", path, asctime(localtime(&ltime)));
			fflush(p->progress);
			pthread_mutex_unlock(&p->progress_mtx);
		}

		char *id = basename_no_ext(path);
		sample_id_t *sid = sample_id_new(id, p->sample_type);
		free(id);

		/* SINGLE PASS: read file once, increment both global + scratch */
		calculate_kmer_count_dual(path, p->seed, p->h,
		                          p->global_col, scratch_col);

		/* emit + zero scratch column */
		emit_and_zero_column(p->h, scratch_col, sid, p->writer);

		sample_id_unref(sid);

		free(job->filepath);
		free(job);
	}
	return NULL;
}

/* ── Public entry point ───────────────────────────────────────────────── */

void GEN_per_sample_kmer_counts_dual(const char *list_path,
                                     const char *sample_type,
                                     unsigned int global_col,
                                     int seed,
                                     BIO_hash h,
                                     int num_threads,
                                     streaming_writer *writer,
                                     FILE *progress,
                                     const char *skip_file)
{
	FILE *fp = fopen(list_path, "r");
	if (!fp) {
		fprintf(stderr,
		        "GEN_per_sample_kmer_counts_dual: cannot open %s: %s\n",
		        list_path, strerror(errno));
		exit(EXIT_FAILURE);
	}

	int nlines = 0;
	{
		char *line = NULL;
		size_t cap = 0;
		while (getline(&line, &cap, fp) != -1) {
			char *q = line;
			while (*q == ' ' || *q == '\t') q++;
			if (*q != '\n' && *q != '\0') nlines++;
		}
		free(line);
		rewind(fp);
	}

	if (nlines == 0) {
		fclose(fp);
		return;
	}

	worker_pool_t p;
	memset(&p, 0, sizeof p);
	p.jcap        = nlines + 1;
	p.jobs        = malloc(sizeof(sample_job_t *) * p.jcap);
	p.shutdown    = 0;
	p.tid_counter = 0;
	p.seed        = seed;
	p.h           = h;
	p.num_threads = num_threads;
	p.writer      = writer;
	p.sample_type = sample_type;
	p.global_col  = global_col;
	p.skip_file   = skip_file;
	p.progress    = progress;
	pthread_mutex_init(&p.jmtx, NULL);
	pthread_cond_init(&p.jhas_work, NULL);
	pthread_mutex_init(&p.progress_mtx, NULL);

	{
		char *line = NULL;
		size_t cap = 0;
		ssize_t n;
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
	pthread_mutex_destroy(&p.jmtx);
	pthread_cond_destroy(&p.jhas_work);
	pthread_mutex_destroy(&p.progress_mtx);
}
