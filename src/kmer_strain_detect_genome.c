/*
 * kmer_strain_detect.c
 *
 * Takes a kmer TSV (with a '#kmer' column) and a targets metagenome list,
 * counts how many times each kmer appears in each metagenome, and writes
 * a gzipped TSV: rows = kmers, columns = metagenome basenames.
 *
 * Each metagenome is processed in its own thread. Because each thread only
 * writes to its own column index in the per-kmer count array, no mutex is
 * needed for count updates. The hash is read-only during scanning.
 *
 * Usage:
 *   kmer_strain_detect -k <kmer.tsv[.gz]> -B <targets.txt> -o <out.kmer_hits.tsv.gz> [-j threads]
 */

#include "zlib.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <time.h>
#include "kseq.h"
#include "BIO_sequence.h"
#include "BIO_hash.h"
#include "genome_compare.h"

#define NOT_PAIRED_END         0
#define MAX_LINE           65536
#define MAX_KMER_LEN         512

KSEQ_INIT(gzFile, gzread)

/* ── data structures ───────────────────────────────────────────────── */

typedef struct {
    char *basename;
    char *file1;
    char *file2;   /* NULL for SE / PEI */
    int   type;
} Metagenome;

typedef struct {
    int        mg_idx;
    Metagenome *mg;
    BIO_hash   h;
    int        seed;
    uint64_t  *total_kmer_counts;
} ksd_job;

typedef struct {
    ksd_job       **queue;
    int            head, tail, size, capacity;
    int            active, shutdown;
    pthread_mutex_t mutex;
    pthread_cond_t  has_work;
    pthread_cond_t  all_done;
} ksd_pool;

/* ── forward declarations ──────────────────────────────────────────── */

static char     *strip_basename(const char *path);
static int       parse_targets(const char *file, Metagenome **out);
static int       parse_kmer_tsv(const char *file, BIO_hash h, int n_mg,
                                 int *seed_out, char ***kmer_list_out);
static void      scan_file(const char *filepath, BIO_hash h, int seed, int mg_idx, uint64_t *total_kmer_counts);
static void     *ksd_worker(void *arg);
static ksd_pool *ksd_pool_create(int nthreads, int capacity);
static void      ksd_pool_submit(ksd_pool *pool, ksd_job *job);
static void      ksd_pool_wait_and_destroy(ksd_pool *pool);
static void      usage(void);

/* ── file type helpers ─────────────────────────────────────────────── */

static char *strip_basename(const char *path) {
    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;
    char *b = strdup(base);
    const char *exts[] = {
        ".fastq.gz", ".fasta.gz", ".fa.gz", ".fq.gz", ".fna.gz",
        ".fastq",    ".fasta",    ".fa",    ".fq",    ".fna",
        NULL
    };
    for (int i = 0; exts[i]; i++) {
        size_t elen = strlen(exts[i]);
        size_t blen = strlen(b);
        if (blen > elen && strcmp(b + blen - elen, exts[i]) == 0) {
            b[blen - elen] = '\0';
            break;
        }
    }
    return b;
}

/* ── targets parser ────────────────────────────────────────────────── */

static int valid_genome_ext(const char *path) {
    const char *exts[] = { ".fna", ".fna.gz", ".fasta", ".fasta.gz", NULL };
    size_t plen = strlen(path);
    for (int i = 0; exts[i]; i++) {
        size_t elen = strlen(exts[i]);
        if (plen >= elen && strcmp(path + plen - elen, exts[i]) == 0) return 1;
    }
    return 0;
}

static int parse_targets(const char *file, Metagenome **out) {
    FILE *fp = fopen(file, "r");
    if (!fp) {
        fprintf(stderr, "cannot open targets file %s: %s\n", file, strerror(errno));
        exit(1);
    }

    int cap = 64, n = 0;
    Metagenome *mgs = malloc(cap * sizeof(Metagenome));

    char *line = NULL;
    size_t len = 0;
    while (getline(&line, &len, fp) != -1) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        char *nl = strchr(line, '\n'); if (nl) *nl = '\0';
        char *cr = strchr(line, '\r'); if (cr) *cr = '\0';
        if (line[0] == '\0') continue;
        if (!valid_genome_ext(line)) {
            fprintf(stderr, "skipping unsupported file type: %s\n", line);
            continue;
        }

        if (n == cap) { cap *= 2; mgs = realloc(mgs, cap * sizeof(Metagenome)); }
        mgs[n].type     = NOT_PAIRED_END;
        mgs[n].file1    = strdup(line);
        mgs[n].file2    = NULL;
        mgs[n].basename = strip_basename(line);
        n++;
    }
    free(line);
    fclose(fp);
    *out = mgs;
    return n;
}

/* ── kmer TSV parser ───────────────────────────────────────────────── */

static int parse_kmer_tsv(const char *file, BIO_hash h, int n_mg,
                           int *seed_out, char ***kmer_list_out) {
    gzFile fp = gzopen(file, "r");
    if (!fp) {
        fprintf(stderr, "cannot open kmer file %s: %s\n", file, strerror(errno));
        exit(1);
    }

    char line[MAX_LINE];
    int kmer_col = -1;
    int seed = -1;

    /* find the '#kmer' column index from the header */
    if (!gzgets(fp, line, MAX_LINE)) {
        fprintf(stderr, "kmer file is empty: %s\n", file);
        exit(1);
    }
    {
        char hdr[MAX_LINE];
        strncpy(hdr, line, MAX_LINE);
        hdr[MAX_LINE - 1] = '\0';
        char *tok = strtok(hdr, "\t\n\r");
        int col = 0;
        while (tok) {
            if (strcmp(tok, "#kmer") == 0) { kmer_col = col; break; }
            tok = strtok(NULL, "\t\n\r");
            col++;
        }
    }
    if (kmer_col < 0) {
        fprintf(stderr, "no '#kmer' column found in %s\n", file);
        exit(1);
    }

    int cap = 4096, n = 0;
    char **kmer_list = malloc(cap * sizeof(char *));
    char kmer_rc[MAX_KMER_LEN];

    while (gzgets(fp, line, MAX_LINE)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        char *nl = strchr(line, '\n'); if (nl) *nl = '\0';
        char *cr = strchr(line, '\r'); if (cr) *cr = '\0';

        /* advance to the kmer column */
        char *tok = strtok(line, "\t");
        for (int c = 0; c < kmer_col && tok; c++)
            tok = strtok(NULL, "\t");
        if (!tok || tok[0] == '\0') continue;

        if (seed < 0) {
            seed = (int)strlen(tok);
            *seed_out = seed;
        }

        /* compute canonical (oriented) form */
        strncpy(kmer_rc, tok, MAX_KMER_LEN - 1);
        kmer_rc[MAX_KMER_LEN - 1] = '\0';
        BIO_reverseComplement(kmer_rc);
        char *oriented = orient_string(tok, kmer_rc, seed);

        /* skip duplicates (same canonical kmer) */
        if (BIO_searchHash(h, oriented) != NULL) continue;

        unsigned int *counts = calloc(n_mg, sizeof(unsigned int));
        BIO_addHashData(h, oriented, counts);

        if (n == cap) { cap *= 2; kmer_list = realloc(kmer_list, cap * sizeof(char *)); }
        kmer_list[n++] = strdup(tok);   /* store original for output */
    }

    gzclose(fp);
    *kmer_list_out = kmer_list;
    return n;
}

/* ── read scanner ──────────────────────────────────────────────────── */

static void scan_file(const char *filepath, BIO_hash h, int seed, int mg_idx, uint64_t *total_kmer_counts) {
    gzFile fp = gzopen(filepath, "r");
    if (!fp) {
        fprintf(stderr, "cannot open %s: %s\n", filepath, strerror(errno));
        return;
    }

    kseq_t *seq = kseq_init(fp);
    size_t rc_cap = 0;
    char *seedStrRevComp = NULL;
    unsigned int i;

    while (kseq_read(seq) >= 0) {
        if ((int)seq->seq.l < seed) continue;
        BIO_stringToUpper(seq->seq.s);

        if (seq->seq.l + 1 > rc_cap) {
            rc_cap = seq->seq.l + 1;
            seedStrRevComp = realloc(seedStrRevComp, rc_cap);
        }
        strcpy(seedStrRevComp, seq->seq.s);
        BIO_reverseComplement(seedStrRevComp);

        char *seed_seq    = seq->seq.s;
        char *seed_seq_rc = &seedStrRevComp[seq->seq.l - seed];
        int   has_N       = contains_N(seed_seq);

        for (i = 0; i < seq->seq.l - (unsigned int)seed + 1; i++) {
            total_kmer_counts[mg_idx]++;
            char temp_nuc     = seed_seq[seed];
            seed_seq[seed]    = '\0';
            seed_seq_rc[seed] = '\0';

            char *orientStr = (strcmp(seed_seq, seed_seq_rc) > 0)
                              ? seed_seq : seed_seq_rc;

            if (!has_N || !contains_N(orientStr)) {
                unsigned int *count = (unsigned int *)BIO_searchHash(h, orientStr);
                if (count) count[mg_idx]++;   /* only this thread writes column mg_idx */
            }

            seed_seq[seed] = temp_nuc;
            seed_seq++;
            seed_seq_rc--;
        }
    }

    free(seedStrRevComp);
    kseq_destroy(seq);
    gzclose(fp);
}

/* ── thread pool ───────────────────────────────────────────────────── */

static void *ksd_worker(void *arg) {
    ksd_pool *pool = (ksd_pool *)arg;
    for (;;) {
        pthread_mutex_lock(&pool->mutex);
        while (pool->size == 0 && !pool->shutdown)
            pthread_cond_wait(&pool->has_work, &pool->mutex);
        if (pool->shutdown && pool->size == 0) {
            pthread_mutex_unlock(&pool->mutex);
            break;
        }
        ksd_job *job = pool->queue[pool->head];
        pool->head = (pool->head + 1) % pool->capacity;
        pool->size--;
        pool->active++;
        pthread_mutex_unlock(&pool->mutex);

        scan_file(job->mg->file1, job->h, job->seed, job->mg_idx, job->total_kmer_counts);
        free(job);

        pthread_mutex_lock(&pool->mutex);
        pool->active--;
        if (pool->size == 0 && pool->active == 0)
            pthread_cond_signal(&pool->all_done);
        pthread_mutex_unlock(&pool->mutex);
    }
    return NULL;
}

static ksd_pool *ksd_pool_create(int nthreads, int capacity) {
    ksd_pool *pool = malloc(sizeof(ksd_pool));
    pool->queue    = malloc(sizeof(ksd_job *) * capacity);
    pool->head = pool->tail = pool->size = pool->active = pool->shutdown = 0;
    pool->capacity = capacity;
    pthread_mutex_init(&pool->mutex, NULL);
    pthread_cond_init(&pool->has_work, NULL);
    pthread_cond_init(&pool->all_done, NULL);
    pthread_t *threads = malloc(sizeof(pthread_t) * nthreads);
    for (int i = 0; i < nthreads; i++)
        pthread_create(&threads[i], NULL, ksd_worker, pool);
    free(threads);
    return pool;
}

static void ksd_pool_submit(ksd_pool *pool, ksd_job *job) {
    pthread_mutex_lock(&pool->mutex);
    pool->queue[pool->tail] = job;
    pool->tail = (pool->tail + 1) % pool->capacity;
    pool->size++;
    pthread_cond_signal(&pool->has_work);
    pthread_mutex_unlock(&pool->mutex);
}

static void ksd_pool_wait_and_destroy(ksd_pool *pool) {
    pthread_mutex_lock(&pool->mutex);
    while (pool->size > 0 || pool->active > 0)
        pthread_cond_wait(&pool->all_done, &pool->mutex);
    pool->shutdown = 1;
    pthread_cond_broadcast(&pool->has_work);
    pthread_mutex_unlock(&pool->mutex);
    struct timespec ts = {0, 10000000};
    nanosleep(&ts, NULL);
    pthread_mutex_destroy(&pool->mutex);
    pthread_cond_destroy(&pool->has_work);
    pthread_cond_destroy(&pool->all_done);
    free(pool->queue);
    free(pool);
}

/* ── usage ─────────────────────────────────────────────────────────── */

static void usage(void) {
    fprintf(stderr, "Usage: kmer_strain_detect -k <kmer_tsv> -B <targets.txt> -o <out.kmer_hits.tsv.gz> [-G <background.txt>] [-j threads]\n\n");
    fprintf(stderr, "  -k  kmer TSV with a '#kmer' column (plain or gzipped)\n");
    fprintf(stderr, "  -B  target genomes file (one genome path per line)\n");
    fprintf(stderr, "  -G  background metagenomes file (same format as -B); columns prefixed 'backmeta_'\n");
    fprintf(stderr, "      types: PE, SE, PEI\n");
    fprintf(stderr, "  -o  output file (e.g. sample.kmer_hits.tsv.gz)\n");
    fprintf(stderr, "  -j  threads (default: 4)\n\n");
    fprintf(stderr, "Output: gzipped TSV — first column is #kmer, then target columns,\n");
    fprintf(stderr, "        then backmeta_* columns for background metagenomes.\n");
}

/* ── main ──────────────────────────────────────────────────────────── */

int main(int argc, char *argv[]) {
    char *kmer_file       = NULL;
    char *targets_file    = NULL;
    char *background_file = NULL;
    char *outfile         = NULL;
    int   num_threads     = 4;
    int   c;

    while ((c = getopt(argc, argv, "k:B:G:o:j:h")) != EOF) {
        switch (c) {
            case 'k': kmer_file       = strdup(optarg); break;
            case 'B': targets_file    = strdup(optarg); break;
            case 'G': background_file = strdup(optarg); break;
            case 'o': outfile         = strdup(optarg); break;
            case 'j': num_threads     = atoi(optarg);   break;
            case 'h': usage(); return 0;
            default:  usage(); return 1;
        }
    }

    if (!kmer_file || !targets_file || !outfile) { usage(); return 1; }

    /* 1. parse metagenome targets */
    Metagenome *mgs;
    int n_mg = parse_targets(targets_file, &mgs);
    if (n_mg == 0) { fprintf(stderr, "no valid metagenomes in targets file\n"); return 1; }
    fprintf(stderr, "loaded %d metagenome(s)\n", n_mg);

    /* 1b. parse background metagenomes and append with backmeta_ prefix */
    int n_total = n_mg;
    if (background_file) {
        Metagenome *bg;
        int n_bg = parse_targets(background_file, &bg);
        fprintf(stderr, "loaded %d background metagenome(s)\n", n_bg);
        mgs = realloc(mgs, (n_mg + n_bg) * sizeof(Metagenome));
        for (int i = 0; i < n_bg; i++) {
            char *prefixed = malloc(strlen(bg[i].basename) + 9); /* "backmeta_" + '\0' */
            sprintf(prefixed, "backmeta_%s", bg[i].basename);
            free(bg[i].basename);
            bg[i].basename = prefixed;
            mgs[n_mg + i] = bg[i];
        }
        free(bg);
        n_total = n_mg + n_bg;
    }

    /* 2. load kmers into hash; count array has one slot per metagenome (target + background) */
    BIO_hash h = BIO_initHash(DEFAULT_GENOME_HASH_SIZE);
    char **kmer_list;
    int seed = 0, n_kmers;
    n_kmers = parse_kmer_tsv(kmer_file, h, n_total, &seed, &kmer_list);
    fprintf(stderr, "loaded %d kmer(s) (k=%d)\n", n_kmers, seed);

    /* 3. scan all metagenomes in parallel */
    uint64_t *total_kmer_counts = calloc(n_total, sizeof(uint64_t));
    ksd_pool *pool = ksd_pool_create(num_threads, n_total + 1);
    for (int i = 0; i < n_total; i++) {
        ksd_job *job = malloc(sizeof(ksd_job));
        job->mg_idx           = i;
        job->mg               = &mgs[i];
        job->h                = h;
        job->seed             = seed;
        job->total_kmer_counts = total_kmer_counts;
        ksd_pool_submit(pool, job);
    }
    ksd_pool_wait_and_destroy(pool);
    fprintf(stderr, "done scanning\n");

    /* 4. write gzipped TSV output */
    gzFile out = gzopen(outfile, "wb9");
    if (!out) {
        fprintf(stderr, "cannot open output %s: %s\n", outfile, strerror(errno));
        return 1;
    }

    /* header row */
    gzprintf(out, "#kmer");
    for (int i = 0; i < n_total; i++) gzprintf(out, "\t%s", mgs[i].basename);
    gzprintf(out, "\n");

    /* data rows — in the same order as the input TSV */
    char kmer_rc[MAX_KMER_LEN];
    for (int i = 0; i < n_kmers; i++) {
        strncpy(kmer_rc, kmer_list[i], MAX_KMER_LEN - 1);
        kmer_rc[MAX_KMER_LEN - 1] = '\0';
        BIO_reverseComplement(kmer_rc);
        char *oriented = orient_string(kmer_list[i], kmer_rc, seed);
        unsigned int *counts = (unsigned int *)BIO_searchHash(h, oriented);

        gzprintf(out, "%s", kmer_list[i]);
        for (int j = 0; j < n_total; j++)
            gzprintf(out, "\t%u", counts ? counts[j] : 0);
        gzprintf(out, "\n");
        free(kmer_list[i]);
    }

    /* summary row — total kmer positions evaluated per metagenome */
    gzprintf(out, "total_evaluated");
    for (int i = 0; i < n_total; i++)
        gzprintf(out, "\t%" PRIu64, total_kmer_counts[i]);
    gzprintf(out, "\n");

    gzclose(out);
    fprintf(stderr, "output written to %s\n", outfile);

    /* cleanup */
    free(total_kmer_counts);
    free(kmer_list);
    BIO_destroyHashD(h);
    for (int i = 0; i < n_total; i++) {
        free(mgs[i].basename);
        free(mgs[i].file1);
        free(mgs[i].file2);
    }
    free(mgs);
    free(kmer_file);
    free(targets_file);
    free(background_file);
    free(outfile);

    return 0;
}
