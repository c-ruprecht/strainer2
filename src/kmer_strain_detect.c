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
#define _GNU_SOURCE
#include "zlib.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <strings.h>
#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <time.h>
#include "kseq.h"
#include "BIO_sequence.h"
#include "BIO_hash.h"
#include "genome_compare.h"

#define NOT_PAIRED_END         0
#define IS_PAIRED_END          1
#define IS_PAIRED_END_INTERLEAVE 2
#define UNKNOWN_FILE_TYPE     -1
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

static int       get_file_type(const char *s);
static char     *strip_basename(const char *path);
static char     *pair_sample_name(const char *f1, const char *f2);
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

static int get_file_type(const char *s) {
    if (strcmp(s,"SE")==0 || strcmp(s,"se")==0) return NOT_PAIRED_END;
    if (strcmp(s,"PE")==0 || strcmp(s,"pe")==0) return IS_PAIRED_END;
    if (strcmp(s,"PEI")==0||strcmp(s,"pei")==0||strcmp(s,"IPE")==0||strcmp(s,"ipe")==0)
        return IS_PAIRED_END_INTERLEAVE;
    return UNKNOWN_FILE_TYPE;
}

static char *strip_basename(const char *path) {
    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;
    char *b = strdup(base);
    const char *exts[] = {
        ".fastq.gz", ".fasta.gz", ".fa.gz", ".fq.gz",
        ".fastq",    ".fasta",    ".fa",    ".fq",
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

/* ── sample naming for PE entries ──────────────────────────────────────
 * Take the extension-stripped basenames of both mates, keep their longest
 * common prefix (i.e. cut where the two names start to disagree), then
 * clean up what is left:
 *   1. drop trailing connectors  (_ - . : space)
 *   2. drop a trailing read tag  (R, PE, read) if it sits behind a
 *      connector, so "x_R1_001"/"x_R2_001" -> "x_R" -> "x"
 *   3. drop connectors again
 * Falls back to the stripped basename of f1 if nothing sensible is left.
 *   1001099B_150804_B6_s09_PE1 / ..._PE2  -> 1001099B_150804_B6_s09
 *   DK-D32-3524-250106-H3_R1_001 / _R2_001 -> DK-D32-3524-250106-H3
 */
static int is_connector(char c) {
    return c == '_' || c == '-' || c == '.' || c == ':' || c == ' ';
}

static void trim_connectors(char *s) {
    size_t n = strlen(s);
    while (n > 0 && is_connector(s[n-1])) s[--n] = '\0';
}

static char *pair_sample_name(const char *f1, const char *f2) {
    char *a = strip_basename(f1);
    char *b = strip_basename(f2);

    size_t i = 0;
    while (a[i] && b[i] && a[i] == b[i]) i++;
    a[i] = '\0';                       /* a is now the common prefix */
    free(b);

    trim_connectors(a);

    /* strip a dangling read tag such as "_R", "_PE", "_read" */
    static const char *tags[] = { "read", "Read", "READ", "PE", "pe", "R", "r", NULL };
    size_t n = strlen(a);
    for (int t = 0; tags[t]; t++) {
        size_t tl = strlen(tags[t]);
        if (n > tl + 1 && strcmp(a + n - tl, tags[t]) == 0 && is_connector(a[n - tl - 1])) {
            a[n - tl] = '\0';
            trim_connectors(a);
            break;
        }
    }

    if (a[0] == '\0') {                /* nothing in common: keep old behaviour */
        free(a);
        return strip_basename(f1);
    }
    return a;
}

/* ── targets parser ──────────────────────────────────────────────────
 * Two layouts are accepted:
 *
 *  1. Headerless (original):   type <TAB> file1 [<TAB> file2]
 *     Sample name is derived from the file name(s) (see pair_sample_name).
 *
 *  2. With a header line (first non-blank line, optional leading '#').
 *     Columns are found by name, in any order, case-insensitive:
 *        type | sequencing_type      PE / SE / PEI
 *        file1, file2                 or  seq_file (comma-separated mates)
 *        sample_name                  optional; used verbatim as the column
 *                                     name in the output. Empty cell -> the
 *                                     automatic name is used for that row.
 *     Other columns (sample_id, isolates_to_track, ...) are ignored.
 */

#define MAX_FIELDS 64

static int split_tabs(char *line, char **f, int max) {
    int n = 0;
    char *p = line;
    while (n < max) {
        f[n++] = p;
        char *t = strchr(p, '\t');
        if (!t) break;
        *t = '\0';
        p = t + 1;
    }
    return n;
}

static char *trim_ws(char *s) {
    while (*s == ' ' || *s == '\t') s++;
    size_t n = strlen(s);
    while (n > 0 && (s[n-1] == ' ' || s[n-1] == '\t')) s[--n] = '\0';
    return s;
}

/* NULL if the column is absent or the cell is empty */
static char *cell(char **f, int nf, int col) {
    if (col < 0 || col >= nf) return NULL;
    char *s = trim_ws(f[col]);
    return *s ? s : NULL;
}

static int parse_targets(const char *file, Metagenome **out) {
    FILE *fp = fopen(file, "r");
    if (!fp) {
        fprintf(stderr, "cannot open targets file %s: %s\n", file, strerror(errno));
        exit(1);
    }

    int cap = 64, n = 0;
    Metagenome *mgs = malloc(cap * sizeof(Metagenome));

    /* column indices; defaults = headerless layout */
    int c_type = 0, c_f1 = 1, c_f2 = 2, c_seq = -1, c_name = -1;
    int first_line = 1;

    char *line = NULL;
    size_t len = 0;
    while (getline(&line, &len, fp) != -1) {
        char *nl = strchr(line, '\n'); if (nl) *nl = '\0';
        char *cr = strchr(line, '\r'); if (cr) *cr = '\0';
        if (line[0] == '\0') continue;

        char *f[MAX_FIELDS];

        /* header detection: first non-blank line only */
        if (first_line) {
            first_line = 0;
            char *hl = strdup(line);
            char *h = hl;
            while (*h == '#') h++;
            int nf = split_tabs(h, f, MAX_FIELDS);
            int ht = -1, h1 = -1, h2 = -1, hs = -1, hn = -1;
            for (int i = 0; i < nf; i++) {
                char *c = trim_ws(f[i]);
                if      (!strcasecmp(c, "type") || !strcasecmp(c, "sequencing_type")) ht = i;
                else if (!strcasecmp(c, "file1"))       h1 = i;
                else if (!strcasecmp(c, "file2"))       h2 = i;
                else if (!strcasecmp(c, "seq_file"))    hs = i;
                else if (!strcasecmp(c, "sample_name")) hn = i;
            }
            free(hl);
            if (ht >= 0 || h1 >= 0 || hs >= 0 || hn >= 0) {
                if (ht < 0 || (h1 < 0 && hs < 0)) {
                    fprintf(stderr, "targets header needs a type/sequencing_type column and "
                                    "file1 or seq_file column: %s\n", file);
                    exit(1);
                }
                c_type = ht; c_f1 = h1; c_f2 = h2; c_seq = hs; c_name = hn;
                continue;   /* header consumed */
            }
        }

        if (line[0] == '#') continue;

        int nf = split_tabs(line, f, MAX_FIELDS);
        char *type_s = cell(f, nf, c_type);
        char *f1 = NULL, *f2 = NULL;
        if (c_seq >= 0) {
            f1 = cell(f, nf, c_seq);
            if (f1) {
                char *comma = strchr(f1, ',');
                if (comma) { *comma = '\0'; f2 = trim_ws(comma + 1); if (!*f2) f2 = NULL; }
                f1 = trim_ws(f1);
            }
        } else {
            f1 = cell(f, nf, c_f1);
            f2 = cell(f, nf, c_f2);
        }
        if (!type_s || !f1) continue;

        int type = get_file_type(type_s);
        if (type == UNKNOWN_FILE_TYPE) {
            fprintf(stderr, "unknown type '%s', skipping line\n", type_s);
            continue;
        }

        if (type == IS_PAIRED_END && !f2) {
            fprintf(stderr, "PE entry missing second file for %s, skipping\n", f1);
            continue;
        }
        if (type != IS_PAIRED_END) f2 = NULL;

        if (n == cap) { cap *= 2; mgs = realloc(mgs, cap * sizeof(Metagenome)); }
        mgs[n].type  = type;
        mgs[n].file1 = strdup(f1);
        mgs[n].file2 = f2 ? strdup(f2) : NULL;

        char *given = cell(f, nf, c_name);
        if (given)      mgs[n].basename = strdup(given);                 /* explicit sample_name */
        else if (f2)    mgs[n].basename = pair_sample_name(f1, f2);      /* PE: common prefix    */
        else            mgs[n].basename = strip_basename(f1);            /* SE/PEI: strip ext    */
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
    char *seedStrRevComp = malloc(10000);
    unsigned int i;

    while (kseq_read(seq) >= 0) {
        if ((int)seq->seq.l < seed) continue;
        BIO_stringToUpper(seq->seq.s);

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
        if (job->mg->type == IS_PAIRED_END && job->mg->file2)
            scan_file(job->mg->file2, job->h, job->seed, job->mg_idx, job->total_kmer_counts);
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
    fprintf(stderr, "  -B  target metagenomes file (tab-separated: type file1 [file2]), or with a header\n");
    fprintf(stderr, "      line naming columns: type|sequencing_type, file1/file2 or seq_file (comma-sep),\n");
    fprintf(stderr, "      and optional sample_name (used as the output column name)\n");
    fprintf(stderr, "  -G  background metagenomes file (same format as -B); columns prefixed 'b_'\n");
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
            char *prefixed = NULL;
            if (asprintf(&prefixed, "b_%s", bg[i].basename) < 0) {
                perror("asprintf"); exit(1);
            }
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
