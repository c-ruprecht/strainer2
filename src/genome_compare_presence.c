/* genome_compare_presence -- serial inverted k-mer presence index over a set
 * of genomes, plus the pairwise k-mer identity matrix that falls out of it.
 *
 *   genome_compare_presence -B <list|dir> [-k 31] [-o prefix]
 *
 * WHAT IT DOES
 * ------------
 * One serial pass over the genome list. Genome i gets strain_id i (assigned in
 * list order). Every distinct canonical k-mer of that genome is looked up in a
 * single global hash; a k-mer seen for the first time gets a new entry, one
 * already present just gets i appended to its id list. That is the inverted
 * index:
 *
 *      #kmer                              n_strains   list_strain_ids
 *      AAAA...AAA                         2           0,2
 *      ACGT...TTG                         1           1
 *
 * Because genomes are processed in id order, each id list is already sorted
 * ascending and de-duplication is a single comparison against the last element
 * -- no per-genome scratch set, no sorting of the lists at the end.
 *
 * OUTPUTS (<prefix> from -o, default "presence")
 *   <prefix>.presence.tsv[.gz|.zst]  inverted index, one row per k-mer
 *   <prefix>.manifest.tsv            strain_id -> file, with per-genome stats
 *   <prefix>.identity.tsv            square N x N Jaccard matrix
 *   <prefix>.pairs.tsv               long form: shared/union/jaccard/containment
 *
 * K-MER CONVENTION
 * ----------------
 * 2-bit packed into an unsigned __int128, so k = 1..64 (31 and 51 both fit;
 * up2bit.c is capped at 32 and uses A=0,C=1,T=2,G=3, which is why the codes
 * here are local: with A=0,C=1,G=2,T=3 the numeric order of a packed k-mer is
 * the lexicographic order of its string, so canonicalisation is one integer
 * compare). Canonical form is the lexicographically LARGER of the forward and
 * reverse-complement strings, matching orient_string()/GEN_hash_sequences() in
 * genome_compare.c so the k-mer strings printed here are the same strings the
 * scrub tools hash. -F disables canonicalisation (forward strand only).
 *
 * Any position containing a non-ACGT character (N, IUPAC codes) is skipped and
 * the rolling encoder restarts after it -- same effect as the contains_N()
 * check in the older code, but without re-scanning each window.
 *
 * IDENTITY IS OVER DISTINCT K-MERS
 * --------------------------------
 *      jaccard(i,j)     = |Ki n Kj| / |Ki u Kj|
 *      containment(i,j) = |Ki n Kj| / |Ki|
 *
 * A k-mer repeated 7x inside an rRNA operon counts once, not seven times.
 * GEN_calculate_coverage() in the old path counts k-mer POSITIONS instead, so
 * the two agree on repeat-free sequence and drift apart on repeat-rich
 * assemblies: do not carry a clustering threshold across from that tool
 * without recalibrating it on a known-ANI ladder.
 *
 * HOW THE MATRIX IS COMPUTED
 * --------------------------
 * Intersections come out of the index itself, in one sweep, split by list
 * length. A k-mer present in L strains contributes to L(L-1)/2 pairs; doing
 * that directly costs L^2/2 increments, while carrying it as one column of a
 * per-strain bitset costs N^2/128 words of popcount-AND. Rare k-mers (the long
 * tail) are cheaper the first way, near-core k-mers the second, and the two
 * are equal at L = N/8 -- which is the default crossover (-x). Both paths are
 * exact and interchangeable; -x 0 forces everything through the bitset path,
 * which is what the test script uses to cross-check them against each other.
 *
 * MEMORY
 * ------
 * ~46 B per distinct k-mer for the hash (32 B entry at 0.7 load), plus 4 B per
 * (k-mer, strain) hit beyond the second -- the first two ids live inline in the
 * entry, so k-mers private to one or two genomes never touch the allocator.
 * The bitset side is n_dense_kmers x N bits and is usually the small term.
 */

#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <ctype.h>
#include <errno.h>
#include <unistd.h>
#include <dirent.h>
#include <time.h>
#include <sys/stat.h>
#include <zlib.h>

#ifdef HAVE_ZSTD
#include <zstd.h>
#endif

#include "kseq.h"
KSEQ_INIT(gzFile, gzread)

#define DEFAULT_K              31
#define MAX_K                  64
#define DEFAULT_PREFIX         "presence"
#define INITIAL_TABLE_SLOTS    (1u << 16)   /* grows by doubling; costs ~2x the
                                             * final size in total rehash work */
#define OW_BUFFER              (1u << 20)

typedef unsigned __int128 kmer_t;

/* ---------------------------------------------------------------- helpers */

static double now_s(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

static void die(const char *what, const char *detail)
{
	if (detail)
		fprintf(stderr, "genome_compare_presence: %s: %s\n", what, detail);
	else
		fprintf(stderr, "genome_compare_presence: %s\n", what);
	exit(EXIT_FAILURE);
}

static void *xmalloc(size_t n)
{
	void *p = malloc(n ? n : 1);
	if (!p) die("out of memory", NULL);
	return p;
}

static void *xcalloc(size_t n, size_t sz)
{
	void *p = calloc(n ? n : 1, sz ? sz : 1);
	if (!p) die("out of memory", NULL);
	return p;
}

/* ------------------------------------------------------- 2-bit k-mer codes */

/* A=0 C=1 G=2 T=3 so that packed-integer order == string order. */
static uint8_t BASE_CODE[256];
static const char CODE_BASE[4] = { 'A', 'C', 'G', 'T' };

static void init_codes(void)
{
	for (int i = 0; i < 256; i++) BASE_CODE[i] = 255;
	BASE_CODE[(unsigned char)'A'] = BASE_CODE[(unsigned char)'a'] = 0;
	BASE_CODE[(unsigned char)'C'] = BASE_CODE[(unsigned char)'c'] = 1;
	BASE_CODE[(unsigned char)'G'] = BASE_CODE[(unsigned char)'g'] = 2;
	BASE_CODE[(unsigned char)'T'] = BASE_CODE[(unsigned char)'t'] = 3;
}

static void kmer_to_string(kmer_t key, int k, char *out)
{
	for (int i = 0; i < k; i++) {
		unsigned b = (unsigned)((key >> (2 * (k - 1 - i))) & 3);
		out[i] = CODE_BASE[b];
	}
	out[k] = '\0';
}

/* ------------------------------------------------------------- hash table */

#define INLINE_IDS 2

typedef struct {
	kmer_t   key;
	uint32_t n;      /* number of strains holding this k-mer; 0 == empty slot */
	uint32_t cap;    /* == INLINE_IDS while the ids live inline */
	union {
		uint32_t  inl[INLINE_IDS];
		uint32_t *ptr;
	} ids;
} kentry;

typedef struct {
	kentry  *slot;
	uint64_t cap;      /* power of two */
	uint64_t mask;
	uint64_t used;
	uint64_t limit;    /* grow when used exceeds this */
} ktable;

static inline uint32_t *entry_ids(kentry *e)
{
	return e->cap == INLINE_IDS ? e->ids.inl : e->ids.ptr;
}

static inline uint64_t mix64(uint64_t x)
{
	x += 0x9E3779B97F4A7C15ULL;
	x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
	x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
	return x ^ (x >> 31);
}

static inline uint64_t key_hash(kmer_t key)
{
	return mix64((uint64_t)key ^ mix64((uint64_t)(key >> 64)));
}

static void table_init(ktable *t, uint64_t slots)
{
	uint64_t cap = 1;
	while (cap < slots) cap <<= 1;
	t->slot  = (kentry *)xcalloc(cap, sizeof *t->slot);
	t->cap   = cap;
	t->mask  = cap - 1;
	t->used  = 0;
	t->limit = (cap / 10) * 7;          /* 0.7 load factor */
}

static void table_grow(ktable *t)
{
	kentry  *old = t->slot;
	uint64_t oldcap = t->cap;

	t->cap <<= 1;
	t->mask  = t->cap - 1;
	t->limit = (t->cap / 10) * 7;
	t->slot  = (kentry *)xcalloc(t->cap, sizeof *t->slot);

	for (uint64_t i = 0; i < oldcap; i++) {
		if (old[i].n == 0) continue;
		uint64_t j = key_hash(old[i].key) & t->mask;
		while (t->slot[j].n != 0) j = (j + 1) & t->mask;
		t->slot[j] = old[i];         /* moves the ids union verbatim */
	}
	free(old);
}

/* Record that strain `id` contains `key`. Returns 1 if this was a k-mer the
 * strain had not already contributed (i.e. a new distinct k-mer for it). */
static int table_add(ktable *t, kmer_t key, uint32_t id)
{
	uint64_t i = key_hash(key) & t->mask;

	while (t->slot[i].n != 0) {
		if (t->slot[i].key == key) {
			kentry *e = &t->slot[i];
			uint32_t *ids = entry_ids(e);
			if (ids[e->n - 1] == id) return 0;   /* lists are id-ordered */
			if (e->n == e->cap) {
				uint32_t ncap = e->cap * 2;
				uint32_t *heap = (uint32_t *)xmalloc((size_t)ncap * sizeof *heap);
				memcpy(heap, ids, (size_t)e->n * sizeof *heap);
				if (e->cap != INLINE_IDS) free(e->ids.ptr);
				e->ids.ptr = heap;
				e->cap = ncap;
				ids = heap;
			}
			ids[e->n++] = id;
			return 1;
		}
		i = (i + 1) & t->mask;
	}

	t->slot[i].key = key;
	t->slot[i].cap = INLINE_IDS;
	t->slot[i].ids.inl[0] = id;
	t->slot[i].n = 1;
	if (++t->used > t->limit) table_grow(t);
	return 1;
}

static void table_free(ktable *t)
{
	for (uint64_t i = 0; i < t->cap; i++)
		if (t->slot[i].n != 0 && t->slot[i].cap != INLINE_IDS)
			free(t->slot[i].ids.ptr);
	free(t->slot);
	t->slot = NULL;
}

/* --------------------------------------------------------- output writers */

typedef enum { OW_PLAIN, OW_GZ, OW_ZSTD } ow_kind;

typedef struct {
	ow_kind  kind;
	FILE    *fp;
	gzFile   gz;
	char    *buf;
	size_t   len, cap;
	char    *path;
#ifdef HAVE_ZSTD
	ZSTD_CCtx *cctx;
	char      *zbuf;
	size_t     zcap;
#endif
} ow;

static int has_suffix(const char *s, const char *suf)
{
	size_t ls = strlen(s), lf = strlen(suf);
	return ls >= lf && strcmp(s + ls - lf, suf) == 0;
}

static ow *ow_open(const char *path)
{
	ow *w = (ow *)xcalloc(1, sizeof *w);
	w->path = strdup(path);
	w->cap  = OW_BUFFER;
	w->buf  = (char *)xmalloc(w->cap);

	if (has_suffix(path, ".zst")) {
#ifdef HAVE_ZSTD
		w->kind = OW_ZSTD;
		w->fp = fopen(path, "wb");
		if (!w->fp) die("cannot open output", path);
		w->cctx = ZSTD_createCCtx();
		if (!w->cctx) die("ZSTD_createCCtx failed", path);
		ZSTD_CCtx_setParameter(w->cctx, ZSTD_c_compressionLevel, 3);
		w->zcap = ZSTD_CStreamOutSize();
		w->zbuf = (char *)xmalloc(w->zcap);
#else
		die("built without zstd support (rebuild with libzstd, "
		    "or use a .gz / plain output name)", path);
#endif
	} else if (has_suffix(path, ".gz")) {
		w->kind = OW_GZ;
		w->gz = gzopen(path, "wb");
		if (!w->gz) die("cannot open output", path);
	} else {
		w->kind = OW_PLAIN;
		w->fp = fopen(path, "w");
		if (!w->fp) die("cannot open output", path);
	}
	return w;
}

static void ow_drain(ow *w, int final)
{
	switch (w->kind) {
	case OW_PLAIN:
		if (w->len && fwrite(w->buf, 1, w->len, w->fp) != w->len)
			die("short write", w->path);
		w->len = 0;
		break;
	case OW_GZ:
		if (w->len && gzwrite(w->gz, w->buf, (unsigned)w->len) <= 0)
			die("short write", w->path);
		w->len = 0;
		break;
	case OW_ZSTD:
#ifdef HAVE_ZSTD
	{
		ZSTD_inBuffer in = { w->buf, w->len, 0 };
		ZSTD_EndDirective mode = final ? ZSTD_e_end : ZSTD_e_continue;
		for (;;) {
			ZSTD_outBuffer out = { w->zbuf, w->zcap, 0 };
			size_t rem = ZSTD_compressStream2(w->cctx, &out, &in, mode);
			if (ZSTD_isError(rem)) die("zstd error", ZSTD_getErrorName(rem));
			if (out.pos && fwrite(w->zbuf, 1, out.pos, w->fp) != out.pos)
				die("short write", w->path);
			if (final) { if (rem == 0) break; }
			else if (in.pos == in.size) break;
		}
		w->len = 0;
	}
#endif
		break;
	}
	(void)final;
}

static inline void ow_reserve(ow *w, size_t need)
{
	if (w->len + need > w->cap) {
		ow_drain(w, 0);
		if (need > w->cap) {
			w->cap = need * 2;
			w->buf = (char *)realloc(w->buf, w->cap);
			if (!w->buf) die("out of memory", NULL);
		}
	}
}

static inline void ow_puts(ow *w, const char *s, size_t n)
{
	ow_reserve(w, n);
	memcpy(w->buf + w->len, s, n);
	w->len += n;
}

static inline void ow_u32(ow *w, uint32_t v)
{
	char tmp[12];
	int i = 12;
	do { tmp[--i] = (char)('0' + v % 10); v /= 10; } while (v);
	ow_puts(w, tmp + i, (size_t)(12 - i));
}

static inline void ow_ch(ow *w, char c)
{
	ow_reserve(w, 1);
	w->buf[w->len++] = c;
}

static void ow_close(ow *w)
{
	ow_drain(w, 1);
	switch (w->kind) {
	case OW_GZ:   gzclose(w->gz); break;
	case OW_ZSTD:
#ifdef HAVE_ZSTD
		ZSTD_freeCCtx(w->cctx);
		free(w->zbuf);
#endif
		fclose(w->fp);
		break;
	case OW_PLAIN: fclose(w->fp); break;
	}
	free(w->buf);
	free(w->path);
	free(w);
}

/* ------------------------------------------------------------ input paths */

typedef struct {
	char **path;
	size_t n, cap;
} pathvec;

static void pv_push(pathvec *v, const char *p)
{
	if (v->n == v->cap) {
		v->cap = v->cap ? v->cap * 2 : 64;
		v->path = (char **)realloc(v->path, v->cap * sizeof *v->path);
		if (!v->path) die("out of memory", NULL);
	}
	v->path[v->n++] = strdup(p);
}

static int is_fasta_name(const char *name)
{
	static const char *suf[] = {
		".fna.gz", ".fa.gz", ".fasta.gz", ".fsa.gz", ".fna", ".fa",
		".fasta", ".fsa", ".ffn", ".ffn.gz", NULL
	};
	for (int i = 0; suf[i]; i++)
		if (has_suffix(name, suf[i])) return 1;
	return 0;
}

static int cmp_str(const void *a, const void *b)
{
	char *const *x = (char *const *)a;
	char *const *y = (char *const *)b;
	return strcmp(*x, *y);
}

/* Append every FASTA file directly inside `dir`, sorted, so a run is
 * reproducible regardless of readdir order. */
static void add_directory(pathvec *v, const char *dir)
{
	DIR *d = opendir(dir);
	if (!d) { fprintf(stderr, "warning: cannot open directory %s: %s\n",
	                  dir, strerror(errno)); return; }

	pathvec local = { NULL, 0, 0 };
	struct dirent *de;
	while ((de = readdir(d)) != NULL) {
		if (de->d_name[0] == '.') continue;
		if (!is_fasta_name(de->d_name)) continue;
		char buf[4096];
		snprintf(buf, sizeof buf, "%s/%s", dir, de->d_name);
		struct stat st;
		if (stat(buf, &st) == 0 && S_ISDIR(st.st_mode)) continue;
		pv_push(&local, buf);
	}
	closedir(d);

	qsort(local.path, local.n, sizeof *local.path, cmp_str);
	for (size_t i = 0; i < local.n; i++) {
		pv_push(v, local.path[i]);
		free(local.path[i]);
	}
	free(local.path);

	if (local.n == 0)
		fprintf(stderr, "warning: no FASTA files found in %s\n", dir);
}

/* -B accepts either a directory of genomes or a text file listing genome
 * paths, where each listed path may itself be a file or a directory. */
static pathvec collect_inputs(const char *arg)
{
	pathvec v = { NULL, 0, 0 };
	struct stat st;

	if (stat(arg, &st) != 0) die("cannot stat input", arg);

	if (S_ISDIR(st.st_mode)) {
		add_directory(&v, arg);
		return v;
	}

	FILE *fp = fopen(arg, "r");
	if (!fp) die("cannot open genome list", arg);

	char *line = NULL;
	size_t cap = 0;
	ssize_t len;
	while ((len = getline(&line, &cap, fp)) != -1) {
		while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r' ||
		                   line[len - 1] == ' '  || line[len - 1] == '\t'))
			line[--len] = '\0';
		char *p = line;
		while (*p == ' ' || *p == '\t') p++;
		if (*p == '\0' || *p == '#') continue;

		struct stat es;
		if (stat(p, &es) != 0) {
			fprintf(stderr, "warning: skipping unreadable path %s\n", p);
			continue;
		}
		if (S_ISDIR(es.st_mode)) add_directory(&v, p);
		else                     pv_push(&v, p);
	}
	free(line);
	fclose(fp);
	return v;
}

static const char *base_name(const char *path)
{
	const char *s = strrchr(path, '/');
	return s ? s + 1 : path;
}

/* basename with one or two FASTA extensions removed: foo.fna.gz -> foo */
static char *sample_id_of(const char *path)
{
	char *s = strdup(base_name(path));
	for (int round = 0; round < 2; round++) {
		char *dot = strrchr(s, '.');
		if (!dot) break;
		if (strcmp(dot, ".gz") == 0 || strcmp(dot, ".zst") == 0 ||
		    strcmp(dot, ".bz2") == 0 || strcmp(dot, ".fna") == 0 ||
		    strcmp(dot, ".fa") == 0 || strcmp(dot, ".fasta") == 0 ||
		    strcmp(dot, ".fsa") == 0 || strcmp(dot, ".ffn") == 0)
			*dot = '\0';
		else break;
	}
	return s;
}

/* -------------------------------------------------------- genome scanning */

typedef struct {
	uint64_t bases;
	uint64_t positions;      /* k-mer positions actually encoded (non-N) */
	uint64_t distinct;       /* distinct canonical k-mers contributed */
	uint32_t contigs;
	uint32_t short_contigs;  /* contigs shorter than k */
} gstat;

static int scan_genome(const char *path, int k, int canonical, uint32_t id,
                       ktable *t, gstat *gs)
{
	gzFile fp = gzopen(path, "r");
	if (!fp) { fprintf(stderr, "warning: cannot open %s\n", path); return -1; }

	kseq_t *seq = kseq_init(fp);
	const kmer_t mask = (k == 64) ? (~(kmer_t)0)
	                              : ((((kmer_t)1) << (2 * k)) - 1);
	const unsigned shift = (unsigned)(2 * (k - 1));
	int l;

	while ((l = kseq_read(seq)) >= 0) {
		gs->contigs++;
		gs->bases += (uint64_t)seq->seq.l;
		if ((size_t)l < (size_t)k) { gs->short_contigs++; continue; }

		kmer_t fwd = 0, rc = 0;
		int filled = 0;
		const char *s = seq->seq.s;

		for (size_t i = 0; i < seq->seq.l; i++) {
			uint8_t code = BASE_CODE[(unsigned char)s[i]];
			if (code > 3) { filled = 0; fwd = rc = 0; continue; }

			fwd = ((fwd << 2) | (kmer_t)code) & mask;
			rc  = (rc >> 2) | (((kmer_t)(3 - code)) << shift);

			if (filled < k) filled++;
			if (filled < k) continue;

			gs->positions++;
			kmer_t key = fwd;
			if (canonical && rc > fwd) key = rc;
			gs->distinct += (uint64_t)table_add(t, key, id);
		}
	}

	kseq_destroy(seq);
	gzclose(fp);
	return 0;
}

/* ------------------------------------------------------------------ popcnt */

static inline uint64_t and_popcount(const uint64_t *a, const uint64_t *b,
                                    uint64_t words)
{
	uint64_t n = 0;
	for (uint64_t w = 0; w < words; w++)
		n += (uint64_t)__builtin_popcountll(a[w] & b[w]);
	return n;
}

static int cmp_entry_ptr(const void *a, const void *b)
{
	const kentry *x = *(const kentry *const *)a;
	const kentry *y = *(const kentry *const *)b;
	if (x->key < y->key) return -1;
	if (x->key > y->key) return 1;
	return 0;
}

/* ------------------------------------------------------------------- main */

static void usage(void)
{
	fprintf(stderr,
"Usage: genome_compare_presence -B <genome list or directory> [options]\n"
"\n"
"  -B <path>   REQUIRED. Either a directory holding .fna.gz files, or a text\n"
"              file listing one genome path per line; a listed path may itself\n"
"              be a directory (expanded, sorted). Blank lines and #comments are\n"
"              ignored. Accepts .fna/.fa/.fasta/.ffn, plain or gzipped.\n"
"  -k <int>    k-mer length, 1..%d (default %d; 51 works).\n"
"  -o <prefix> output prefix (default \"%s\").\n"
"  -z          compress the presence file (.zst if built with libzstd,\n"
"              otherwise .gz).\n"
"  -F          forward strand only; do not fold k-mers to their canonical\n"
"              (lexicographically larger) orientation.\n"
"  -m <int>    write only k-mers present in at least this many strains\n"
"              (default 1). Filters the presence FILE only.\n"
"  -M <int>    write only k-mers present in at most this many strains\n"
"              (default 0 = no cap). Filters the presence FILE only; the\n"
"              identity matrix always uses every k-mer.\n"
"  -U          do not sort the presence file by k-mer (faster, hash order).\n"
"  -x <int>    strain-count crossover between the direct and bitset paths for\n"
"              the intersection sweep (default auto = N/8). Tuning only; both\n"
"              paths give identical numbers. -x 0 forces all-bitset.\n"
"  -h          this message.\n"
"\n"
"Outputs:\n"
"  <prefix>.presence.tsv[.gz|.zst]  #kmer, n_strains, list_strain_ids\n"
"  <prefix>.manifest.tsv            strain_id, sample_id, filename, path,\n"
"                                   n_contigs, n_bases, n_kmer_positions,\n"
"                                   n_kmers_distinct\n"
"  <prefix>.identity.tsv            square Jaccard matrix (rows = strain_id)\n"
"  <prefix>.pairs.tsv               strain_a, strain_b, n_shared, n_union,\n"
"                                   jaccard, containment_a_in_b,\n"
"                                   containment_b_in_a\n"
"\n"
"Load the presence file with polars:\n"
"  pl.scan_csv('presence.tsv', separator='\\t')\n"
"    .with_columns(pl.col('list_strain_ids').str.split(','))\n",
	MAX_K, DEFAULT_K, DEFAULT_PREFIX);
	exit(EXIT_FAILURE);
}

int main(int argc, char *argv[])
{
	const char *b_arg = NULL;
	const char *prefix = DEFAULT_PREFIX;
	int k = DEFAULT_K;
	int canonical = 1, compress = 0, sorted = 1;
	long min_strains = 1, max_strains = 0, crossover = -1;
	int c;

	while ((c = getopt(argc, argv, "B:k:s:o:m:M:x:zFUh")) != -1) {
		switch (c) {
		case 'B': b_arg = optarg; break;
		case 'k':
		case 's': k = atoi(optarg); break;      /* -s: same letter the other tools use */
		case 'o': prefix = optarg; break;
		case 'm': min_strains = atol(optarg); break;
		case 'M': max_strains = atol(optarg); break;
		case 'x': crossover = atol(optarg); break;
		case 'z': compress = 1; break;
		case 'F': canonical = 0; break;
		case 'U': sorted = 0; break;
		case 'h':
		default: usage();
		}
	}

	if (!b_arg) usage();
	if (k < 1 || k > MAX_K)
		die("-k must be between 1 and 64", NULL);
	if (min_strains < 1) min_strains = 1;

	init_codes();
	double t0 = now_s();

	pathvec files = collect_inputs(b_arg);
	if (files.n == 0) die("no genome files found for -B", b_arg);
	if (files.n > UINT32_MAX) die("too many genomes", NULL);
	const uint32_t N = (uint32_t)files.n;

	fprintf(stderr, "[presence] %u genomes, k=%d, %s orientation\n",
	        N, k, canonical ? "canonical" : "forward-only");

	/* ---- pass over the genomes: build the inverted index -------------- */
	ktable table;
	table_init(&table, INITIAL_TABLE_SLOTS);
	gstat *gs = (gstat *)xcalloc(N, sizeof *gs);

	for (uint32_t i = 0; i < N; i++) {
		scan_genome(files.path[i], k, canonical, i, &table, &gs[i]);
		if (N > 20 && (i + 1) % 50 == 0)
			fprintf(stderr, "[presence]   %u/%u genomes, %" PRIu64
			        " distinct k-mers so far\n",
			        i + 1, N, table.used);
	}

	double t1 = now_s();
	fprintf(stderr, "[presence] index: %" PRIu64 " distinct %d-mers "
	        "(%.1f MB hash) [%.2fs]\n",
	        table.used, k,
	        (double)table.cap * sizeof(kentry) / 1e6, t1 - t0);

	/* ---- manifest ----------------------------------------------------- */
	char path[4096];
	snprintf(path, sizeof path, "%s.manifest.tsv", prefix);
	FILE *mf = fopen(path, "w");
	if (!mf) die("cannot open manifest", path);
	fprintf(mf, "strain_id\tsample_id\tfilename\tpath\tn_contigs\tn_bases\t"
	            "n_kmer_positions\tn_kmers_distinct\n");
	for (uint32_t i = 0; i < N; i++) {
		char *sid = sample_id_of(files.path[i]);
		fprintf(mf, "%u\t%s\t%s\t%s\t%u\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\n",
		        i, sid, base_name(files.path[i]), files.path[i],
		        gs[i].contigs, gs[i].bases, gs[i].positions, gs[i].distinct);
		if (gs[i].short_contigs)
			fprintf(stderr, "warning: %s has %u contig(s) shorter than k=%d\n",
			        base_name(files.path[i]), gs[i].short_contigs, k);
		if (gs[i].distinct == 0)
			fprintf(stderr, "warning: %s contributed no k-mers\n",
			        base_name(files.path[i]));
		free(sid);
	}
	fclose(mf);
	fprintf(stderr, "[presence] wrote %s.manifest.tsv\n", prefix);

	/* ---- collect entries (sorted by default for reproducible output) --- */
	kentry **ent = (kentry **)xmalloc((size_t)table.used * sizeof *ent);
	size_t nent = 0;
	for (uint64_t i = 0; i < table.cap; i++)
		if (table.slot[i].n != 0) ent[nent++] = &table.slot[i];
	if (sorted) qsort(ent, nent, sizeof *ent, cmp_entry_ptr);

	/* auto crossover: direct sweep costs L^2/2 increments, the bitset path
	 * N^2/128 words -- equal at L = N/8. */
	uint64_t thresh;
	if (crossover >= 0)      thresh = (uint64_t)crossover;
	else if (N < 64)         thresh = N;              /* never dense for tiny N */
	else                     thresh = N / 8 < 16 ? 16 : N / 8;

	uint64_t n_dense = 0;
	for (size_t i = 0; i < nent; i++)
		if (ent[i]->n > thresh) n_dense++;

	uint64_t words = (n_dense + 63) / 64;
	uint64_t *bits = NULL;
	if (n_dense) {
		double gb = (double)N * (double)words * 8.0 / 1e9;
		if (gb > 64.0)
			fprintf(stderr, "warning: bitset needs %.1f GB; raise -x to push "
			        "more k-mers through the direct path\n", gb);
		bits = (uint64_t *)xcalloc((size_t)N * words, sizeof *bits);
		fprintf(stderr, "[presence] %" PRIu64 " k-mers in >%" PRIu64
		        " strains -> bitset (%.1f MB), rest direct\n",
		        n_dense, thresh, (double)N * words * 8 / 1e6);
	}

	uint64_t *inter = (uint64_t *)xcalloc((size_t)N * N, sizeof *inter);

	/* ---- one sweep: write presence rows and accumulate intersections --- */
	snprintf(path, sizeof path, "%s.presence.tsv%s", prefix,
	         compress ?
#ifdef HAVE_ZSTD
	         ".zst"
#else
	         ".gz"
#endif
	         : "");
	ow *pw = ow_open(path);
	const char *phdr = "#kmer\tn_strains\tlist_strain_ids\n";
	ow_puts(pw, phdr, strlen(phdr));

	char kbuf[MAX_K + 1];
	uint64_t written = 0, dense_col = 0, singletons = 0, core = 0;

	for (size_t i = 0; i < nent; i++) {
		kentry *e = ent[i];
		uint32_t *ids = entry_ids(e);

		if (e->n == 1) singletons++;
		if (e->n == N) core++;

		if (e->n > thresh) {
			uint64_t w = dense_col >> 6;
			uint64_t b = 1ULL << (dense_col & 63);
			for (uint32_t a = 0; a < e->n; a++)
				bits[(size_t)ids[a] * words + w] |= b;
			dense_col++;
		} else {
			for (uint32_t a = 0; a < e->n; a++)
				for (uint32_t b = a + 1; b < e->n; b++)
					inter[(size_t)ids[a] * N + ids[b]]++;
		}

		if ((long)e->n < min_strains) continue;
		if (max_strains > 0 && (long)e->n > max_strains) continue;

		kmer_to_string(e->key, k, kbuf);
		ow_puts(pw, kbuf, (size_t)k);
		ow_ch(pw, '\t');
		ow_u32(pw, e->n);
		ow_ch(pw, '\t');
		for (uint32_t a = 0; a < e->n; a++) {
			if (a) ow_ch(pw, ',');
			ow_u32(pw, ids[a]);
		}
		ow_ch(pw, '\n');
		written++;
	}
	ow_close(pw);
	fprintf(stderr, "[presence] wrote %s (%" PRIu64 " of %zu k-mers)\n",
	        path, written, nent);

	if (n_dense) {
		for (uint32_t i = 0; i < N; i++)
			for (uint32_t j = i + 1; j < N; j++)
				inter[(size_t)i * N + j] +=
					and_popcount(bits + (size_t)i * words,
					             bits + (size_t)j * words, words);
		free(bits);
	}

	double t2 = now_s();
	fprintf(stderr, "[presence] intersections done [%.2fs]\n", t2 - t1);

	/* ---- identity matrix + long-form pairs ---------------------------- */
	char **sid = (char **)xmalloc((size_t)N * sizeof *sid);
	for (uint32_t i = 0; i < N; i++) sid[i] = sample_id_of(files.path[i]);

	snprintf(path, sizeof path, "%s.identity.tsv", prefix);
	FILE *im = fopen(path, "w");
	if (!im) die("cannot open identity matrix", path);
	fprintf(im, "strain_id\tsample_id");
	for (uint32_t j = 0; j < N; j++) fprintf(im, "\t%s", sid[j]);
	fprintf(im, "\n");

	for (uint32_t i = 0; i < N; i++) {
		fprintf(im, "%u\t%s", i, sid[i]);
		for (uint32_t j = 0; j < N; j++) {
			double v;
			if (i == j) {
				v = gs[i].distinct ? 1.0 : 0.0;
			} else {
				uint32_t a = i < j ? i : j, b = i < j ? j : i;
				uint64_t sh = inter[(size_t)a * N + b];
				uint64_t un = gs[i].distinct + gs[j].distinct - sh;
				v = un ? (double)sh / (double)un : 0.0;
			}
			fprintf(im, "\t%.6f", v);
		}
		fprintf(im, "\n");
	}
	fclose(im);
	fprintf(stderr, "[presence] wrote %s.identity.tsv\n", prefix);

	snprintf(path, sizeof path, "%s.pairs.tsv", prefix);
	FILE *pf = fopen(path, "w");
	if (!pf) die("cannot open pairs file", path);
	fprintf(pf, "strain_a\tstrain_b\tsample_a\tsample_b\tn_kmers_a\tn_kmers_b\t"
	            "n_shared\tn_union\tjaccard\tcontainment_a_in_b\t"
	            "containment_b_in_a\n");
	for (uint32_t i = 0; i < N; i++)
		for (uint32_t j = i + 1; j < N; j++) {
			uint64_t sh = inter[(size_t)i * N + j];
			uint64_t un = gs[i].distinct + gs[j].distinct - sh;
			fprintf(pf, "%u\t%u\t%s\t%s\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64
			            "\t%" PRIu64 "\t%.6f\t%.6f\t%.6f\n",
			        i, j, sid[i], sid[j], gs[i].distinct, gs[j].distinct,
			        sh, un,
			        un ? (double)sh / (double)un : 0.0,
			        gs[i].distinct ? (double)sh / (double)gs[i].distinct : 0.0,
			        gs[j].distinct ? (double)sh / (double)gs[j].distinct : 0.0);
		}
	fclose(pf);
	fprintf(stderr, "[presence] wrote %s.pairs.tsv\n", prefix);

	fprintf(stderr, "[presence] %" PRIu64 " k-mers in exactly 1 strain, "
	        "%" PRIu64 " in all %u; total %.2fs\n",
	        singletons, core, N, now_s() - t0);

	/* ---- cleanup ------------------------------------------------------ */
	for (uint32_t i = 0; i < N; i++) { free(sid[i]); free(files.path[i]); }
	free(sid);
	free(files.path);
	free(inter);
	free(ent);
	free(gs);
	table_free(&table);
	return 0;
}
