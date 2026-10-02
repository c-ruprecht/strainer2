"""Assign scrub-DB lineages (and GTDB taxonomy) to one or more strain genomes.

Also imported by kmer_block_filter.py for call_lineage / get_lineage_kmers."""
import argparse
import gzip
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import polars as pl

LINEAGE_COL = 'lineage_id'
TAX_COL = 'gtdb_taxonomy'
DEFAULT_SKETCH_SCALE = 100
DEFAULT_MIN_FRAC_LINEAGE = 0.7

# Mirrors COMPLEMENT[] in BIO_sequence.c (incl. its K -> '.' quirk)
_COMP = str.maketrans('ACGTRYMKSWBVDHN', 'TGCAYRK.SWVBHDN')
_FASTA_EXTS = ('.fna.gz', '.fasta.gz', '.fa.gz', '.fna', '.fasta', '.fa')


def strain_name_from_path(path):
    base = os.path.basename(path)
    for ext in _FASTA_EXTS:
        if base.endswith(ext):
            return base[: -len(ext)]
    return base.split('.')[0]


def _read_fasta(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    name, chunks = None, []
    with opener(path, 'rt') as fh:
        for line in fh:
            if line.startswith('>'):
                if name is not None:
                    yield name, ''.join(chunks)
                name, chunks = line[1:].split()[0], []
            else:
                chunks.append(line.strip())
    if name is not None:
        yield name, ''.join(chunks)


def create_kmers(genome, kmer_size=31):
    """Canonical k-mers exactly as GEN_hash_sequences_set_count_vec builds them:
    uppercase, skip windows containing 'N', keep max(forward, revcomp)."""
    kmers = set()
    k = kmer_size
    for _, seq in _read_fasta(genome):
        s = seq.upper()
        rc = s.translate(_COMP)[::-1]
        L = len(s)
        for i in range(L - k + 1):
            f = s[i:i + k]
            if 'N' in f:
                continue
            r = rc[L - i - k:L - i]
            kmers.add(f if f >= r else r)
    return kmers


def kmers_to_frame(kmers):
    """set of kmers -> single-column pl.DataFrame expected by call_lineage."""
    return pl.DataFrame({'#kmer': list(kmers)}, schema={'#kmer': pl.Utf8})


def call_lineage(query_kmers, lineage_db, lineage_col=LINEAGE_COL, tax_col=TAX_COL,
                 metric='n_hits', top_n=5, verbose=True, lineages=None):
    """Group lineage-DB hits of the query kmers by lineage and pick the best one.

    query_kmers : pl.DataFrame with a single '#kmer' column (unique)
    Returns (lineage_call, gtdb_tax, df_calls) where df_calls holds the per-lineage
    scores for every lineage with >= 1 hit.
    lineages    : optional list of lineage ids; only these are scored (exact counts).
    """
    lf = pl.scan_parquet(lineage_db)
    cols = lf.collect_schema().names()
    missing = {'#kmer', lineage_col, tax_col} - set(cols)
    if missing:
        raise ValueError(f"lineage_db missing columns {sorted(missing)}, found: {cols}")
    if lineages is not None:
        lf = lf.filter(pl.col(lineage_col).is_in(list(lineages)))

    # purity metadata is optional: DBs built before it was added simply lack it
    meta_cols = [c for c in ('n_labels', 'lineage_purity', 'lineage_members') if c in cols]
    aggs = [pl.len().alias('n_hits'), pl.col(tax_col).first().alias('gtdb_tax')]
    aggs += [pl.col(c).first() for c in meta_cols]

    df_hits = (lf.join(query_kmers.lazy(), on='#kmer', how='semi')
                 .group_by(lineage_col)
                 .agg(aggs)
                 .collect(engine='streaming'))

    if df_hits.height == 0:
        print('WARNING: no query kmers found in lineage_db — no lineage call')
        return None, None, df_hits

    df_tot = (lf.join(df_hits.lazy().select(lineage_col), on=lineage_col, how='semi')
                .group_by(lineage_col)
                .agg(pl.len().alias('n_lineage_kmers'))
                .collect(engine='streaming'))

    df_calls = (df_hits.join(df_tot, on=lineage_col, how='left')
                .with_columns(
                    (pl.col('n_hits') / pl.col('n_lineage_kmers')).alias('frac_lineage_hit'))
                .sort([metric, 'n_hits'], descending=True))

    if verbose:
        print(f'Lineage hits (top {top_n} of {df_calls.height}):')
        print(df_calls.head(top_n))

    best = df_calls.row(0, named=True)
    if df_calls.height > 1:
        second = df_calls.row(1, named=True)
        ratio = second[metric] / best[metric] if best[metric] else float('nan')
        if ratio > 0.5:
            print(f"WARNING: ambiguous lineage call — runner-up {second[lineage_col]} "
                  f"scores {ratio:.2f}x of best")

    if verbose:
        print(f"Lineage call: {best[lineage_col]}  ({best['n_hits']} hits, "
              f"{best['frac_lineage_hit']:.3f} of lineage kmers)")
        print(f"GTDB tax:     {best['gtdb_tax']}")
    return best[lineage_col], best['gtdb_tax'], df_calls


def get_lineage_kmers(lineage_db, lineage, lineage_col=LINEAGE_COL):
    """All unique kmers of the called lineage with their DB block id and block size."""
    return (pl.scan_parquet(lineage_db)
              .filter(pl.col(lineage_col) == lineage)
              .select('#kmer', 'kmer_block_id', 'block_n_kmers')
              .sort('kmer_block_id')
              .unique(subset='#kmer', keep='first', maintain_order=True)
              .collect(engine='streaming'))


def sketch_path_for(lineage_db, sketch_scale):
    base = lineage_db[:-len('.parquet')] if lineage_db.endswith('.parquet') else lineage_db
    return f'{base}.fmh{sketch_scale}.parquet'


def build_sketch(lineage_db, out_path, sketch_scale=DEFAULT_SKETCH_SCALE,
                 lineage_col=LINEAGE_COL):
    """One-off FracMinHash sketch: keep every kmer whose hash is in the lowest
    1/sketch_scale of the hash space. That is a plain row filter, so it streams
    through the DB in constant memory; only the ~1/sketch_scale survivors are
    deduplicated and counted in memory.

    Hash rather than sort order, because lexicographically first kmers are
    poly-A rich. The hash is only used here; the query side just checks
    membership, so hash stability across polars versions doesn't matter."""
    print(f'building lineage sketch -> {out_path} (one-off)')
    # pid in the temp names and a rename at the end: several jobs may reach this
    # at once, and a half-written sketch must never appear under out_path
    tmp = f'{out_path}.{os.getpid()}.raw'
    part = f'{out_path}.{os.getpid()}.part'
    threshold = (2 ** 64) // sketch_scale
    (pl.scan_parquet(lineage_db)
       .select('#kmer', lineage_col)
       .filter(pl.col('#kmer').hash(seed=42) < threshold)
       .sink_parquet(tmp))
    (pl.scan_parquet(tmp)
       .unique()
       .with_columns(pl.len().over(lineage_col).alias('n_sketch'))
       .collect()
       .write_parquet(part))
    os.replace(part, out_path)
    os.remove(tmp)


def ensure_sketch(lineage_db, sketch_scale=DEFAULT_SKETCH_SCALE):
    """Path to the prefilter sketch for this DB, building it if it is missing or
    older than the DB (a stale sketch would prefilter against lineages that no
    longer exist). Call it before starting workers, not inside them."""
    path = sketch_path_for(lineage_db, sketch_scale)
    if not os.path.exists(path):
        build_sketch(lineage_db, path, sketch_scale)
    elif os.path.getmtime(path) < os.path.getmtime(lineage_db):
        print(f'sketch {path} is older than the lineage_db — rebuilding')
        build_sketch(lineage_db, path, sketch_scale)
    return path


def candidate_lineages(query_kmers, sketch_path, top_k=5, min_ratio=0.2,
                       lineage_col=LINEAGE_COL):
    """Stage 1: estimate containment per lineage from its sketch and keep the
    top_k lineages scoring >= min_ratio of the best."""
    est = (pl.scan_parquet(sketch_path)
             .join(query_kmers.lazy(), on='#kmer', how='semi')
             .group_by(lineage_col)
             .agg((pl.len() / pl.col('n_sketch').first()).alias('est_frac'))
             .sort('est_frac', descending=True)
             .head(top_k)
             .collect())
    if est.height == 0:
        return []
    best = est['est_frac'][0]
    return est.filter(pl.col('est_frac') >= min_ratio * best)[lineage_col].to_list()


def collect_genomes(args):
    if args.genome:
        return [Path(args.genome)]
    if args.genome_dir:
        return sorted(p for p in Path(args.genome_dir).iterdir()
                      if p.name.endswith(_FASTA_EXTS))
    with open(args.genome_list) as fh:
        return [Path(l.strip()) for l in fh if l.strip()]


NOT_FOUND = 'not_found'


def minor_labels(members, top_n=3):
    """The GTDB labels in the lineage other than its most common one, as
    'label:count', biggest first. Shows what else a call can mean: a lineage at
    purity 0.06 is a complex, and these are the organisms in it.

    Labels are species strings ('Collinsella aerofaciens_M'), so the words every
    member shares are dropped and only the part where they disagree is kept --
    usually the epithet, since members normally share a genus. When the genus
    itself differs nothing is shared and the full labels are kept."""
    if members is None:
        return None
    counts = []
    for part in str(members).split(','):
        label, _, n = part.rpartition(':')
        try:
            counts.append((int(n), label.strip()))
        except ValueError:
            return None
    counts.sort(reverse=True)
    rest = counts[1:]
    if not rest:
        return ''

    words = [lab.split() for _, lab in counts]
    shared = 0
    for col in zip(*words):
        if len(set(col)) > 1:
            break
        shared += 1
    shared = min(shared, min(len(w) for w in words) - 1)  # never strip a whole label

    out = [f"{' '.join(lab.split()[shared:])}:{n}" for n, lab in rest[:top_n]]
    if len(rest) > top_n:
        out.append(f'(+{len(rest) - top_n} more)')
    return ','.join(out)


CALL_TRUE = 'true'
CALL_LOW = 'too_low_coverage'
CALL_AMBIGUOUS = 'ambiguous'
AMBIGUOUS_RATIO = 0.5  # same cutoff as the warning in call_lineage


def summarise(name, df_calls, min_frac_lineage=0.0, metric='frac_lineage_hit'):
    """One row per genome: best lineage and its scores, plus lineage_call:
      true              best lineage has frac_lineage_hit > min_frac_lineage
      too_low_coverage  best lineage found, but covered below the threshold
                        (lineage/scores are still reported)
      not_found         no hit anywhere in the DB: lineage_id/gtdb_tax say
                        not_found and n_hits is 0. n_lineage_kmers and
                        frac_lineage_hit stay empty, since without a lineage
                        there is nothing to count against.
    ';ambiguous' is appended (e.g. 'true;ambiguous') when the runner-up scores
    more than AMBIGUOUS_RATIO of the best on `metric`. The runner-up_* columns
    describe the second-best lineage whether or not it is ambiguous."""
    if df_calls is None or df_calls.height == 0:
        return {'genome': name, LINEAGE_COL: NOT_FOUND, 'gtdb_tax': NOT_FOUND,
                'lineage_call': NOT_FOUND, 'n_hits': 0}
    best = df_calls.row(0, named=True)
    call = CALL_TRUE if best['frac_lineage_hit'] > min_frac_lineage else CALL_LOW
    row = {'genome': name, LINEAGE_COL: best[LINEAGE_COL], 'gtdb_tax': best['gtdb_tax'],
           'lineage_call': call,
           'n_hits': best['n_hits'], 'n_lineage_kmers': best['n_lineage_kmers'],
           'frac_lineage_hit': best['frac_lineage_hit']}
    for c in ('n_labels', 'lineage_purity'):
        if c in best:
            row[c] = best[c]
    if 'lineage_members' in best:
        row['other_labels'] = minor_labels(best['lineage_members'])

    if df_calls.height > 1:
        second = df_calls.row(1, named=True)
        ratio = second[metric] / best[metric] if best[metric] else float('nan')
        row['runner_up_lineage_id'] = second[LINEAGE_COL]
        row['runner_up_gtdb_tax'] = second['gtdb_tax']
        row['runner_up_frac_lineage_hit'] = second['frac_lineage_hit']
        row['runner_up_n_hits'] = second['n_hits']
        row['runner_up_ratio'] = ratio
        if ratio > AMBIGUOUS_RATIO:
            row['lineage_call'] = f'{call};{CALL_AMBIGUOUS}'
    return row


def tax_one(genome, lineage_db, sketch_path, metric, kmer_size, top_k,
            min_frac_lineage=0.0):
    """Worker: k-mers -> sketch prefilter -> exact call on candidates -> summary row."""
    name = strain_name_from_path(str(genome))
    query = kmers_to_frame(create_kmers(genome, kmer_size))
    cands = candidate_lineages(query, sketch_path, top_k=top_k) if sketch_path else None
    if cands == []:  # nothing hit the sketch -> fall back to the full scan
        cands = None
    _, _, df_calls = call_lineage(query, lineage_db, metric=metric, verbose=False,
                                  lineages=cands)
    return summarise(name, df_calls, min_frac_lineage, metric)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--genome', help='a single genome file')
    src.add_argument('--genome_dir', help='a directory containing genomes')
    src.add_argument('--genome_list', help='text file with one genome path per line')
    parser.add_argument('--lineage_db', required=True, help='path to lineage .parquet')
    parser.add_argument('--lineage_metric', choices=['n_hits', 'frac_lineage_hit'], default='frac_lineage_hit')
    parser.add_argument('--kmer_size', type=int, default=31)
    parser.add_argument('--threads', type=int, default=4, help='genomes processed in parallel')
    parser.add_argument('--output', default='lineage_calls.tsv',
                        help='output TSV, or a directory (writes lineage_calls.tsv in it)')
    parser.add_argument('--sketch_scale', type=int, default=DEFAULT_SKETCH_SCALE,
                        help='prefilter sketch keeps ~1 in N lineage kmers')
    parser.add_argument('--top_k', type=int, default=5,
                        help='candidate lineages scored exactly after the prefilter')
    parser.add_argument('--min_frac_lineage', type=float, default=DEFAULT_MIN_FRAC_LINEAGE,
                        help='lineage_call is true only if more than this fraction of the best '
                             "lineage's kmers is covered by the genome, else "
                             'too_low_coverage (0 disables); lineage and scores are '
                             'reported either way')
    parser.add_argument('--no_sketch', action='store_true',
                        help='skip the prefilter and score every lineage exactly')
    args = parser.parse_args()

    output = args.output
    if os.path.isdir(output):
        output = os.path.join(output, 'lineage_calls.tsv')

    # built here, before the workers start
    sketch_path = None if args.no_sketch else ensure_sketch(args.lineage_db,
                                                            args.sketch_scale)

    genomes = collect_genomes(args)
    n_workers = max(1, min(args.threads, len(genomes)))

    # split the CPUs between workers so each worker's polars pool doesn't oversubscribe;
    # set before spawning so the children pick it up when they import polars
    n_cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    os.environ['POLARS_MAX_THREADS'] = str(max(1, n_cpus // n_workers))

    rows = []
    # 'spawn' rather than fork: forking after polars has started its thread pool can deadlock
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context('spawn')) as ex:
        futures = {ex.submit(tax_one, g, args.lineage_db, sketch_path,
                             args.lineage_metric, args.kmer_size, args.top_k,
                             args.min_frac_lineage): g
                   for g in genomes}
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            rows.append(row)
            print(f'[{i}/{len(genomes)}] {row["genome"]}: {row[LINEAGE_COL]}')

    # call first, then how good the hit is, then what the lineage is made of
    front = ['genome', LINEAGE_COL, 'gtdb_tax', 'lineage_call',
             'frac_lineage_hit', 'n_hits', 'n_lineage_kmers',
             'lineage_purity', 'n_labels', 'other_labels',
             'runner_up_lineage_id', 'runner_up_gtdb_tax',
             'runner_up_frac_lineage_hit', 'runner_up_n_hits', 'runner_up_ratio']
    df = pl.DataFrame(rows, infer_schema_length=None)
    df = df.select([c for c in front if c in df.columns] +
                   [c for c in df.columns if c not in front])
    (df.sort('genome')
       .with_columns(pl.selectors.float().round(3))
       .write_csv(output, separator='\t'))
    print(f'wrote {len(rows)} calls to {output}')


if __name__ == '__main__':
    main()