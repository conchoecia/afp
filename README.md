# afp — Another Fastx Parser

[![CI](https://github.com/conchoecia/afp/actions/workflows/ci.yml/badge.svg)](https://github.com/conchoecia/afp/actions/workflows/ci.yml)
![Coverage](images/coverage-badge.svg)
[![PyPI](https://img.shields.io/pypi/v/run-afp.svg)](https://pypi.org/project/run-afp/)
[![Python versions](https://img.shields.io/pypi/pyversions/run-afp.svg)](https://pypi.org/project/run-afp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A tiny, dependency-free Python reader/writer for **FASTA** and **FASTQ** files
with transparent `gzip` / `bzip2` / `zip` / `zstandard` decompression. Standard
library only — `zstandard` is an optional extra used only for `.zst` inputs.

The whole module is a single file: [`afp.py`](afp.py).

## Install

Three ways to use it:

**Drop the file into your project:**

```sh
curl -O https://raw.githubusercontent.com/conchoecia/afp/main/afp.py
# put afp.py somewhere on your python path
```

**Or pip-install:**

```sh
pip install run-afp               # core, no extras
pip install "run-afp[zstd]"       # also read .zst-compressed files
pip install "run-afp[dev]"        # pytest + zstandard for development
```

The PyPI distribution is `run-afp` (the bare `afp` name was already taken).
The import name stays `import afp`.

**Or vendor inside another repo:** copy `afp.py` into your `dependencies/`
directory, add that directory to `sys.path`, then `import afp`.

## Quick start

```python
import afp

# Auto-detects FASTA vs FASTQ from the first byte, and gzip/bzip2/zip/zstd
# compression from the file's magic bytes (not its extension).
for rec in afp.parse("reads.fq.gz"):
    print(rec.id, len(rec.seq), rec.qual[:10])

for rec in afp.parse("genome.fa"):
    print(rec.id, rec.desc, rec.seq[:50])
```

Force a specific format if needed:

```python
for rec in afp.parse("weirdly_named_file", format="fasta"):
    ...

# Or use the explicit parsers:
afp.parse_fasta("genome.fa")
afp.parse_fastq("reads.fq")
```

## The `Record` object

```python
class Record:
    id: str             # token after '>' or '@', up to first whitespace
    seq: str            # sequence, newlines stripped
    desc: str | None    # everything after id on the header line (or None)
    qual: str | None    # quality string (FASTQ only; None for FASTA)
```

Records are mutable. You can rewrite `record.id` in place.

```python
rec.format()           # back to FASTA / FASTQ text
rec.format(wrap=80)    # FASTA only: wrap sequence at 80 columns
len(rec)               # length of seq
"ACGT" in rec          # membership on the sequence string
list(rec)              # iterate over letters
```

## Writing

```python
afp.write(records, "out.fa")          # plain
afp.write(records, "out.fa.gz")       # auto-gzip from .gz extension
afp.write(records, "out.fq.gz")       # FASTQ if the first record has `qual`
afp.write(records, "out.fa", wrap=80) # wrapped FASTA
```

Mixing FASTA and FASTQ records in a single output stream is rejected.

## Compression helpers

```python
afp.detect_compression("file")   # 'gzip' | 'bzip2' | 'zip' | 'zstd' | 'none'
afp.get_open_func("file.gz")     # returns gzip.open
```

`detect_compression` reads only the first 4 bytes — it's cheap to call.

## Why

Built for projects that want to vendor a single Python file rather than pull
in a multi-megabyte sequence toolkit, under a permissive license they can
carry through to their own code. The whole module is one ~400-line file, no
external runtime dependencies, no compiled extensions.

## License

MIT — see `LICENSE`.
