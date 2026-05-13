"""Tests for afp."""
from __future__ import annotations

import bz2
import gzip
import zipfile
from pathlib import Path

import pytest

import afp


FASTA_TEXT = """>seq1 first record
ACGTACGT
ACGTACGT
>seq2
GGGGCCCC
>seq3 third
AAAA
TTTT
GGGG
"""

FASTQ_TEXT = """@read1 first
ACGTACGT
+
!!!!IIII
@read2
GGGGCCCC
+read2 again
########
@read3
AAAATTTT
+
@@@@!!!!
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


def _write_gz(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    with gzip.open(p, "wt") as fh:
        fh.write(text)
    return p


def _write_bz2(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    with bz2.open(p, "wt") as fh:
        fh.write(text)
    return p


def _write_zip(tmp_path: Path, name: str, inner: str, text: str) -> Path:
    p = tmp_path / name
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr(inner, text)
    return p


# ---- compression detection ----

def test_detect_plain(tmp_path):
    p = _write(tmp_path, "a.fa", FASTA_TEXT)
    assert afp.detect_compression(p) == "none"


def test_detect_gzip(tmp_path):
    p = _write_gz(tmp_path, "a.fa.gz", FASTA_TEXT)
    assert afp.detect_compression(p) == "gzip"


def test_detect_bzip2(tmp_path):
    p = _write_bz2(tmp_path, "a.fa.bz2", FASTA_TEXT)
    assert afp.detect_compression(p) == "bzip2"


def test_detect_zip(tmp_path):
    p = _write_zip(tmp_path, "a.zip", "a.fa", FASTA_TEXT)
    assert afp.detect_compression(p) == "zip"


def test_detect_by_magic_not_extension(tmp_path):
    # File named .fa but contents are gzip — detection uses magic, not name.
    p = tmp_path / "weird.fa"
    with gzip.open(p, "wt") as fh:
        fh.write(FASTA_TEXT)
    assert afp.detect_compression(p) == "gzip"


# ---- FASTA parsing ----

def test_parse_fasta_basic(tmp_path):
    p = _write(tmp_path, "a.fa", FASTA_TEXT)
    recs = list(afp.parse_fasta(p))
    assert [r.id for r in recs] == ["seq1", "seq2", "seq3"]
    assert recs[0].desc == "first record"
    assert recs[1].desc is None
    assert recs[0].seq == "ACGTACGTACGTACGT"
    assert recs[2].seq == "AAAATTTTGGGG"


def test_parse_fasta_gz(tmp_path):
    p = _write_gz(tmp_path, "a.fa.gz", FASTA_TEXT)
    recs = list(afp.parse(p))
    assert len(recs) == 3
    assert recs[0].seq == "ACGTACGTACGTACGT"


def test_parse_fasta_bz2(tmp_path):
    p = _write_bz2(tmp_path, "a.fa.bz2", FASTA_TEXT)
    recs = list(afp.parse(p))
    assert len(recs) == 3


def test_parse_fasta_zip(tmp_path):
    p = _write_zip(tmp_path, "a.zip", "inner.fa", FASTA_TEXT)
    recs = list(afp.parse(p))
    assert len(recs) == 3


def test_parse_fasta_empty(tmp_path):
    # Empty input is not an error — it just yields zero records. Callers
    # that need to treat empty as an error check `len(records) == 0`.
    p = _write(tmp_path, "empty.fa", "")
    assert list(afp.parse(p)) == []


def test_parse_fasta_seq_before_header(tmp_path):
    p = _write(tmp_path, "bad.fa", "ACGT\n>seq1\nACGT\n")
    with pytest.raises(ValueError, match="format|sequence line"):
        list(afp.parse(p))


def test_parse_fasta_single_record_no_trailing_newline(tmp_path):
    p = _write(tmp_path, "a.fa", ">seq1\nACGT")
    recs = list(afp.parse(p))
    assert len(recs) == 1
    assert recs[0].seq == "ACGT"


# ---- FASTQ parsing ----

def test_parse_fastq_basic(tmp_path):
    p = _write(tmp_path, "a.fq", FASTQ_TEXT)
    recs = list(afp.parse_fastq(p))
    assert [r.id for r in recs] == ["read1", "read2", "read3"]
    assert recs[0].desc == "first"
    assert recs[0].seq == "ACGTACGT"
    assert recs[0].qual == "!!!!IIII"
    assert recs[1].seq == "GGGGCCCC"
    assert recs[1].qual == "########"


def test_parse_fastq_autodetect(tmp_path):
    p = _write(tmp_path, "a.fq", FASTQ_TEXT)
    recs = list(afp.parse(p))
    assert len(recs) == 3
    assert recs[0].qual is not None


def test_parse_fastq_gz(tmp_path):
    p = _write_gz(tmp_path, "a.fq.gz", FASTQ_TEXT)
    recs = list(afp.parse(p))
    assert len(recs) == 3
    assert recs[0].qual == "!!!!IIII"


def test_parse_fastq_length_mismatch(tmp_path):
    # Quality string is longer than the sequence — that's a real length
    # mismatch (vs. truncation, which we exercise separately).
    bad = "@r1\nACGT\n+\n!!!!!!\n"
    p = _write(tmp_path, "bad.fq", bad)
    with pytest.raises(ValueError, match="quality length"):
        list(afp.parse_fastq(p))


def test_parse_fastq_truncated(tmp_path):
    bad = "@r1\nACGT\n"
    p = _write(tmp_path, "trunc.fq", bad)
    with pytest.raises(ValueError, match="unexpected EOF"):
        list(afp.parse_fastq(p))


# ---- Record class ----

def test_record_mutable():
    r = afp.Record(id="a", seq="ACGT", desc="x")
    r.id = "b"
    assert r.id == "b"
    assert "AC" in r
    assert len(r) == 4
    assert list(r) == ["A", "C", "G", "T"]


def test_record_format_fasta_no_wrap():
    r = afp.Record(id="seq1", seq="ACGT" * 10, desc="d")
    assert r.format() == ">seq1 d\n" + ("ACGT" * 10) + "\n"


def test_record_format_fasta_wrap():
    r = afp.Record(id="seq1", seq="ACGT" * 4)
    out = r.format(wrap=8)
    assert out == ">seq1\nACGTACGT\nACGTACGT\n"


def test_record_format_fastq():
    r = afp.Record(id="r", seq="ACGT", qual="IIII", desc="d")
    assert r.format() == "@r d\nACGT\n+\nIIII\n"


def test_record_description_property():
    fa = afp.Record(id="a", seq="A", desc="hello")
    assert fa.description == ">a hello"
    fq = afp.Record(id="b", seq="A", qual="!", desc=None)
    assert fq.description == "@b"


# ---- write round-trip ----

def test_write_fasta_roundtrip(tmp_path):
    src = _write(tmp_path, "a.fa", FASTA_TEXT)
    recs = list(afp.parse(src))
    out = tmp_path / "out.fa"
    n = afp.write(recs, out)
    assert n == 3
    again = list(afp.parse(out))
    assert [r.id for r in again] == [r.id for r in recs]
    assert [r.seq for r in again] == [r.seq for r in recs]


def test_write_fastq_roundtrip_gz(tmp_path):
    src = _write(tmp_path, "a.fq", FASTQ_TEXT)
    recs = list(afp.parse(src))
    out = tmp_path / "out.fq.gz"
    afp.write(recs, out)
    assert afp.detect_compression(out) == "gzip"
    again = list(afp.parse(out))
    assert [r.qual for r in again] == [r.qual for r in recs]


def test_write_mixed_types_rejected(tmp_path):
    mixed = [
        afp.Record(id="a", seq="A"),
        afp.Record(id="b", seq="A", qual="!"),
    ]
    with pytest.raises(ValueError, match="mix FASTA and FASTQ"):
        afp.write(mixed, tmp_path / "out")
