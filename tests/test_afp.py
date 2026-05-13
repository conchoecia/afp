"""Test suite for afp.

Scope is deliberately broad: pathological inputs, every supported
compression, line-ending variants, IUPAC ambiguity codes, NCBI-style
headers, FASTQ quirks (qualities that look like headers, qual char range),
streaming behaviour, and round-trip integrity.
"""
from __future__ import annotations

import bz2
import gzip
import io
import os
import string
import zipfile
from pathlib import Path

import pytest

import afp


# ---------------------------------------------------------------------------
# Fixture text & helpers
# ---------------------------------------------------------------------------

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

# Quality string that exercises the full printable ASCII range FASTQ permits
# (Phred 33: chars 33-126). Used to make sure we don't choke on '@', '+', '>'
# inside qualities.
FULL_QUAL_RANGE = "".join(chr(i) for i in range(33, 127))
FULL_SEQ = "A" * len(FULL_QUAL_RANGE)

# IUPAC ambiguity codes for nucleotides + the standard 20 amino acids + RNA.
IUPAC_NUC = "ACGTUNRYSWKMBDHV"
PROTEIN_20 = "ACDEFGHIKLMNPQRSTVWY"


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


def _write_bytes(tmp_path: Path, name: str, data: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(data)
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


def _zstd_available() -> bool:
    try:
        import zstandard  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Compression detection
# ---------------------------------------------------------------------------


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


def test_detect_empty_zip_central_directory(tmp_path):
    """Empty zip files have the PK\\x05\\x06 (EOCD) magic, not PK\\x03\\x04."""
    p = tmp_path / "empty.zip"
    with zipfile.ZipFile(p, "w"):
        pass
    assert afp.detect_compression(p) == "zip"


@pytest.mark.skipif(not _zstd_available(), reason="zstandard not installed")
def test_detect_zstd(tmp_path):
    import zstandard
    p = tmp_path / "a.fa.zst"
    p.write_bytes(zstandard.compress(FASTA_TEXT.encode()))
    assert afp.detect_compression(p) == "zstd"


def test_detect_by_magic_not_extension(tmp_path):
    p = tmp_path / "weird.fa"
    with gzip.open(p, "wt") as fh:
        fh.write(FASTA_TEXT)
    assert afp.detect_compression(p) == "gzip"


def test_detect_short_file(tmp_path):
    """File shorter than any magic prefix shouldn't crash."""
    p = _write_bytes(tmp_path, "tiny", b"AB")
    assert afp.detect_compression(p) == "none"


def test_detect_zero_byte_file(tmp_path):
    p = _write_bytes(tmp_path, "zero", b"")
    assert afp.detect_compression(p) == "none"


# ---------------------------------------------------------------------------
# FASTA parsing — basics
# ---------------------------------------------------------------------------


def test_parse_fasta_basic(tmp_path):
    p = _write(tmp_path, "a.fa", FASTA_TEXT)
    recs = list(afp.parse_fasta(p))
    assert [r.id for r in recs] == ["seq1", "seq2", "seq3"]
    assert recs[0].desc == "first record"
    assert recs[1].desc is None
    assert recs[0].seq == "ACGTACGTACGTACGT"
    assert recs[2].seq == "AAAATTTTGGGG"
    # All FASTA records have qual = None
    assert all(r.qual is None for r in recs)


def test_parse_fasta_gz(tmp_path):
    p = _write_gz(tmp_path, "a.fa.gz", FASTA_TEXT)
    recs = list(afp.parse(p))
    assert len(recs) == 3
    assert recs[0].seq == "ACGTACGTACGTACGT"


def test_parse_fasta_bz2(tmp_path):
    p = _write_bz2(tmp_path, "a.fa.bz2", FASTA_TEXT)
    recs = list(afp.parse(p))
    assert len(recs) == 3
    assert recs[0].seq == "ACGTACGTACGTACGT"


def test_parse_fasta_zip(tmp_path):
    p = _write_zip(tmp_path, "a.zip", "inner.fa", FASTA_TEXT)
    recs = list(afp.parse(p))
    assert len(recs) == 3


def test_parse_fasta_zip_multi_member_rejected(tmp_path):
    """Zip with two members should raise (silent-data-loss avoidance)."""
    p = tmp_path / "two.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("a.fa", FASTA_TEXT)
        zf.writestr("b.fa", FASTA_TEXT)
    with pytest.raises(ValueError, match="exactly one file"):
        list(afp.parse(p))


@pytest.mark.skipif(not _zstd_available(), reason="zstandard not installed")
def test_parse_fasta_zstd(tmp_path):
    import zstandard
    p = tmp_path / "a.fa.zst"
    p.write_bytes(zstandard.compress(FASTA_TEXT.encode()))
    recs = list(afp.parse(p))
    assert len(recs) == 3


def test_parse_fasta_empty_file(tmp_path):
    p = _write(tmp_path, "empty.fa", "")
    assert list(afp.parse(p)) == []


def test_parse_fasta_only_blank_lines(tmp_path):
    p = _write(tmp_path, "blank.fa", "\n\n\n   \n\n")
    # First non-blank line check raises on lstripped non-empty;
    # all-whitespace input results in zero records.
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


def test_parse_fasta_header_only_no_seq(tmp_path):
    """A header with no sequence after it should yield a record with empty seq."""
    p = _write(tmp_path, "a.fa", ">seq1\n>seq2\nACGT\n")
    recs = list(afp.parse(p))
    assert recs[0].id == "seq1"
    assert recs[0].seq == ""
    assert recs[1].id == "seq2"
    assert recs[1].seq == "ACGT"


def test_parse_fasta_trailing_header_only(tmp_path):
    """Final record having only a header should still yield with empty seq."""
    p = _write(tmp_path, "a.fa", ">seq1\nACGT\n>tail\n")
    recs = list(afp.parse(p))
    assert recs[1].id == "tail"
    assert recs[1].seq == ""


# ---------------------------------------------------------------------------
# FASTA parsing — line endings
# ---------------------------------------------------------------------------


def test_parse_fasta_crlf(tmp_path):
    crlf = FASTA_TEXT.replace("\n", "\r\n")
    p = _write_bytes(tmp_path, "crlf.fa", crlf.encode())
    recs = list(afp.parse(p))
    assert len(recs) == 3
    assert recs[0].seq == "ACGTACGTACGTACGT"
    # \r is fully stripped, no \r leaks into seq or desc
    assert "\r" not in recs[0].desc
    assert all("\r" not in r.seq for r in recs)


def test_parse_fasta_mixed_line_endings(tmp_path):
    mixed = ">seq1\r\nACGT\nGGGG\r\n>seq2\nTTTT\r"
    p = _write_bytes(tmp_path, "mixed.fa", mixed.encode())
    recs = list(afp.parse(p))
    assert recs[0].seq == "ACGTGGGG"
    assert recs[1].seq == "TTTT"


# ---------------------------------------------------------------------------
# FASTA parsing — header parsing variants
# ---------------------------------------------------------------------------


def test_parse_fasta_ncbi_style_header(tmp_path):
    """NCBI uses pipe-delimited identifiers like
    `>gi|12345|ref|NP_001.1| description text`. The id is the first
    whitespace-delimited token, and pipes are part of it."""
    text = ">gi|12345|ref|NP_001.1| description text\nACGTACGT\n"
    p = _write(tmp_path, "ncbi.fa", text)
    rec = next(iter(afp.parse(p)))
    assert rec.id == "gi|12345|ref|NP_001.1|"
    assert rec.desc == "description text"


def test_parse_fasta_id_only(tmp_path):
    text = ">solo\nACGT\n"
    p = _write(tmp_path, "solo.fa", text)
    rec = next(iter(afp.parse(p)))
    assert rec.id == "solo"
    assert rec.desc is None


def test_parse_fasta_tab_separated_desc(tmp_path):
    """If id and desc are tab-separated (some tools emit this), split on tab."""
    text = ">seq1\tdescription tab style\nACGT\n"
    p = _write(tmp_path, "tab.fa", text)
    rec = next(iter(afp.parse(p)))
    assert rec.id == "seq1"
    assert rec.desc == "description tab style"


def test_parse_fasta_unicode_description(tmp_path):
    """Descriptions with non-ASCII unicode should pass through verbatim."""
    text = ">seq1 α-globin (greek alpha)\nACGT\n"
    p = tmp_path / "uni.fa"
    p.write_text(text, encoding="utf-8")
    rec = next(iter(afp.parse(p)))
    assert "α" in rec.desc


def test_parse_fasta_long_description(tmp_path):
    """Very long description lines should not be truncated."""
    long_desc = "word " * 500
    text = f">seq1 {long_desc}\nACGT\n"
    p = _write(tmp_path, "long.fa", text)
    rec = next(iter(afp.parse(p)))
    assert rec.desc.startswith("word")
    assert len(rec.desc) > 2000


# ---------------------------------------------------------------------------
# FASTA parsing — sequence content
# ---------------------------------------------------------------------------


def test_parse_fasta_iupac_nucleotide_codes(tmp_path):
    text = f">seq1\n{IUPAC_NUC}\n"
    p = _write(tmp_path, "iupac.fa", text)
    rec = next(iter(afp.parse(p)))
    assert rec.seq == IUPAC_NUC


def test_parse_fasta_protein_residues(tmp_path):
    text = f">prot1\n{PROTEIN_20}\n"
    p = _write(tmp_path, "prot.fa", text)
    rec = next(iter(afp.parse(p)))
    assert rec.seq == PROTEIN_20


def test_parse_fasta_mixed_case_seq(tmp_path):
    text = ">seq1\naCgTAcGt\n"
    p = _write(tmp_path, "mixed.fa", text)
    rec = next(iter(afp.parse(p)))
    # Parser preserves case as-is; downstream is responsible for upper().
    assert rec.seq == "aCgTAcGt"


def test_parse_fasta_long_single_sequence(tmp_path):
    """100k bp wrapped at 70 columns is still one record."""
    seq = "ACGT" * 25_000
    wrapped = "\n".join(seq[i : i + 70] for i in range(0, len(seq), 70))
    text = f">long\n{wrapped}\n"
    p = _write(tmp_path, "long.fa", text)
    rec = next(iter(afp.parse(p)))
    assert rec.seq == seq
    assert len(rec.seq) == 100_000


def test_parse_fasta_many_records(tmp_path):
    """Several thousand short records, gzipped."""
    text = "".join(f">r{i}\nACGT{i}\n" for i in range(5_000))
    p = _write_gz(tmp_path, "many.fa.gz", text)
    recs = list(afp.parse(p))
    assert len(recs) == 5_000
    assert recs[1234].id == "r1234"
    assert recs[1234].seq == "ACGT1234"


def test_parse_fasta_irregular_wrap(tmp_path):
    """Wrap width that changes mid-record should still join cleanly."""
    text = ">seq1\nA\nCG\nTAC\nGTACG\nT\n"
    p = _write(tmp_path, "irr.fa", text)
    rec = next(iter(afp.parse(p)))
    assert rec.seq == "ACGTACGTACGT"


def test_parse_fasta_blank_lines_inside(tmp_path):
    """Blank lines inside a record are skipped."""
    text = ">seq1\nACGT\n\nGGGG\n\n>seq2\nTTTT\n"
    p = _write(tmp_path, "blanks.fa", text)
    recs = list(afp.parse(p))
    assert recs[0].seq == "ACGTGGGG"
    assert recs[1].seq == "TTTT"


def test_parse_fasta_duplicate_ids(tmp_path):
    """Duplicate ids are permitted at the parse layer; uniqueness is a
    caller-side concern."""
    text = ">dup\nA\n>dup\nC\n>dup\nG\n"
    p = _write(tmp_path, "dup.fa", text)
    recs = list(afp.parse(p))
    assert [r.seq for r in recs] == ["A", "C", "G"]


# ---------------------------------------------------------------------------
# FASTQ parsing — basics
# ---------------------------------------------------------------------------


def test_parse_fastq_basic(tmp_path):
    p = _write(tmp_path, "a.fq", FASTQ_TEXT)
    recs = list(afp.parse_fastq(p))
    assert [r.id for r in recs] == ["read1", "read2", "read3"]
    assert recs[0].desc == "first"
    assert recs[0].seq == "ACGTACGT"
    assert recs[0].qual == "!!!!IIII"
    assert recs[1].seq == "GGGGCCCC"
    assert recs[1].qual == "########"
    # All FASTQ records expose qual
    assert all(r.qual is not None for r in recs)


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


def test_parse_fastq_bz2(tmp_path):
    p = _write_bz2(tmp_path, "a.fq.bz2", FASTQ_TEXT)
    recs = list(afp.parse(p))
    assert len(recs) == 3


def test_parse_fastq_length_mismatch(tmp_path):
    bad = "@r1\nACGT\n+\n!!!!!!\n"
    p = _write(tmp_path, "bad.fq", bad)
    with pytest.raises(ValueError, match="quality length"):
        list(afp.parse_fastq(p))


def test_parse_fastq_truncated_after_header(tmp_path):
    p = _write(tmp_path, "trunc.fq", "@r1\nACGT\n")
    with pytest.raises(ValueError, match="unexpected EOF"):
        list(afp.parse_fastq(p))


def test_parse_fastq_truncated_qualities(tmp_path):
    p = _write(tmp_path, "trunc.fq", "@r1\nACGT\n+\n!!")
    with pytest.raises(ValueError, match="unexpected EOF"):
        list(afp.parse_fastq(p))


def test_parse_fastq_no_header(tmp_path):
    p = _write(tmp_path, "noheader.fq", "ACGT\n+\n!!!!\n")
    with pytest.raises(ValueError, match="expected FASTQ header"):
        list(afp.parse_fastq(p))


def test_parse_fastq_id_only(tmp_path):
    text = "@r1\nACGT\n+\n!!!!\n"
    p = _write(tmp_path, "id.fq", text)
    rec = next(iter(afp.parse(p)))
    assert rec.id == "r1"
    assert rec.desc is None
    assert rec.qual == "!!!!"


def test_parse_fastq_plus_with_repeated_desc(tmp_path):
    """The optional `+` line may repeat the header. Should be ignored."""
    text = "@r1 description\nACGT\n+r1 description\n!!!!\n"
    p = _write(tmp_path, "plus.fq", text)
    rec = next(iter(afp.parse(p)))
    assert rec.id == "r1"
    assert rec.desc == "description"
    assert rec.qual == "!!!!"


def test_parse_fastq_full_quality_range(tmp_path):
    """Qualities span the full Phred+33 printable range, including '@' and
    '+' characters that look like FASTQ structural markers."""
    text = f"@r1\n{FULL_SEQ}\n+\n{FULL_QUAL_RANGE}\n"
    p = _write(tmp_path, "full.fq", text)
    rec = next(iter(afp.parse(p)))
    assert rec.seq == FULL_SEQ
    assert rec.qual == FULL_QUAL_RANGE
    assert "@" in rec.qual
    assert "+" in rec.qual


def test_parse_fastq_quality_starts_with_at(tmp_path):
    """Quality strings can start with '@', which is also the header sentinel.
    The parser must not confuse it for a new record."""
    seq = "ACGTACGT"
    qual = "@@@@!!!!"  # starts with '@'
    text = f"@r1\n{seq}\n+\n{qual}\n@r2\nTTTT\n+\n!!!!\n"
    p = _write(tmp_path, "atq.fq", text)
    recs = list(afp.parse_fastq(p))
    assert len(recs) == 2
    assert recs[0].qual == qual
    assert recs[1].id == "r2"


def test_parse_fastq_many_records_gz(tmp_path):
    text = "".join(f"@r{i}\nACGTACGT\n+\n!!!!!!!!\n" for i in range(5_000))
    p = _write_gz(tmp_path, "many.fq.gz", text)
    recs = list(afp.parse(p))
    assert len(recs) == 5_000
    assert recs[2500].id == "r2500"


# ---------------------------------------------------------------------------
# Gzip variants: multi-member streams, concatenated files
# ---------------------------------------------------------------------------


def test_parse_concatenated_gzip(tmp_path):
    """gzip supports concatenation of independent gzip streams. cat a.gz b.gz
    is also a valid gzip file. Make sure we read records across the boundary."""
    p = tmp_path / "concat.fa.gz"
    body_a = ">a\nACGT\n"
    body_b = ">b\nGGGG\n"
    with open(p, "wb") as out:
        out.write(gzip.compress(body_a.encode()))
        out.write(gzip.compress(body_b.encode()))
    recs = list(afp.parse(p))
    assert [r.id for r in recs] == ["a", "b"]


# ---------------------------------------------------------------------------
# Path-as-str / pathlib.Path symmetry
# ---------------------------------------------------------------------------


def test_parse_path_object(tmp_path):
    p: Path = _write(tmp_path, "a.fa", FASTA_TEXT)
    assert isinstance(p, Path)
    recs = list(afp.parse(p))
    assert len(recs) == 3


def test_parse_path_string(tmp_path):
    p = _write(tmp_path, "a.fa", FASTA_TEXT)
    recs = list(afp.parse(str(p)))
    assert len(recs) == 3


# ---------------------------------------------------------------------------
# Iterator semantics & streaming
# ---------------------------------------------------------------------------


def test_parse_returns_generator(tmp_path):
    p = _write(tmp_path, "a.fa", FASTA_TEXT)
    it = afp.parse(p)
    # Generators have __iter__ and __next__, but no __len__
    assert hasattr(it, "__iter__")
    assert hasattr(it, "__next__")
    assert not hasattr(it, "__len__")


def test_parse_is_streaming(tmp_path):
    """Parsing 100 MB of FASTA should not require 100 MB of memory."""
    p = tmp_path / "big.fa.gz"
    with gzip.open(p, "wt") as fh:
        for i in range(50_000):
            fh.write(f">r{i}\n{'A' * 200}\n")
    # Just pull the first record; do not exhaust the generator.
    it = afp.parse(p)
    first = next(it)
    assert first.id == "r0"
    assert first.seq == "A" * 200


def test_parse_partial_consumption_no_error(tmp_path):
    p = _write(tmp_path, "a.fa", FASTA_TEXT)
    it = afp.parse(p)
    next(it)  # consume one
    # Letting the generator go out of scope should close cleanly.
    del it


# ---------------------------------------------------------------------------
# Record class
# ---------------------------------------------------------------------------


def test_record_mutable():
    r = afp.Record(id="a", seq="ACGT", desc="x")
    r.id = "b"
    assert r.id == "b"
    assert "AC" in r
    assert len(r) == 4
    assert list(r) == ["A", "C", "G", "T"]


def test_record_attribute_assignment_supports_record_name(tmp_path):
    """odp's odp_rbh_to_alignments path does `record.name = record.id`. Our
    Record is a plain dataclass, so arbitrary attribute assignment is fine."""
    p = _write(tmp_path, "a.fa", ">seq1\nACGT\n")
    rec = next(iter(afp.parse(p)))
    rec.name = rec.id
    rec.id = "renamed"
    assert rec.name == "seq1"
    assert rec.id == "renamed"


def test_record_format_fasta_no_wrap():
    r = afp.Record(id="seq1", seq="ACGT" * 10, desc="d")
    assert r.format() == ">seq1 d\n" + ("ACGT" * 10) + "\n"


def test_record_format_fasta_wrap():
    r = afp.Record(id="seq1", seq="ACGT" * 4)
    out = r.format(wrap=8)
    assert out == ">seq1\nACGTACGT\nACGTACGT\n"


def test_record_format_fasta_wrap_uneven():
    r = afp.Record(id="s", seq="ACGTACGTAC")  # 10 chars, wrap 4 = 4/4/2
    out = r.format(wrap=4)
    assert out == ">s\nACGT\nACGT\nAC\n"


def test_record_format_fastq():
    r = afp.Record(id="r", seq="ACGT", qual="IIII", desc="d")
    assert r.format() == "@r d\nACGT\n+\nIIII\n"


def test_record_description_property_fasta():
    fa = afp.Record(id="a", seq="A", desc="hello")
    assert fa.description == ">a hello"
    fa_no_desc = afp.Record(id="a", seq="A")
    assert fa_no_desc.description == ">a"


def test_record_description_property_fastq():
    fq = afp.Record(id="b", seq="A", qual="!", desc=None)
    assert fq.description == "@b"
    fq_with_desc = afp.Record(id="b", seq="A", qual="!", desc="hello")
    assert fq_with_desc.description == "@b hello"


def test_record_iter_membership_len():
    r = afp.Record(id="a", seq="ACGTAC")
    assert len(r) == 6
    assert "CG" in r
    assert "XX" not in r
    assert list(r) == ["A", "C", "G", "T", "A", "C"]


def test_record_dataclass_equality():
    """@dataclass gives us __eq__ for free, which makes round-trip assertions
    a single == rather than per-attribute checks."""
    a = afp.Record(id="x", seq="ACGT", desc="y")
    b = afp.Record(id="x", seq="ACGT", desc="y")
    c = afp.Record(id="x", seq="ACGT", desc="z")
    assert a == b
    assert a != c


# ---------------------------------------------------------------------------
# Round-trip integrity (parse → write → parse)
# ---------------------------------------------------------------------------


def test_roundtrip_fasta(tmp_path):
    src = _write(tmp_path, "a.fa", FASTA_TEXT)
    recs = list(afp.parse(src))
    out = tmp_path / "out.fa"
    n = afp.write(recs, out)
    assert n == 3
    again = list(afp.parse(out))
    assert again == recs


def test_roundtrip_fastq_gz(tmp_path):
    src = _write(tmp_path, "a.fq", FASTQ_TEXT)
    recs = list(afp.parse(src))
    out = tmp_path / "out.fq.gz"
    n = afp.write(recs, out)
    assert n == 3
    assert afp.detect_compression(out) == "gzip"
    again = list(afp.parse(out))
    assert again == recs


def test_roundtrip_fasta_bz2(tmp_path):
    src = _write(tmp_path, "a.fa", FASTA_TEXT)
    recs = list(afp.parse(src))
    out = tmp_path / "out.fa.bz2"
    afp.write(recs, out)
    assert afp.detect_compression(out) == "bzip2"
    again = list(afp.parse(out))
    assert again == recs


def test_roundtrip_fasta_wrap_then_parse(tmp_path):
    """Writing wrapped FASTA, then parsing it, should reconstruct the same
    single-string sequences (wrap is a serialization detail, not data)."""
    src = _write(tmp_path, "a.fa", FASTA_TEXT)
    recs = list(afp.parse(src))
    out = tmp_path / "wrapped.fa"
    afp.write(recs, out, wrap=10)
    again = list(afp.parse(out))
    assert [r.seq for r in again] == [r.seq for r in recs]


def test_roundtrip_explicit_compression_kwarg(tmp_path):
    """compression='gzip' overrides extension-based detection (file without
    .gz suffix still ends up gzip-compressed)."""
    src = _write(tmp_path, "a.fa", FASTA_TEXT)
    recs = list(afp.parse(src))
    out = tmp_path / "out.bin"
    afp.write(recs, out, compression="gzip")
    assert afp.detect_compression(out) == "gzip"
    assert list(afp.parse(out)) == recs


def test_write_count_returned(tmp_path):
    recs = [afp.Record(id=f"r{i}", seq="ACGT") for i in range(7)]
    n = afp.write(recs, tmp_path / "x.fa")
    assert n == 7


def test_write_mixed_types_rejected(tmp_path):
    mixed = [
        afp.Record(id="a", seq="A"),
        afp.Record(id="b", seq="A", qual="!"),
    ]
    with pytest.raises(ValueError, match="mix FASTA and FASTQ"):
        afp.write(mixed, tmp_path / "out")


def test_write_unknown_compression_rejected(tmp_path):
    recs = [afp.Record(id="a", seq="A")]
    with pytest.raises(ValueError, match="unsupported output compression"):
        afp.write(recs, tmp_path / "x.fa", compression="lzma")


def test_write_empty_iterable(tmp_path):
    """Writing zero records should produce an empty file."""
    out = tmp_path / "empty.fa"
    n = afp.write([], out)
    assert n == 0
    assert out.read_text() == ""


# ---------------------------------------------------------------------------
# get_open_func direct usage
# ---------------------------------------------------------------------------


def test_get_open_func_plain(tmp_path):
    p = _write(tmp_path, "a.fa", FASTA_TEXT)
    opener = afp.get_open_func(p)
    assert opener is open
    with opener(p, "rt") as fh:
        assert fh.read().startswith(">seq1")


def test_get_open_func_gzip(tmp_path):
    p = _write_gz(tmp_path, "a.fa.gz", FASTA_TEXT)
    opener = afp.get_open_func(p)
    with opener(p, "rt") as fh:
        assert fh.read().startswith(">seq1")


def test_get_open_func_bzip2(tmp_path):
    p = _write_bz2(tmp_path, "a.fa.bz2", FASTA_TEXT)
    opener = afp.get_open_func(p)
    with opener(p, "rt") as fh:
        assert fh.read().startswith(">seq1")


def test_get_open_func_zip(tmp_path):
    p = _write_zip(tmp_path, "a.zip", "a.fa", FASTA_TEXT)
    opener = afp.get_open_func(p)
    with opener(p, "rt") as fh:
        assert fh.read().startswith(">seq1")


# ---------------------------------------------------------------------------
# Format override + autodetect edge cases
# ---------------------------------------------------------------------------


def test_parse_format_override_fasta(tmp_path):
    """A FASTQ file forced through fasta parser produces malformed records,
    but should not crash on the type override path itself."""
    text = ">seq1\nACGT\n"
    p = _write(tmp_path, "a.fa", text)
    recs = list(afp.parse(p, format="fasta"))
    assert len(recs) == 1


def test_parse_format_override_unsupported(tmp_path):
    p = _write(tmp_path, "a.fa", ">seq1\nACGT\n")
    with pytest.raises(ValueError, match="unsupported format"):
        list(afp.parse(p, format="genbank"))


def test_parse_autodetect_garbage_input(tmp_path):
    p = _write(tmp_path, "junk.fa", "not a fasta or fastq file\nlol\n")
    with pytest.raises(ValueError, match="detect FASTA/FASTQ"):
        list(afp.parse(p))


# ---------------------------------------------------------------------------
# Public API surface (defensive: make sure nothing leaks)
# ---------------------------------------------------------------------------


def test_public_api_surface_matches_all():
    """Everything declared in `__all__` must exist on the module."""
    for name in afp.__all__:
        assert hasattr(afp, name), f"afp.__all__ lists {name!r} but it's missing"


def test_version_string_format():
    v = afp.__version__
    parts = v.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts), f"non-numeric component in {v!r}"
