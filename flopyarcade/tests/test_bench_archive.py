"""B26: Longitudinal benchmark archive tests."""
import os
import tempfile
import pytest

from flopyarcade.bench_archive import (
    ArchiveEntry,
    append_entry,
    read_archive,
    get_score_history,
    detect_regression,
    trend_scores,
    archive_summary,
    format_archive_report,
)


@pytest.fixture
def archive_path():
    """Return a temp file path for the archive."""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    os.unlink(path)  # start with missing file
    yield path
    if os.path.exists(path):
        os.unlink(path)


def make_entry(scores, ts="2025-01-01T00:00:00Z", sha="abc123", meta=None):
    return ArchiveEntry(timestamp=ts, git_sha=sha, scores=scores, metadata=meta)


def test_append_entry_creates_file(archive_path):
    """append_entry creates the file when it does not exist."""
    assert not os.path.exists(archive_path)
    append_entry(archive_path, make_entry({"4s-c": 0.85}))
    assert os.path.exists(archive_path)
    with open(archive_path) as fh:
        assert len(fh.readlines()) == 1


def test_append_entry_appends(archive_path):
    """Two appends produce two lines."""
    append_entry(archive_path, make_entry({"4s-c": 0.80}, ts="t1"))
    append_entry(archive_path, make_entry({"4s-c": 0.90}, ts="t2"))
    with open(archive_path) as fh:
        lines = fh.readlines()
    assert len(lines) == 2


def test_read_archive_returns_correct_count(archive_path):
    """read_archive returns the exact number of entries written."""
    for i in range(5):
        append_entry(archive_path, make_entry({"4s-c": float(i)}, ts=f"t{i}"))
    entries = read_archive(archive_path)
    assert len(entries) == 5
    assert entries[0].timestamp == "t0"
    assert entries[4].scores["4s-c"] == 4.0


def test_read_archive_missing_file_returns_empty(archive_path):
    """read_archive on a non-existent file returns []."""
    assert read_archive(archive_path) == []


def test_get_score_history_filters_correctly(archive_path):
    """get_score_history returns only entries that contain the given envtype."""
    append_entry(archive_path, make_entry({"4s-c": 0.7, "6s-c": 0.5}, ts="t1"))
    append_entry(archive_path, make_entry({"4s-c": 0.8}, ts="t2"))
    append_entry(archive_path, make_entry({"6s-c": 0.6}, ts="t3"))
    hist = get_score_history(archive_path, "4s-c")
    assert len(hist) == 2
    assert hist[0]["score"] == 0.7
    assert hist[1]["score"] == 0.8
    hist6 = get_score_history(archive_path, "6s-c")
    assert len(hist6) == 2
    assert hist6[0]["score"] == 0.5
    assert hist6[1]["score"] == 0.6


def test_get_score_history_missing_envtype_returns_empty(archive_path):
    """get_score_history for an envtype not present returns []."""
    append_entry(archive_path, make_entry({"4s-c": 0.7}))
    assert get_score_history(archive_path, "99s-unknown") == []


def test_detect_regression_with_improvement(archive_path):
    """Score goes up -> no regression."""
    append_entry(archive_path, make_entry({"4s-c": 0.80}, ts="t1"))
    append_entry(archive_path, make_entry({"4s-c": 0.90}, ts="t2"))
    result = detect_regression(archive_path, "4s-c")
    assert result["latest"] == 0.90
    assert result["previous"] == 0.80
    assert result["change_pct"] > 0
    assert result["regressed"] is False


def test_detect_regression_with_regression(archive_path):
    """Score drops by more than threshold -> regressed."""
    append_entry(archive_path, make_entry({"4s-c": 0.90}, ts="t1"))
    append_entry(archive_path, make_entry({"4s-c": 0.75}, ts="t2"))
    result = detect_regression(archive_path, "4s-c")
    assert result["regressed"] is True
    assert result["change_pct"] < -5.0


def test_detect_regression_at_threshold_boundary(archive_path):
    """Score drops exactly at threshold -> NOT regressed (strict <)."""
    prev = 100.0
    latest = 95.0
    append_entry(archive_path, make_entry({"4s-c": prev}, ts="t1"))
    append_entry(archive_path, make_entry({"4s-c": latest}, ts="t2"))
    result = detect_regression(archive_path, "4s-c", threshold_pct=5.0)
    assert result["change_pct"] == pytest.approx(-5.0)
    assert result["regressed"] is False


def test_trend_scores_positive_trend(archive_path):
    """Increasing scores -> positive slope."""
    for i in range(5):
        append_entry(archive_path, make_entry({"4s-c": 0.5 + i * 0.1}, ts=f"t{i}"))
    result = trend_scores(archive_path, "4s-c", window=5)
    assert result["slope"] > 0
    assert result["n_points"] == 5
    assert result["predicted_next"] > 0.9


def test_trend_scores_negative_trend(archive_path):
    """Decreasing scores -> negative slope."""
    for i in range(5):
        append_entry(archive_path, make_entry({"4s-c": 1.0 - i * 0.1}, ts=f"t{i}"))
    result = trend_scores(archive_path, "4s-c", window=5)
    assert result["slope"] < 0
    assert result["n_points"] == 5


def test_archive_summary_structure(archive_path):
    """archive_summary returns expected keys with correct types."""
    append_entry(archive_path, make_entry({"4s-c": 0.8, "6s-c": 0.6}, ts="2025-01-01T00:00:00Z"))
    append_entry(archive_path, make_entry({"4s-c": 0.85, "6s-c": 0.62}, ts="2025-01-02T00:00:00Z"))
    s = archive_summary(archive_path)
    assert s["n_entries"] == 2
    assert s["envtypes"] == ["4s-c", "6s-c"]
    assert s["first_timestamp"] == "2025-01-01T00:00:00Z"
    assert s["last_timestamp"] == "2025-01-02T00:00:00Z"
    assert s["n_regressions"] == 0


def test_archive_summary_empty(archive_path):
    """archive_summary on missing file returns zeroed structure."""
    s = archive_summary(archive_path)
    assert s["n_entries"] == 0
    assert s["envtypes"] == []
    assert s["first_timestamp"] is None
    assert s["last_timestamp"] is None
    assert s["n_regressions"] == 0


def test_format_archive_report_contains_text(archive_path):
    """format_archive_report returns a non-empty string with key labels."""
    append_entry(archive_path, make_entry({"4s-c": 0.8}, ts="2025-01-01T00:00:00Z", sha="aaa"))
    append_entry(archive_path, make_entry({"4s-c": 0.85}, ts="2025-01-02T00:00:00Z", sha="bbb"))
    report = format_archive_report(archive_path)
    assert isinstance(report, str)
    assert "Benchmark Archive Report" in report
    assert "Entries" in report
    assert "4s-c" in report
