"""The OpenML parquet download goes over HTTPS (openml 0.15 hardcodes HTTP)."""

import pytest

from metapilot import sources


class _Resp:
    def __init__(self, status, content=b""):
        self.status_code = status
        self.content = content


def test_parquet_download_uses_https(tmp_path, monkeypatch):
    requests = pytest.importorskip("requests")
    seen = []

    def fake_get(url, timeout):
        seen.append(url)
        return _Resp(200, b"PAR1data")

    monkeypatch.setattr(requests, "get", fake_get)
    dest = tmp_path / "sub" / "dataset_61.pq"
    sources._download_parquet_https("http://data.openml.org/datasets/0000/0061/dataset_61.pq", dest)
    assert seen == ["https://data.openml.org/datasets/0000/0061/dataset_61.pq"]
    assert dest.read_bytes() == b"PAR1data"

    # Cached file: no second request.
    sources._download_parquet_https("http://data.openml.org/datasets/0000/0061/dataset_61.pq", dest)
    assert len(seen) == 1


def test_parquet_download_missing_raises_file_not_found(tmp_path, monkeypatch):
    requests = pytest.importorskip("requests")
    monkeypatch.setattr(requests, "get", lambda url, timeout: _Resp(404))
    with pytest.raises(FileNotFoundError):
        sources._download_parquet_https("https://data.openml.org/x/y.pq", tmp_path / "y.pq")
    assert not (tmp_path / "y.pq").exists()
