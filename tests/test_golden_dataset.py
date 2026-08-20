import pytest

from app.eval.golden_dataset import GoldenDataset


def test_golden_dataset_loads():
    dataset = GoldenDataset("eval/golden_dataset.json")
    assert len(dataset.all()) > 0


def test_golden_dataset_summary():
    dataset = GoldenDataset("eval/golden_dataset.json")
    summary = dataset.summary()
    assert summary["total"] == len(dataset.all())
    assert summary["by_source"]["chromadb_info.txt"] == 8
    assert summary["multi_doc"] == 2


def test_golden_dataset_filters():
    dataset = GoldenDataset("eval/golden_dataset.json")
    assert len(dataset.get_by_difficulty("easy")) > 0
    assert len(dataset.get_by_category("general")) > 0
