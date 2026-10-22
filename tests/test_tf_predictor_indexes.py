from contextlib import contextmanager
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from docling_ibm_models.tableformer.data_management.tf_predictor import TFPredictor


@pytest.mark.parametrize("method", ["predict", "predict_dummy"])
def test_inference_guard_excludes_cpu_preparation_and_matching(monkeypatch, method):
    active = False
    calls = []

    @contextmanager
    def guard():
        nonlocal active
        assert not active
        active = True
        calls.append("enter")
        try:
            yield
        finally:
            active = False

    def prepare(image):
        assert not active
        calls.append("prepare")
        return torch.zeros(1, 3, 4, 4)

    transfer = torch.Tensor.to
    tolist = torch.Tensor.tolist

    def guarded_transfer(tensor, *args, **kwargs):
        assert active
        return transfer(tensor, *args, **kwargs)

    def guarded_materialize(tensor):
        assert active
        return tolist(tensor)

    def infer(*args):
        assert active
        calls.append("infer")
        return [2, 5, 9, 3], torch.zeros(1, 2), torch.ones(1, 4)

    def boxes(tensor):
        assert active
        calls.append("boxes")
        return tensor

    def match(*args):
        assert not active
        calls.append("match")
        return {"table_cells": [], "matches": {}, "pdf_cells": []}

    predictor = TFPredictor.__new__(TFPredictor)
    predictor._config = {"predict": {"max_steps": 32, "beam_size": 1, "bbox": True}}
    predictor._prof = False
    predictor._device = "cpu"
    predictor._inference_context = guard
    predictor._remove_padding = False
    predictor._model = Mock(predict=infer)
    predictor._prepare_image = prepare
    predictor._get_html_tags = lambda tags: ["fcel", "nl"]
    predictor._check_bbox_sync = lambda prediction: (True, [])
    predictor._cell_matcher = Mock(match_cells=match, match_cells_dummy=match)
    predictor._generate_tf_response = lambda *args: []
    predictor._generate_tf_response_dummy = lambda *args: []
    predictor._merge_tf_output = lambda *args: []
    monkeypatch.setattr(
        "docling_ibm_models.tableformer.data_management.tf_predictor.u.box_cxcywh_to_xyxy",
        boxes,
    )
    monkeypatch.setattr(torch.Tensor, "to", guarded_transfer)
    monkeypatch.setattr(torch.Tensor, "tolist", guarded_materialize)
    getattr(predictor, method)({"tokens": []}, [0, 0, 10, 10], np.zeros((4, 4, 3)), 1)
    assert calls == ["prepare", "enter", "infer", "boxes", "match"]


def _cell(start_col, start_row, col_span=1, row_span=1):
    return {
        "start_col_offset_idx": start_col,
        "start_row_offset_idx": start_row,
        "col_span": col_span,
        "row_span": row_span,
    }


def test_response_index_preserves_first_duplicate_and_missing_matches():
    cells = [
        {
            "cell_id": 4,
            "row_id": 0,
            "column_id": 2,
            "label": "ched",
            "bbox": [1, 2, 3, 4],
        },
        {
            "cell_id": 4,
            "row_id": 9,
            "column_id": 9,
            "label": "rhed",
            "bbox": [8, 8, 9, 9],
        },
    ]
    matches = {"7": [{"table_cell_id": 4}, {"table_cell_id": 999}]}
    result = TFPredictor.__new__(TFPredictor)._generate_tf_response(cells, matches)[0]
    assert result["row_ids"] == [0]
    assert result["column_ids"] == [2]
    assert result["column_header"] and not result["row_header"]
    assert result["bbox"] == {"l": 1, "t": 2, "r": 3, "b": 4}


def _issue_123_cells():
    r"""
    The table reported in docling-ibm-models#123.

    TableFormer predicts 6 columns and 7 rows. Column ID 4 is covered by a
    single empty "ched" that starts on row 1 and spans the remaining 6 rows;
    the CellMatcher drops it (and its "ucel" continuations) because it has no
    text, so no cell starts at column ID 4 anymore. The header on row 0 spans
    all 6 predicted columns.
    """
    cells = [_cell(0, 0, col_span=6)]
    for col in (0, 1, 2, 3, 5):
        cells.append(_cell(col, 1))
    for row in range(2, 7):
        for col in (0, 1, 2, 3, 5):
            cells.append(_cell(col, row))
    return cells


def test_compress_row_col_indexes_does_not_count_dropped_columns():
    cells = _issue_123_cells()

    num_cols, num_rows = TFPredictor._compress_row_col_indexes(cells)

    # Column ID 4 held only empty cells, so the table has 5 columns, not 6.
    assert num_cols == 5
    assert num_rows == 7

    header = cells[0]
    assert header["start_col_offset_idx"] == 0
    assert header["end_col_offset_idx"] == 5
    assert header["col_span"] == 5

    # No cell may point past the compressed table.
    for cell in cells:
        assert cell["end_col_offset_idx"] <= num_cols
        assert cell["end_row_offset_idx"] <= num_rows


def test_compress_row_col_indexes_does_not_count_dropped_rows():
    r"""Same defect on the row axis: row ID 2 has no surviving cell."""
    cells = [_cell(0, 0, row_span=4), _cell(1, 0), _cell(1, 1), _cell(1, 3)]

    num_cols, num_rows = TFPredictor._compress_row_col_indexes(cells)

    assert num_cols == 2
    assert num_rows == 3

    spanning = cells[0]
    assert spanning["start_row_offset_idx"] == 0
    assert spanning["end_row_offset_idx"] == 3
    assert spanning["row_span"] == 3


@pytest.mark.parametrize("col_span", [1, 2, 3])
@pytest.mark.parametrize("row_span", [1, 2, 3])
def test_compress_row_col_indexes_keeps_gap_free_tables_intact(col_span, row_span):
    r"""A table without dropped rows/columns must survive unchanged."""
    cells = [_cell(col, row) for col in range(3) for row in range(3)]
    cells[0]["col_span"] = col_span
    cells[0]["row_span"] = row_span

    num_cols, num_rows = TFPredictor._compress_row_col_indexes(cells)

    assert (num_cols, num_rows) == (3, 3)
    for cell in cells:
        assert cell["end_col_offset_idx"] == (
            cell["start_col_offset_idx"] + cell["col_span"]
        )
        assert cell["end_row_offset_idx"] == (
            cell["start_row_offset_idx"] + cell["row_span"]
        )
