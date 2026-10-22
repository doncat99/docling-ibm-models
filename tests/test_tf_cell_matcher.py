import random

from docling_ibm_models.tableformer.data_management.tf_cell_matcher import (
    CellMatcher,
    find_intersection,
)


def _brute_matches(tables, pdfs):
    matches = {}
    count = 0
    for table in tables:
        for pdf in pdfs:
            intersection = find_intersection(table["bbox"], pdf["bbox"])
            area = (pdf["bbox"][2] - pdf["bbox"][0]) * (pdf["bbox"][3] - pdf["bbox"][1])
            if intersection is None or area <= 0:
                continue
            ratio = (
                (intersection[2] - intersection[0])
                * (intersection[3] - intersection[1])
                / float(area)
            )
            if ratio > 0:
                match = {"table_cell_id": table["cell_id"], "iopdf": ratio}
                previous = matches.setdefault(pdf["id"], [])
                if match not in previous:
                    previous.append(match)
                    count += 1
    return matches, count


def test_spatial_match_preserves_order_duplicates_and_degenerate_boxes():
    rng = random.Random(347)
    matcher = CellMatcher({"predict": {"pdf_cell_iou_thres": 0.1}})
    for _ in range(500):
        tables = [
            {"cell_id": rng.randrange(5), "bbox": [rng.randrange(10) for _ in range(4)]}
            for _ in range(rng.randrange(1, 25))
        ]
        pdfs = [
            {"id": rng.randrange(5), "bbox": [rng.randrange(10) for _ in range(4)]}
            for _ in range(rng.randrange(1, 25))
        ]
        expected = _brute_matches(tables, pdfs)
        actual = matcher._intersection_over_pdf_match(tables, pdfs)
        assert actual == expected
        assert list(actual[0]) == list(expected[0])


def test_spatial_match_avoids_disjoint_row_pairs(monkeypatch):
    calls = 0

    def counted(left, right):
        nonlocal calls
        calls += 1
        return find_intersection(left, right)

    monkeypatch.setattr(
        "docling_ibm_models.tableformer.data_management.tf_cell_matcher.find_intersection",
        counted,
    )
    tables = [{"cell_id": i, "bbox": [0, i * 10, 5, i * 10 + 5]} for i in range(100)]
    pdfs = [{"id": i, "bbox": [0, i * 10, 5, i * 10 + 5]} for i in range(100)]
    matches, count = CellMatcher(
        {"predict": {"pdf_cell_iou_thres": 0.1}}
    )._intersection_over_pdf_match(tables, pdfs)
    assert count == len(matches) == 100
    assert calls <= 100


def test_spatial_match_handles_empty_inputs_and_touching_edges():
    matcher = CellMatcher({"predict": {"pdf_cell_iou_thres": 0.1}})
    tables = [{"cell_id": 1, "bbox": [0.5, 1.25, 4.5, 3.25]}]
    pdfs = [
        {"id": 8, "bbox": [0.5, 3.25, 4.5, 4.25]},
        {"id": 9, "bbox": [0.5, 2.25, 4.5, 3.25]},
    ]
    assert matcher._intersection_over_pdf_match([], pdfs) == ({}, 0)
    assert matcher._intersection_over_pdf_match(tables, []) == ({}, 0)
    assert matcher._intersection_over_pdf_match(tables, pdfs) == _brute_matches(
        tables, pdfs
    )


def test_find_intersection_overlapping():
    # Two overlapping bboxes ([x1, y1, x2, y2], y1 <= y2) share a rectangle.
    b1 = [0, 0, 10, 10]
    b2 = [5, 5, 15, 15]
    assert find_intersection(b1, b2) == [5, 5, 10, 10]


def test_find_intersection_disjoint_horizontally():
    # b2 sits entirely to the right of b1.
    assert find_intersection([0, 0, 10, 10], [20, 0, 30, 10]) is None
    # b2 sits entirely to the left of b1.
    assert find_intersection([20, 0, 30, 10], [0, 0, 10, 10]) is None


def test_find_intersection_disjoint_vertically():
    # b1 sits entirely below b2 (in bbox coordinates, b1[1] > b2[3]).
    assert find_intersection([0, 20, 10, 30], [0, 0, 10, 10]) is None
    # b2 sits entirely below b1 (b2[1] > b1[3]).  This is the case the buggy
    # ``b2[1] > b2[3]`` check missed: it never fired for a well-formed box, so
    # find_intersection returned a degenerate, y-inverted rectangle instead of
    # None.
    assert find_intersection([0, 0, 10, 10], [0, 20, 10, 30]) is None
