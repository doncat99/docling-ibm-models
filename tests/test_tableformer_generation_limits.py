import pytest
import torch
from torch import nn

from docling_ibm_models.tableformer.models.table04_rs.bbox_decoder_rs import BBoxDecoder
from docling_ibm_models.tableformer.models.table04_rs.tablemodel04_rs import (
    TableModel04_rs,
    _TagDraft,
)
from docling_ibm_models.tableformer.models.table04_rs.transformer_rs import (
    PositionalEncoding,
    Tag_Transformer,
)


class ScriptedTags(nn.Module):
    """Control EOS timing while exercising the real prediction loop."""

    def __init__(self, eos_step: int | None):
        super().__init__()
        self.eos_step = eos_step
        self.calls = 0

    def forward(self, hidden):
        self.calls += 1
        logits = hidden.new_full((1, 13), -100)
        logits[0, 3 if self.calls == self.eos_step else 5] = 100
        return logits


def model_with_eos(eos_step: int | None) -> TableModel04_rs:
    model = TableModel04_rs.__new__(TableModel04_rs)
    nn.Module.__init__(model)
    model._prof = False
    model._device = "cpu"
    model._max_pred_len = 2
    model._bbox = True
    model._init_data = {
        "word_map": {
            "word_map_tag": dict(
                zip(
                    [
                        "<pad>",
                        "<unk>",
                        "<start>",
                        "<end>",
                        "ecel",
                        "fcel",
                        "lcel",
                        "ucel",
                        "xcel",
                        "nl",
                        "ched",
                        "rhed",
                        "srow",
                    ],
                    range(13),
                )
            )
        }
    }
    model._encoder = nn.Identity()
    model._tag_transformer = Tag_Transformer("cpu", 13, [], 512, 1, 1, 2)
    model._tag_transformer._fc = ScriptedTags(eos_step)
    model._bbox_decoder = BBoxDecoder("cpu", 512, 256, 512, 512, 2)
    return model.eval()


@torch.no_grad()
def test_call_budget_allows_eos_beyond_constructor_budget():
    model = model_with_eos(3)
    tags, _, _ = model.predict(torch.zeros(1, 2, 2, 256), max_steps=4, k=1)
    assert tags == [2, 5, 5, 3]


@torch.no_grad()
def test_exhausted_budget_never_returns_partial_table_as_success():
    model = model_with_eos(None)
    with pytest.raises(RuntimeError, match="without.*end") as error:
        model.predict(torch.zeros(1, 2, 2, 256), max_steps=2, k=1)
    assert error.value.max_steps == 2
    assert error.value.generated_steps == 2


@torch.no_grad()
def test_eos_on_last_allowed_step_is_complete():
    tags, _, _ = model_with_eos(2).predict(torch.zeros(1, 2, 2, 256), max_steps=2, k=1)
    assert tags == [2, 5, 3]


def test_positions_extend_beyond_checkpoint_capacity_without_changing_prefix():
    encoding = PositionalEncoding(16, dropout=0).eval()
    prefix = encoding.pe.clone()
    result = encoding(torch.zeros(1082, 1, 16))
    torch.testing.assert_close(result[:1024], prefix, rtol=0, atol=0)
    assert result.shape == (1082, 1, 16)
    assert torch.isfinite(result).all()
    assert not torch.equal(result[1024], result[0])


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_budget_is_rejected(budget):
    with pytest.raises(ValueError, match="positive integer"):
        model_with_eos(1).predict(torch.zeros(1, 2, 2, 256), budget, k=1)


def test_incremental_positions_match_full_prefix():
    encoding = PositionalEncoding(16, dropout=0).eval()
    full = encoding(torch.zeros(1082, 1, 16))
    for position in [0, 1023, 1024, 1081]:
        torch.testing.assert_close(
            encoding(torch.zeros(1, 1, 16), offset=position),
            full[position : position + 1],
            rtol=0,
            atol=0,
        )


@torch.no_grad()
def test_verified_block_matches_each_serial_greedy_step():
    torch.manual_seed(19)
    transformer = Tag_Transformer("cpu", 13, [], 512, 1, 2, 2).eval()
    memory = torch.randn(4, 1, 512)
    tokens = torch.randn(8, 1, 512)
    states = transformer._decoder.prepare_cache(memory, len(tokens))
    expected = []
    for position in range(len(tokens)):
        expected.append(
            transformer._decoder.forward_step(
                tokens[position : position + 1], position, states
            )
        )
    blocked = transformer._decoder.prepare_cache(memory, len(tokens))
    actual = transformer._decoder.forward_block(tokens, 0, blocked)
    torch.testing.assert_close(actual, torch.cat(expected), rtol=1e-5, atol=1e-6)


@torch.no_grad()
def test_verified_block_reuses_one_additive_causal_mask(monkeypatch):
    torch.manual_seed(29)
    transformer = Tag_Transformer("cpu", 13, [], 512, 1, 2, 2).eval()
    decoder = transformer._decoder
    memory = torch.randn(4, 1, 512)
    tokens = torch.randn(8, 1, 512)
    serial = decoder.prepare_cache(memory, len(tokens))
    expected = torch.cat(
        [decoder.forward_step(tokens[i : i + 1], i, serial) for i in range(8)]
    )
    states = decoder.prepare_cache(memory, len(tokens))
    for position in range(3):
        decoder.forward_step(tokens[position : position + 1], position, states)
    masks = []
    attention = torch.nn.functional.scaled_dot_product_attention

    def capture_mask(*args, **kwargs):
        if kwargs.get("attn_mask") is not None:
            masks.append(kwargs["attn_mask"])
        return attention(*args, **kwargs)

    monkeypatch.setattr(
        torch.nn.functional, "scaled_dot_product_attention", capture_mask
    )
    actual = decoder.forward_block(tokens[3:], 3, states)
    torch.testing.assert_close(actual, expected[3:], rtol=1e-5, atol=1e-6)
    assert len(masks) == len(decoder.layers)
    assert all(mask is masks[0] for mask in masks)
    assert masks[0].dtype == tokens.dtype
    for row in range(5):
        assert torch.all(masks[0][row, : row + 4] == 0)
        assert torch.all(masks[0][row, row + 4 :] == -torch.inf)


def test_tableformer_drafts_the_longest_repeated_tag_suffix():
    draft = _TagDraft(1, 5, row_end=9)
    for tag in [2, 3, 4, 2, 3, 4]:
        draft.append(tag)
    assert draft.propose(3) == [2, 3, 4]


def full_history_draft(history, count, row_end):
    """Search accepted history directly, independently of incremental indexes."""
    positions = [0]
    for tag in history[1:]:
        positions.append(0 if tag == row_end else positions[-1] + 1)
    position = positions[-1]
    proposed = []
    working = history.copy()
    for _ in range(count):
        found = None
        for same_position in (True, False):
            for width in range(min(8, len(working)), 0, -1):
                suffix = working[-width:]
                for index in range(len(history) - width - 1, -1, -1):
                    if same_position and positions[index + width - 1] != position:
                        continue
                    if history[index : index + width] == suffix:
                        found = history[index + width]
                        break
                if found is not None:
                    break
            if found is not None:
                break
        proposed.append(5 if found is None else found)
        working.append(proposed[-1])
        position = 0 if proposed[-1] == row_end else position + 1
    return proposed


def test_incremental_draft_index_matches_full_history_after_every_accept():
    history = [2]
    draft = _TagDraft(2, 5, row_end=9)
    generator = torch.Generator().manual_seed(19)
    # Repeated rows plus irregular spans exercise longest-suffix replacement.
    stream = [5, 5, 9] * 20 + torch.randint(4, 13, (80,), generator=generator).tolist()
    for tag in stream:
        history.append(tag)
        draft.append(tag)
        assert draft.propose(15) == full_history_draft(history, 15, row_end=9)


@pytest.mark.parametrize("row_end", [9, 77])
def test_draft_distinguishes_same_suffix_at_different_row_positions(row_end):
    draft = _TagDraft(2, 5, row_end=row_end, max_context=1)
    for tag in [5, 6, row_end, 5, 5, 7, row_end, 5]:
        draft.append(tag)
    before = draft.history.copy()
    # At row offset 1, the latest accepted continuation of [5] is 5, not 7.
    assert draft.propose(4) == [5, 7, row_end, 5]
    assert draft.propose(4) == [5, 7, row_end, 5]
    assert draft.history == before
    draft.append(6)
    assert draft.history == [*before, 6]


@torch.no_grad()
@pytest.mark.parametrize("seed", [7, 19, 31])
@pytest.mark.parametrize("block_size", [16, 32, 64])
def test_rejected_proposals_preserve_greedy_tags_and_hidden_states(seed, block_size):
    torch.manual_seed(seed)
    model = model_with_eos(None)
    model._tag_transformer._fc = nn.Linear(512, 13)
    memory = torch.randn(4, 1, 512)
    vocab = model._init_data["word_map"]["word_map_tag"]
    serial = list(model._verified_tag_steps(memory, 48, vocab, block_size=1))
    speculative = list(
        model._verified_tag_steps(memory, 48, vocab, block_size=block_size)
    )
    assert [tag for _, tag in speculative] == [tag for _, tag in serial]
    torch.testing.assert_close(
        torch.cat([state for state, _ in speculative]),
        torch.cat([state for state, _ in serial]),
        rtol=1e-5,
        atol=1e-6,
    )


def test_speculative_tag_correction_matches_tableformer_state_rules():
    word_map = {"xcel": 8, "lcel": 6, "ucel": 7, "fcel": 5}
    correct = TableModel04_rs._correct_tag
    assert correct(8, 0, 0, word_map) == 6
    assert correct(8, 0, 1, word_map) == 8
    assert correct(6, 7, 0, word_map) == 5


@torch.no_grad()
def test_kv_cache_matches_decoder_and_is_isolated_per_table():
    torch.manual_seed(7)
    transformer = Tag_Transformer("cpu", 13, [], 512, 1, 2, 2).eval()
    memory = torch.randn(4, 1, 512)
    tokens = torch.randn(7, 1, 512)
    states = transformer._decoder.prepare_cache(memory, len(tokens))
    original_cache = None
    for position in range(len(tokens)):
        reference, original_cache = transformer._decoder(
            tokens[: position + 1], memory, original_cache
        )
        actual = transformer._decoder.forward_step(
            tokens[position : position + 1], position, states
        )
        torch.testing.assert_close(actual, reference[-1:], rtol=1e-5, atol=1e-6)
    other = transformer._decoder.prepare_cache(memory, len(tokens))
    first = transformer._decoder.forward_step(tokens[:1], 0, other)
    reference, _ = transformer._decoder(tokens[:1], memory)
    torch.testing.assert_close(first, reference[-1:], rtol=1e-5, atol=1e-6)


@torch.no_grad()
@pytest.mark.parametrize("count", [0, 1, 33])
def test_batched_boxes_match_individual_cells(count):
    torch.manual_seed(11)
    decoder = BBoxDecoder("cpu", 512, 256, 512, 512, 2).eval()
    image = torch.randn(1, 2, 2, 256)
    tags = [torch.randn(1, 512) for _ in range(count)]
    classes, boxes = decoder.inference(image, tags)
    if not count:
        assert classes.numel() == boxes.numel() == 0
        return
    encoded = (
        decoder._input_filter(image.permute(0, 3, 1, 2))
        .permute(0, 2, 3, 1)
        .reshape(1, -1, 512)
    )
    expected_classes, expected_boxes = [], []
    for tag in tags:
        hidden = decoder._init_hidden_state(encoded, 1)
        attended, _ = decoder._attention(encoded, tag, hidden)
        hidden = decoder._sigmoid(decoder._f_beta(hidden)) * attended * hidden
        expected_classes.append(decoder._class_embed(hidden))
        expected_boxes.append(decoder._bbox_embed(hidden).sigmoid())
    torch.testing.assert_close(
        classes, torch.cat(expected_classes), rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(boxes, torch.cat(expected_boxes), rtol=1e-5, atol=1e-6)
