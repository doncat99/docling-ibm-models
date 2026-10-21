import logging
from collections.abc import Iterator

import torch
import torch.nn as nn

import docling_ibm_models.tableformer.settings as s
from docling_ibm_models.tableformer.models.common.base_model import BaseModel
from docling_ibm_models.tableformer.models.table04_rs.bbox_decoder_rs import BBoxDecoder
from docling_ibm_models.tableformer.models.table04_rs.encoder04_rs import Encoder04
from docling_ibm_models.tableformer.models.table04_rs.transformer_rs import (
    Tag_Transformer,
)
from docling_ibm_models.tableformer.utils.app_profiler import AggProfiler

LOG_LEVEL = logging.WARN
# LOG_LEVEL = logging.INFO
# LOG_LEVEL = logging.DEBUG


class _TagDraft:
    """Index accepted tags once; rejected proposals never enter the index."""

    def __init__(self, start, default_tag, max_context=16):
        self.history = [start]
        self.default_tag = default_tag
        self.patterns = [{} for _ in range(max_context + 1)]

    def append(self, tag):
        for width in range(1, min(len(self.patterns) - 1, len(self.history)) + 1):
            self.patterns[width][tuple(self.history[-width:])] = tag
        self.history.append(tag)

    def propose(self, count):
        working = self.history[-(len(self.patterns) - 1) :]
        proposed = []
        for _ in range(count):
            tag = self.default_tag
            for width in range(min(len(self.patterns) - 1, len(working)), 0, -1):
                candidate = self.patterns[width].get(tuple(working[-width:]))
                if candidate is not None:
                    tag = candidate
                    break
            proposed.append(tag)
            working.append(tag)
        return proposed


class TableDecodingLimitError(RuntimeError):
    """A bounded decode must never publish a table without its end token."""

    def __init__(self, max_steps, generated_steps):
        self.max_steps = max_steps
        self.generated_steps = generated_steps
        super().__init__(
            f"Table decoding exhausted {max_steps} steps without an end token"
        )


class TableModel04_rs(BaseModel, nn.Module):
    r"""
    TableNet04Model encoder, dual-decoder model with OTSL+ support
    """

    def __init__(self, config, init_data, device):
        super(TableModel04_rs, self).__init__(config, init_data, device)

        self._prof = config["predict"].get("profiling", False)
        self._device = device
        # Extract the word_map from the init_data
        word_map = init_data["word_map"]

        # Encoder
        self._enc_image_size = config["model"]["enc_image_size"]
        self._encoder_dim = config["model"]["hidden_dim"]
        self._encoder = Encoder04(self._enc_image_size, self._encoder_dim).to(device)

        tag_vocab_size = len(word_map["word_map_tag"])

        td_encode = []
        for t in ["ecel", "fcel", "ched", "rhed", "srow"]:
            if t in word_map["word_map_tag"]:
                td_encode.append(word_map["word_map_tag"][t])
        self._log().debug("td_encode length: {}".format(len(td_encode)))
        self._log().debug("td_encode: {}".format(td_encode))

        self._tag_attention_dim = config["model"]["tag_attention_dim"]
        self._tag_embed_dim = config["model"]["tag_embed_dim"]
        self._tag_decoder_dim = config["model"]["tag_decoder_dim"]
        self._decoder_dim = config["model"]["hidden_dim"]
        self._dropout = config["model"]["dropout"]

        self._bbox = config["train"]["bbox"]
        self._bbox_attention_dim = config["model"]["bbox_attention_dim"]
        self._bbox_embed_dim = config["model"]["bbox_embed_dim"]
        self._bbox_decoder_dim = config["model"]["hidden_dim"]

        self._enc_layers = config["model"]["enc_layers"]
        self._dec_layers = config["model"]["dec_layers"]
        self._n_heads = config["model"]["nheads"]

        self._num_classes = config["model"]["bbox_classes"]
        self._enc_image_size = config["model"]["enc_image_size"]

        self._tag_transformer = Tag_Transformer(
            device,
            tag_vocab_size,
            td_encode,
            self._decoder_dim,
            self._enc_layers,
            self._dec_layers,
            self._enc_image_size,
            n_heads=self._n_heads,
        ).to(device)

        self._bbox_decoder = BBoxDecoder(
            device,
            self._bbox_attention_dim,
            self._bbox_embed_dim,
            self._tag_decoder_dim,
            self._bbox_decoder_dim,
            self._num_classes,
            self._encoder_dim,
            self._dropout,
        ).to(device)

    _logger = None

    def _log(self):
        # Setup a custom logger, built once per instance
        if self._logger is None:
            self._logger = s.get_custom_logger(self.__class__.__name__, LOG_LEVEL)
        return self._logger

    def mergebboxes(self, bbox1, bbox2):
        new_w = (bbox2[0] + bbox2[2] / 2) - (bbox1[0] - bbox1[2] / 2)
        new_h = (bbox2[1] + bbox2[3] / 2) - (bbox1[1] - bbox1[3] / 2)

        new_left = bbox1[0] - bbox1[2] / 2
        new_top = torch.minimum(bbox2[1] - bbox2[3] / 2, bbox1[1] - bbox1[3] / 2)

        new_cx = new_left + new_w / 2
        new_cy = new_top + new_h / 2

        bboxm = torch.stack([new_cx, new_cy, new_w, new_h])
        return bboxm

    @staticmethod
    def _correct_tag(tag, previous_tag, line_num, word_map):
        if line_num == 0 and tag == word_map["xcel"]:
            tag = word_map["lcel"]
        if previous_tag == word_map["ucel"] and tag == word_map["lcel"]:
            tag = word_map["fcel"]
        return tag

    def _verified_tag_steps(
        self, encoder_out, max_steps, word_map, line_num=0, block_size=16
    ) -> Iterator[tuple[torch.Tensor, int]]:
        transformer = self._tag_transformer
        states = transformer._decoder.prepare_cache(encoder_out, max_steps)
        draft = _TagDraft(word_map["<start>"], word_map["fcel"])
        history = draft.history
        while len(history) - 1 < max_steps:
            position = len(history) - 1
            width = min(block_size, max_steps - position)
            proposals = draft.propose(width - 1)
            inputs = torch.tensor(
                [[history[-1], *proposals]], device=self._device
            ).transpose(0, 1)
            embeddings = transformer._positional_encoding(
                transformer._embedding(inputs), offset=position
            )
            decoded = (
                transformer._decoder.forward_step(embeddings, position, states)
                if width == 1
                else transformer._decoder.forward_block(embeddings, position, states)
            )
            predicted = transformer._fc(decoded[:, 0]).argmax(-1).cpu().tolist()
            for index, raw_tag in enumerate(predicted):
                tag = self._correct_tag(raw_tag, history[-1], line_num, word_map)
                draft.append(tag)
                yield decoded[index : index + 1], raw_tag
                if tag == word_map["<end>"]:
                    return
                if index < len(proposals) and tag != proposals[index]:
                    break

    def predict(self, imgs, max_steps, k, return_attention=False):
        r"""
        Inference.
        The input image must be preprocessed and transformed.

        Parameters
        ----------
        img : tensor FloatTensor - torch.Size([1, 3, 448, 448])
            Input image for the inference

        Returns
        -------
        seq : list
            Predictions for the tags as indices over the word_map
        outputs_class : tensor(x, 3)
            Classes of predicted bboxes. x is the number of bboxes. There are 3 bbox classes

        outputs_coord : tensor(x, 4)
            Coords of predicted bboxes. x is the number of bboxes. Each bbox is in [cxcywh] format
        """
        if (
            not isinstance(max_steps, int)
            or isinstance(max_steps, bool)
            or max_steps < 1
        ):
            raise ValueError("max_steps must be a positive integer")
        AggProfiler().begin("predict_total", self._prof)

        # Invoke encoder
        self._tag_transformer.eval()
        enc_out = self._encoder(imgs)
        AggProfiler().end("model_encoder", self._prof)

        word_map = self._init_data["word_map"]["word_map_tag"]
        n_heads = self._tag_transformer._n_heads
        # [1, 28, 28, 512]
        encoder_out = self._tag_transformer._input_filter(
            enc_out.permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1)

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        enc_inputs = encoder_out.view(batch_size, -1, encoder_dim).to(self._device)
        enc_inputs = enc_inputs.permute(1, 0, 2)
        positions = enc_inputs.shape[0]

        encoder_mask = torch.zeros(
            (batch_size * n_heads, positions, positions), device=self._device
        ) == torch.ones(
            (batch_size * n_heads, positions, positions), device=self._device
        )

        # Invoking tag transformer encoder before the loop to save time
        AggProfiler().begin("model_tag_transformer_encoder", self._prof)
        encoder_out = self._tag_transformer._encoder(enc_inputs, mask=encoder_mask)
        AggProfiler().end("model_tag_transformer_encoder", self._prof)

        decoded_tags = [word_map["<start>"]]
        output_tags = []
        verified = self._verified_tag_steps(encoder_out, max_steps, word_map)
        tag_H_buf = []

        skip_next_tag = True
        prev_tag_ucel = False
        line_num = 0

        # Populate bboxes_to_merge, indexes of first lcel, and last cell in a span
        first_lcel = True
        bboxes_to_merge = {}
        cur_bbox_ind = -1
        bbox_ind = 0

        # i = 0
        while len(output_tags) < max_steps:
            AggProfiler().begin("model_tag_transformer_decoder", self._prof)
            decoded, new_tag = next(verified)
            AggProfiler().end("model_tag_transformer_decoder", self._prof)

            # STRUCTURE ERROR CORRECTION
            # Correction for first line xcel...
            new_tag = self._correct_tag(
                new_tag,
                word_map["ucel"] if prev_tag_ucel else decoded_tags[-1],
                line_num,
                word_map,
            )

            # End of generation
            if new_tag == word_map["<end>"]:
                output_tags.append(new_tag)
                decoded_tags.append(new_tag)
                break
            output_tags.append(new_tag)
            decoded_tags.append(new_tag)

            # BBOX PREDICTION

            # MAKE SURE TO SYNC NUMBER OF CELLS WITH NUMBER OF BBOXes
            if not skip_next_tag:
                if new_tag in [
                    word_map["fcel"],
                    word_map["ecel"],
                    word_map["ched"],
                    word_map["rhed"],
                    word_map["srow"],
                    word_map["nl"],
                    word_map["ucel"],
                ]:
                    # GENERATE BBOX HERE TOO (All other cases)...
                    tag_H_buf.append(decoded[-1, :, :])
                    if first_lcel is not True:
                        # Mark end index for horizontal cell bbox merge
                        bboxes_to_merge[cur_bbox_ind] = bbox_ind
                    bbox_ind += 1

            # Treat horisontal span bboxes...
            if new_tag != word_map["lcel"]:
                first_lcel = True
            else:
                if first_lcel:
                    # GENERATE BBOX HERE (Beginning of horisontal span)...
                    tag_H_buf.append(decoded[-1, :, :])
                    first_lcel = False
                    # Mark start index for cell bbox merge
                    cur_bbox_ind = bbox_ind
                    bboxes_to_merge[cur_bbox_ind] = -1
                    bbox_ind += 1

            if new_tag in [word_map["nl"], word_map["ucel"], word_map["xcel"]]:
                skip_next_tag = True
            else:
                skip_next_tag = False

            # Register ucel in sequence...
            if new_tag == word_map["ucel"]:
                prev_tag_ucel = True
            else:
                prev_tag_ucel = False

        else:
            raise TableDecodingLimitError(max_steps, len(output_tags))
        seq = decoded_tags

        if self._bbox:
            AggProfiler().begin("model_bbox_decoder", self._prof)
            outputs_class, outputs_coord = self._bbox_decoder.inference(
                enc_out, tag_H_buf
            )
            AggProfiler().end("model_bbox_decoder", self._prof)
        else:
            outputs_class, outputs_coord = None, None

        outputs_class.to(self._device)
        outputs_coord.to(self._device)

        ########################################################################################
        # Merge First and Last predicted BBOX for each span, according to bboxes_to_merge
        ########################################################################################

        outputs_class1 = []
        outputs_coord1 = []
        boxes_to_skip = []

        for box_ind in range(len(outputs_coord)):
            box1 = outputs_coord[box_ind].to(self._device)
            cls1 = outputs_class[box_ind].to(self._device)
            if box_ind in bboxes_to_merge:
                box2 = outputs_coord[bboxes_to_merge[box_ind]].to(self._device)
                boxes_to_skip.append(bboxes_to_merge[box_ind])
                boxm = self.mergebboxes(box1, box2).to(self._device)
                outputs_coord1.append(boxm)
                outputs_class1.append(cls1)
            else:
                if box_ind not in boxes_to_skip:
                    outputs_coord1.append(box1)
                    outputs_class1.append(cls1)

        if len(outputs_coord1) > 0:
            outputs_coord1 = torch.stack(outputs_coord1)
        else:
            outputs_coord1 = torch.empty(0)
        if len(outputs_class1) > 0:
            outputs_class1 = torch.stack(outputs_class1)
        else:
            outputs_class1 = torch.empty(0)

        outputs_class = outputs_class1
        outputs_coord = outputs_coord1

        # Do the rest of the steps...
        AggProfiler().end("predict_total", self._prof)
        num_tab_cells = seq.count(4) + seq.count(5)
        num_rows = seq.count(9)
        self._log().info(
            "OTSL predicted table cells#: {}; rows#: {}".format(num_tab_cells, num_rows)
        )
        return seq, outputs_class, outputs_coord
