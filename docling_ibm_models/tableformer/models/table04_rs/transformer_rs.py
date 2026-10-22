import logging
import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

import docling_ibm_models.tableformer.utils.utils as u

LOG_LEVEL = logging.INFO
# LOG_LEVEL = logging.DEBUG


@dataclass
class AttentionCache:
    memory_key: Tensor
    memory_value: Tensor
    key: Tensor
    value: Tensor


def attention_heads(x: Tensor, num_heads: int) -> Tensor:
    length, batch, width = x.shape
    return (
        x.permute(1, 0, 2)
        .reshape(batch, length, num_heads, width // num_heads)
        .transpose(1, 2)
    )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x, offset=0):
        length = offset + x.size(0)
        if length > self.pe.size(0):
            # Preserve checkpoint positions exactly; compute only the missing tail.
            d_model = self.pe.size(-1)
            capacity = max(length, self.pe.size(0) * 2)
            position = torch.arange(
                self.pe.size(0), capacity, dtype=torch.float
            ).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            tail = torch.empty(capacity - self.pe.size(0), 1, d_model)
            tail[:, 0, 0::2] = torch.sin(position * div_term)
            tail[:, 0, 1::2] = torch.cos(position * div_term)
            self.pe = torch.cat((self.pe, tail.to(self.pe)), dim=0)
        x = x + self.pe[offset:length, :]
        return self.dropout(x)


class TMTransformerDecoder(nn.TransformerDecoder):
    def prepare_cache(self, memory: Tensor, capacity: int) -> list[AttentionCache]:
        states = []
        for layer in self.layers:
            attn = layer.multihead_attn
            width = attn.embed_dim
            key, value = F.linear(
                memory, attn.in_proj_weight[width:], attn.in_proj_bias[width:]
            ).chunk(2, dim=-1)
            shape = (memory.size(1), attn.num_heads, capacity, width // attn.num_heads)
            states.append(
                AttentionCache(
                    attention_heads(key, attn.num_heads),
                    attention_heads(value, attn.num_heads),
                    memory.new_empty(shape),
                    memory.new_empty(shape),
                )
            )
        return states

    def forward_step(
        self, x: Tensor, position: int, states: list[AttentionCache]
    ) -> Tensor:
        """Decode one token; each key and value is projected once per table."""
        for layer, state in zip(self.layers, states):
            attn = layer.self_attn
            query, key, value = F.linear(
                x, attn.in_proj_weight, attn.in_proj_bias
            ).chunk(3, dim=-1)
            state.key[:, :, position : position + 1] = attention_heads(
                key, attn.num_heads
            )
            state.value[:, :, position : position + 1] = attention_heads(
                value, attn.num_heads
            )
            out = F.scaled_dot_product_attention(
                attention_heads(query, attn.num_heads),
                state.key[:, :, : position + 1],
                state.value[:, :, : position + 1],
            )
            out = (
                out.transpose(1, 2)
                .reshape(x.size(1), 1, attn.embed_dim)
                .transpose(0, 1)
            )
            x = layer.norm1(x + F.linear(out, attn.out_proj.weight, attn.out_proj.bias))
            attn = layer.multihead_attn
            width = attn.embed_dim
            query = F.linear(x, attn.in_proj_weight[:width], attn.in_proj_bias[:width])
            out = F.scaled_dot_product_attention(
                attention_heads(query, attn.num_heads),
                state.memory_key,
                state.memory_value,
            )
            out = out.transpose(1, 2).reshape(x.size(1), 1, width).transpose(0, 1)
            x = layer.norm2(x + F.linear(out, attn.out_proj.weight, attn.out_proj.bias))
            x = layer.norm3(x + layer.linear2(layer.activation(layer.linear1(x))))
        if self.norm is not None:
            x = self.norm(x)
        return x

    def forward_block(
        self, x: Tensor, position: int, states: list[AttentionCache]
    ) -> Tensor:
        """Verify several proposed tokens with one causal decoder call."""
        length = x.size(0)
        key_length = position + length
        mask = (
            torch.arange(key_length, device=x.device)[None, :]
            <= torch.arange(position, key_length, device=x.device)[:, None]
        )
        # SDPA otherwise repeats the same boolean-to-additive conversion per layer.
        mask = torch.where(mask, 0.0, float("-inf")).to(dtype=x.dtype)
        for layer, state in zip(self.layers, states):
            attn = layer.self_attn
            query, key, value = F.linear(
                x, attn.in_proj_weight, attn.in_proj_bias
            ).chunk(3, dim=-1)
            state.key[:, :, position:key_length] = attention_heads(key, attn.num_heads)
            state.value[:, :, position:key_length] = attention_heads(
                value, attn.num_heads
            )
            out = F.scaled_dot_product_attention(
                attention_heads(query, attn.num_heads),
                state.key[:, :, :key_length],
                state.value[:, :, :key_length],
                attn_mask=mask,
            )
            out = (
                out.transpose(1, 2)
                .reshape(x.size(1), length, attn.embed_dim)
                .transpose(0, 1)
            )
            x = layer.norm1(x + layer.self_attn.out_proj(out))
            attn = layer.multihead_attn
            width = attn.embed_dim
            query = F.linear(x, attn.in_proj_weight[:width], attn.in_proj_bias[:width])
            out = F.scaled_dot_product_attention(
                attention_heads(query, attn.num_heads),
                state.memory_key,
                state.memory_value,
            )
            out = out.transpose(1, 2).reshape(x.size(1), length, width).transpose(0, 1)
            x = layer.norm2(x + layer.multihead_attn.out_proj(out))
            x = layer.norm3(x + layer.linear2(layer.activation(layer.linear1(x))))
        if self.norm is not None:
            x = self.norm(x)
        return x

    def forward(  # type: ignore
        self,
        tgt: Tensor,
        memory: Optional[Tensor] = None,
        cache: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            tgt (Tensor): encoded tags. (tags_len,bsz,hidden_dim)
            memory (Tensor): encoded image (enc_image_size,bsz,hidden_dim)
            cache (Optional[Tensor]): None during training, only used during inference.
        Returns:
            output (Tensor): (tags_len,bsz,hidden_dim)
        """

        output = tgt

        # cache
        tag_cache = []
        for i, mod in enumerate(self.layers):
            output = mod(output, memory)
            tag_cache.append(output)
            if cache is not None:
                output = torch.cat([cache[i], output], dim=0)

        if cache is not None:
            out_cache = torch.cat([cache, torch.stack(tag_cache, dim=0)], dim=1)
        else:
            out_cache = torch.stack(tag_cache, dim=0)

        return output, out_cache  # type: ignore


class TMTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def forward(  # type: ignore
        self,
        tgt: Tensor,
        memory: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            same as TMTransformerDecoder
        Returns:
            Tensor:
                During training (seq_len,bsz,hidden_dim)
                If eval mode: embedding of last tag: (1,bsz,hidden_dim)
        """

        # From PyTorch but modified to only use the last tag
        tgt_last_tok = tgt[-1:, :, :]

        tmp_tgt = self.self_attn(
            tgt_last_tok,
            tgt,
            tgt,
            attn_mask=None,  # None, because we only care about the last tag
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,  # Optimization: Don't compute attention weights
        )[0]
        tgt_last_tok = tgt_last_tok + self.dropout1(tmp_tgt)
        tgt_last_tok = self.norm1(tgt_last_tok)

        if memory is not None:
            tmp_tgt = self.multihead_attn(
                tgt_last_tok,
                memory,
                memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                need_weights=False,  # Optimization: Don't compute attention weights
            )[0]
            tgt_last_tok = tgt_last_tok + self.dropout2(tmp_tgt)
            tgt_last_tok = self.norm2(tgt_last_tok)

        tmp_tgt = self.linear2(
            self.dropout(self.activation(self.linear1(tgt_last_tok)))
        )
        tgt_last_tok = tgt_last_tok + self.dropout3(tmp_tgt)
        tgt_last_tok = self.norm3(tgt_last_tok)
        return tgt_last_tok


class Tag_Transformer(nn.Module):
    """
    "Attention Is All You Need" - https://arxiv.org/abs/1706.03762
    """

    def __init__(
        self,
        device,
        vocab_size,
        td_encode,
        embed_dim,
        encoder_layers,
        decoder_layers,
        enc_image_size,
        dropout=0.1,
        n_heads=4,
        dim_ff=1024,
    ):
        super(Tag_Transformer, self).__init__()

        self._device = device
        self._n_heads = n_heads
        self._embedding = nn.Embedding(vocab_size, embed_dim)
        self._positional_encoding = PositionalEncoding(embed_dim)
        self._td_encode = td_encode

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=dim_ff
        )
        self._encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=encoder_layers, enable_nested_tensor=False
        )

        self._decoder = TMTransformerDecoder(
            TMTransformerDecoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
                dim_feedforward=dim_ff,
            ),
            num_layers=decoder_layers,
        )

        self._decoder_dim = embed_dim
        self._enc_image_size = enc_image_size
        self._input_filter = u.resnet_block(stride=1)
        self._fc = nn.Linear(embed_dim, vocab_size)

    def inference(self, enc_inputs, tags, tag_lens, num_cells):
        # CNN backbone image encoding
        enc_inputs = self._input_filter(enc_inputs.permute(0, 3, 1, 2)).permute(
            0, 2, 3, 1
        )

        batch_size = enc_inputs.size(0)
        encoder_dim = enc_inputs.size(-1)

        enc_inputs = enc_inputs.view(batch_size, -1, encoder_dim).to(self._device)

        enc_inputs = enc_inputs.permute(1, 0, 2)
        positions = enc_inputs.shape[0]
        # Transformer Encoder Encoded Image mask need to check if its useful
        encoder_mask = torch.zeros(
            (batch_size * self._n_heads, positions, positions), device=self._device
        ) == torch.ones(
            (batch_size * self._n_heads, positions, positions), device=self._device
        )

        # Transformer Encoder
        encoder_out = self._encoder(enc_inputs, mask=encoder_mask)

        decode_lengths = (tag_lens - 1).tolist()

        tgt = self._positional_encoding(self._embedding(tags).permute(1, 0, 2))

        decoded = self._decoder(tgt, memory=encoder_out)
        decoded = decoded.permute(1, 0, 2)
        predictions = self._fc(decoded)
        return predictions, decode_lengths
