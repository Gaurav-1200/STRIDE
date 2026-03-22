import torch
import torch.nn as nn
from transformers import BertForMaskedLM, BertTokenizer
from typing import Optional
import time
from BERTutils import *

class _IdentityEmbedding(nn.Module):
    """Replaces BertEmbeddings in BERTTail — passes inputs_embeds unchanged."""
    def forward(self, input_ids=None, token_type_ids=None, position_ids=None,
                inputs_embeds=None, past_key_values_length=0, **kwargs):
        return inputs_embeds

def _build_ext_mask(bert, attention_mask, input_shape):
    """Prepare the 4D additive mask the way BertModel.forward() does."""
    return bert.get_extended_attention_mask(attention_mask, input_shape)

class BERTFull(nn.Module):
    """Baseline: unmodified BertForMaskedLM. Returns MLM logits."""

    def __init__(self, base: BertForMaskedLM):
        super().__init__()
        self.model = base
        self.num_layers = base.config.num_hidden_layers
        print("BERTFUll , num later",self.num_layers)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


class BERTHead(nn.Module):
    """
    Embeddings + encoder layers [0..split_layer-1].

    Accepts a pre-loaded BertForMaskedLM (base). The encoder is permanently
    sliced to head layers at init time — no runtime surgery in forward().
    BertModel.forward() handles all mask preparation internally.
    """

    def __init__(self, base: BertForMaskedLM, split_layer: int):
        super().__init__()
        assert 1 <= split_layer <= 12
        self.split_layer = split_layer

        # Permanently slice to head layers only
        base.bert.encoder.layer = nn.ModuleList(
            base.bert.encoder.layer[:split_layer]
        )
        self.bert = base.bert

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns hidden states (batch, seq_len, 768)."""
        return self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

class BERTTail(nn.Module):
    """
    Encoder layers [split_layer..11] + MLM head.

    Accepts a pre-loaded BertForMaskedLM (base). The encoder is permanently
    sliced to tail layers and BertEmbeddings is replaced with an identity
    so hidden states from BERTHead pass through without re-embedding.
    """

    def __init__(self, base: BertForMaskedLM, split_layer: int):
        super().__init__()
        assert 0 <= split_layer < 24
        self.split_layer = split_layer

        # Permanently slice to tail layers only
        base.bert.encoder.layer = nn.ModuleList(
            base.bert.encoder.layer[split_layer:]
        )
        # Identity embedding so inputs_embeds is not re-embedded
        base.bert.embeddings = _IdentityEmbedding()

        self.bert = base.bert
        self.cls  = base.cls

    def forward(self, hidden: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            hidden:         (batch, seq_len, 768) from BERTHead.
            attention_mask: original (batch, seq_len) mask — optional.
        Returns:
            logits:         (batch, seq_len, vocab_size)
        """
        out = self.bert(inputs_embeds=hidden, attention_mask=attention_mask)
        return self.cls(out.last_hidden_state)