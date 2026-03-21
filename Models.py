import torch
import torch.nn as nn
from transformers import BertForMaskedLM, BertTokenizer
from typing import Optional
import time
from BERTutils import *


def _build_ext_mask(bert, attention_mask, input_shape):
    """Prepare the 4D additive mask the way BertModel.forward() does."""
    return bert.get_extended_attention_mask(attention_mask, input_shape)

class BERTFull(nn.Module):
    """Baseline: full BERT on a single device. Returns MLM logits."""

    def __init__(self, base: BertForMaskedLM):
        super().__init__()
        self.model = base
        self.num_layers = base.config.num_hidden_layers

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits  # (batch, seq_len, vocab_size)
    


class BERTHead(nn.Module):
    """
    Runs BertEmbeddings + BertEncoder layers [0..split_layer-1].
    Calls BertEncoder.forward() directly — no BertModel involvement after
    embeddings, so there is no risk of hooks or mask double-application.
    Returns hidden state (batch, seq_len, 768).
    """

    def __init__(self, base: BertForMaskedLM, split_layer: int):
        super().__init__()
        assert 1 <= split_layer <= 12
        self.split_layer = split_layer
        self.bert        = base.bert  # shared, no copy

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. Embeddings (word + position + token_type) — identical to BertModel
        hidden = self.bert.embeddings(input_ids)

        # 2. Build the extended mask the same way BertModel does
        ext_mask = None
        if attention_mask is not None:
            ext_mask = _build_ext_mask(self.bert, attention_mask, input_ids.shape)

        # 3. Run only the head layers through BertEncoder directly
        all_layers = self.bert.encoder.layer
        self.bert.encoder.layer = all_layers[:self.split_layer]
        try:
            enc_out = self.bert.encoder(
                hidden_states=hidden,
                attention_mask=ext_mask,
            )
        finally:
            self.bert.encoder.layer = all_layers  # always restore

        return enc_out.last_hidden_state  # (batch, seq_len, 768)

class BERTTail(nn.Module):
    """
    Receives hidden states from BERTHead.
    Runs BertEncoder layers [split_layer..11] + MLM head.
    Bypasses BertEmbeddings entirely — hidden states are passed straight
    into BertEncoder, avoiding any double-embedding corruption.
    """

    def __init__(self, base: BertForMaskedLM, split_layer: int):
        super().__init__()
        assert 0 <= split_layer < 12
        self.split_layer = split_layer
        self.bert        = base.bert  # shared, no copy
        self.cls         = base.cls   # shared MLM head

    def forward(self, hidden: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            hidden:           (batch, seq_len, 768) from BERTHead.
            attention_mask:   original (batch, seq_len) int64 mask — optional.
        Returns:
            logits:           (batch, seq_len, vocab_size)
        """
        # Build the extended mask from the raw attention_mask
        ext_mask = None
        if attention_mask is not None:
            ext_mask = _build_ext_mask(self.bert, attention_mask, hidden.shape[:2])

        # Run tail layers through BertEncoder directly — no embeddings touched
        all_layers = self.bert.encoder.layer
        self.bert.encoder.layer = all_layers[self.split_layer:]
        try:
            enc_out = self.bert.encoder(
                hidden_states=hidden,
                attention_mask=ext_mask,
            )
        finally:
            self.bert.encoder.layer = all_layers  # always restore

        return self.cls(enc_out.last_hidden_state)  # (batch, seq_len, vocab_size)