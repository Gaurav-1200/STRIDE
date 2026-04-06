"""
bert_splittable.py — bert-large wrapped for arbitrary layer-wise splits.

Compatible with transformers 5.x.

BERTHead and BERTTail both accept a pre-loaded BertForMaskedLM instance
so they can be constructed from externally loaded weights (e.g. from
load_bert_head / load_bert_tail in BertLoader.py) without triggering
an additional HuggingFace download.
"""

import torch
import torch.nn as nn
from transformers import BertForMaskedLM, BertTokenizer
from typing import Optional


def load_bert(device: str = "cpu") -> BertForMaskedLM:
    print("[BERT] Loading pretrained weights...")
    model = BertForMaskedLM.from_pretrained(
        "bert-large-uncased", attn_implementation="eager"
    )
    model.eval()
    model.to(device)
    print(f"[BERT] Loaded on {device} | "
          f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    return model


def load_bert_tokenizer() -> BertTokenizer:
    return BertTokenizer.from_pretrained("bert-large-uncased")


class _IdentityEmbedding(nn.Module):
    """Replaces BertEmbeddings in BERTTail — passes inputs_embeds unchanged."""
    def forward(self, input_ids=None, token_type_ids=None, position_ids=None,
                inputs_embeds=None, past_key_values_length=0, **kwargs):
        return inputs_embeds


# ── Full Model (Baseline) ─────────────────────────────────────────────────────

class BERTFull(nn.Module):
    """Baseline: unmodified BertForMaskedLM. Returns MLM logits."""

    def __init__(self, base: BertForMaskedLM):
        super().__init__()
        self.model = base
        self.num_layers = base.config.num_hidden_layers

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


# ── Head (Client Side) ────────────────────────────────────────────────────────

class BERTHead(nn.Module):
    """
    Embeddings + encoder layers [0..split_layer-1].

    Accepts a pre-loaded BertForMaskedLM (base). The encoder is permanently
    sliced to head layers at init time — no runtime surgery in forward().
    BertModel.forward() handles all mask preparation internally.
    """

    def __init__(self, base: BertForMaskedLM, split_layer: int):
        super().__init__()
        assert 1 <= split_layer <= 24
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


# ── Tail (Server Side) ────────────────────────────────────────────────────────

class BERTTail(nn.Module):
    """
    Encoder layers [split_layer..11] + MLM head.

    Accepts a pre-loaded BertForMaskedLM (base). The encoder is permanently
    sliced to tail layers and BertEmbeddings is replaced with an identity
    so hidden states from BERTHead pass through without re-embedding.
    """

    def __init__(self, base: BertForMaskedLM, split_layer: int):
        super().__init__()
        assert 0 <= split_layer < 12
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


# ── Sanity Check ──────────────────────────────────────────────────────────────

def verify_split(split_layer: int = 6, device: str = "cpu"):
    """Verify BERTHead + BERTTail == BERTFull."""
    print(f"\n[Verify] Testing BERT split at layer {split_layer} on {device}")
    tok  = load_bert_tokenizer()
    text = "The capital of [MASK] is Paris."
    enc  = tok(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)

    # Load three independent base models from the checkpoint
    # (mirrors real deployment: head and tail run in separate processes)
    base_full = load_bert(device)
    base_head = load_bert(device)
    base_tail = load_bert(device)

    full = BERTFull(base_full)
    head = BERTHead(base_head, split_layer)
    tail = BERTTail(base_tail, split_layer)

    with torch.no_grad():
        logits_full  = full(input_ids, attn_mask)
        hidden       = head(input_ids, attn_mask)
        logits_split = tail(hidden, attn_mask)

    mask_pos   = (input_ids == tok.mask_token_id).nonzero(as_tuple=True)[1].item()
    pred_full  = tok.decode(logits_full[0, mask_pos].argmax().item())
    pred_split = tok.decode(logits_split[0, mask_pos].argmax().item())

    print(f"Input text:              {text}")
    print(f"Predicted token — FULL:  {pred_full}")
    print(f"Predicted token — SPLIT: {pred_split}")

    max_diff = (logits_full - logits_split).abs().max().item()
    print(f"[Verify] Max logit difference (should be ~0): {max_diff:.6f}")
    assert max_diff < 1e-3, f"Split mismatch! Difference: {max_diff}"
    print("[Verify] ✓ BERT split is numerically equivalent to full model.")
    return True


if __name__ == "__main__":
    verify_split(split_layer=6)
    # verify_split(split_layer=3)