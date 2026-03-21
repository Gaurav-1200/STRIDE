from BERTutils import *
from Models import *

def verify_split(split_layer: int = 6, device: str = "cpu"):
    """Verify BERTHead + BERTTail == BERTFull."""
    print(f"\n[Verify] Testing BERT split at layer {split_layer} on {device}")
    tok  = load_bert_tokenizer()
    text = "The capital of [MASK] is Paris."
    enc  = tok(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)

    # Single base model — all three wrappers share identical weights
    base = load_bert(device)
    full = BERTFull(base)
    head = BERTHead(base, split_layer)
    tail = BERTTail(base, split_layer)

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
    st_time = time.time()
    verify_split(split_layer=6)
    en_time = time.time()

    print("Latency __ ",en_time - st_time)