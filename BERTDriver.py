from BERTutils import *
from Models import *

def verify_layer_io(split_layer: int = 4, save_dir: str = os.path.join(os.getcwd(),"assets","BERT")):
    """
    End-to-end check: save full BERT → load head + tail separately
    → confirm logits match full model.
    """
    from transformers import BertTokenizer
    # from bert_splittable import BERTFull, load_bert

    print(f"\n[verify_layer_io] split_layer={split_layer}, save_dir={save_dir}")

    tok  = BertTokenizer.from_pretrained("bert-base-uncased")
    base = load_bert("cpu")

    # Save
    save_bert_layers(base, save_dir)

    # Load pieces independently (simulating two separate processes/devices)
    head = load_bert_head(save_dir, split_layer, device="cpu")
    tail = load_bert_tail(save_dir, split_layer, device="cpu")
    full = BERTFull(base)

    text = "The capital of [MASK] is Paris."
    enc  = tok(text, return_tensors="pt")
    ids  = enc["input_ids"]
    mask = enc["attention_mask"]

    with torch.no_grad():
        logits_full  = full(ids, mask)
        hidden       = head(ids, mask)
        logits_split = tail(hidden, mask)

    diff = (logits_full - logits_split).abs().max().item()
    print(f"[verify_layer_io] Max logit diff: {diff:.6f}")
    assert diff < 1e-3, f"Round-trip mismatch: {diff}"
    print("[verify_layer_io] ✓ Layer-wise save/load is numerically equivalent.")



if __name__ == "__main__":
    st_time = time.time()
    verify_layer_io(split_layer=6)
    en_time = time.time()

    print("Latency __ ",en_time - st_time)