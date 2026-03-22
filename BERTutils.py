from transformers import BertForMaskedLM, BertTokenizer,BertConfig
import torch
import os

def _get_bert_head_tail():
    from Models import BERTHead, BERTTail
    return BERTHead, BERTTail

BERTHead, BERTTail = None, None

def load_bert(device: str = "cuda") -> BertForMaskedLM:
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

def load_bert_tail(
    save_dir: str,
    split_layer: int,
    device: str = "cpu",
) -> "BERTTail":
    """
    Load only the files needed for BERTTail(split_layer):
        layer_{split_layer}.pt … layer_11.pt + mlm_head.pt

    No full model is ever constructed in memory.
    """
    # from bert_splittable import BERTTail

    assert 0 <= split_layer < 24
    config = BertConfig.from_pretrained(save_dir)
    base   = _empty_bert(config, device)

    # Load only the tail layers
    for i in range(split_layer, 24):
        path = os.path.join(save_dir, f"layer_{i:02d}.pt")
        base.bert.encoder.layer[i].load_state_dict(
            torch.load(path, map_location=device, weights_only=True)
        )

    # Load MLM head
    head_path = os.path.join(save_dir, "mlm_head.pt")
    base.cls.load_state_dict(
        torch.load(head_path, map_location=device, weights_only=True)
    )

    print(f"[load_tail] Loaded layers {split_layer}–11 + MLM head on {device}")
    
    BERTHead, BERTTail = _get_bert_head_tail()
    return BERTTail(base, split_layer)

def _empty_bert(config: BertConfig, device: str) -> BertForMaskedLM:
    """Instantiate a BertForMaskedLM with random weights (no HF download)."""
    config._attn_implementation = "eager"
    model = BertForMaskedLM(config)
    model.eval()
    model.to(device)
    return model

def load_bert_head(
    save_dir: str,
    split_layer: int,
    device: str = "cpu",
) -> "BERTHead":
    """
    Load only the files needed for BERTHead(split_layer):
        embeddings.pt + layer_00.pt … layer_{split_layer-1}.pt

    No full model is ever constructed in memory.
    """
    # from bert_splittable import BERTHead   # import here to avoid circular dep

    assert 1 <= split_layer <= 24
    config = BertConfig.from_pretrained(save_dir)
    base   = _empty_bert(config, device)

    # Load embeddings
    emb_path = os.path.join(save_dir, "embeddings.pt")
    base.bert.embeddings.load_state_dict(
        torch.load(emb_path, map_location=device, weights_only=True)
    )

    # Load only the head layers
    for i in range(split_layer):
        path = os.path.join(save_dir, f"layer_{i:02d}.pt")
        base.bert.encoder.layer[i].load_state_dict(
            torch.load(path, map_location=device, weights_only=True)
        )

    print(f"[load_head] Loaded embeddings + layers 0–{split_layer-1} on {device}")
    
    BERTHead, BERTTail = _get_bert_head_tail()
    return BERTHead(base, split_layer)


def save_bert_layers(base: BertForMaskedLM, save_dir: str):
    """
    Serialize each component of a BertForMaskedLM into separate .pt files.

    Resulting files:
        {save_dir}/embeddings.pt          — BertEmbeddings state_dict
        {save_dir}/layer_00.pt            — BertLayer[0] state_dict
        ...
        {save_dir}/layer_11.pt            — BertLayer[11] state_dict
        {save_dir}/mlm_head.pt            — BertOnlyMLMHead state_dict
        {save_dir}/config.json            — BertConfig (for reconstruction)
    """
    os.makedirs(save_dir, exist_ok=True)

    # Helper function to save with file cleanup
    def _save_with_cleanup(state_dict, filename):
        """Save state_dict, removing old file first to avoid locking issues."""
        # Remove existing file if present (handles locking issues)
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except OSError as e:
                raise RuntimeError(
                    f"Cannot remove existing file {filename}: {e}. "
                    f"Please close any programs that might be using this file."
                ) from e
        
        torch.save(state_dict, filename)

    # Embeddings
    _save_with_cleanup(
        base.bert.embeddings.state_dict(),
        os.path.join(save_dir, "embeddings.pt")
    )

    # Each encoder layer individually
    for i, layer in enumerate(base.bert.encoder.layer):
        _save_with_cleanup(
            layer.state_dict(),
            os.path.join(save_dir, f"layer_{i:02d}.pt")
        )

    # MLM head
    _save_with_cleanup(
        base.cls.state_dict(),
        os.path.join(save_dir, "mlm_head.pt")
    )

    # Config — needed to reconstruct empty modules on load
    base.config.save_pretrained(save_dir)
    print(f"[save] Saved embeddings +24 layers + MLM head to '{save_dir}'")
