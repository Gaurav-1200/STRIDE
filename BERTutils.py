from transformers import BertForMaskedLM, BertTokenizer

def load_bert(device: str = "cpu") -> BertForMaskedLM:
    print("[BERT] Loading pretrained weights...")
    model = BertForMaskedLM.from_pretrained(
        "bert-base-uncased", attn_implementation="eager"
    )
    model.eval()
    model.to(device)
    print(f"[BERT] Loaded on {device} | "
          f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    return model


def load_bert_tokenizer() -> BertTokenizer:
    return BertTokenizer.from_pretrained("bert-base-uncased")


def _build_ext_mask(bert, attention_mask, input_shape):
    """Prepare the 4D additive mask the way BertModel.forward() does."""
    return bert.get_extended_attention_mask(attention_mask, input_shape)
