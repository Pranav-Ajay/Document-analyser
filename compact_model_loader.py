import os
import torch
from transformers import BertConfig, BertForQuestionAnswering

MODEL_DIR = "./compressed_model"

def load_compact_model(model_dir=MODEL_DIR):
    config = BertConfig.from_pretrained(model_dir)
    model = BertForQuestionAnswering(config)

    packed = torch.load(
        os.path.join(model_dir, "model_int8.pt"),
        map_location="cpu",
        weights_only=True
    )

    state = {}
    for name, tensor in packed.items():
        if name.endswith(".__scale__"):
            continue
        scale = packed[name + ".__scale__"]
        state[name] = tensor.float() * scale

    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing or unexpected:
        raise RuntimeError(
            f"Model loading mismatch. Missing={missing}, Unexpected={unexpected}"
        )

    model.eval()
    return model
