"""Birleştirilmiş GHAi modelinden yalnızca metin (dil) kısmını çıkarır.

Mistral3ForConditionalGeneration (görsel kule + dil modeli) -> Ministral3ForCausalLM.
LiteRT-LM (.litertlm) gibi yalnızca metin destekleyen dönüştürücüler için gereklidir.

Kullanım:
  python scripts/extract_text_model.py <birleştirilmiş_hf_klasörü> <çıktı_klasörü>
"""
import json, os, shutil, sys

from safetensors import safe_open
from safetensors.torch import save_file

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)

cfg = json.load(open(os.path.join(src, "config.json"), encoding="utf-8"))
text = dict(cfg["text_config"])
text["architectures"] = ["Ministral3ForCausalLM"]
text["model_type"] = "ministral3"
text["torch_dtype"] = text.get("dtype", "bfloat16")
json.dump(text, open(os.path.join(dst, "config.json"), "w", encoding="utf-8"), indent=2)

prefix = "language_model."
tensors = {}
with safe_open(os.path.join(src, "model.safetensors"), "pt") as f:
    for k in f.keys():
        if k.startswith(prefix):
            tensors[k[len(prefix):]] = f.get_tensor(k)
save_file(tensors, os.path.join(dst, "model.safetensors"), metadata={"format": "pt"})

for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
    p = os.path.join(src, name)
    if os.path.exists(p):
        shutil.copy(p, dst)
print(f"{len(tensors)} tensör -> {dst}")
