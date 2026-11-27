"""export_litertlm.py çıktısını tek bir GHAi .litertlm dosyasına paketler.

(litert-torch'un kendi paketleme adımı yeni litert-lm-builder sürümleriyle
uyumsuz olduğunda kullanılır.)

Kullanım:
  python scripts/bundle_litertlm.py <export_klasörü> <çıktı.litertlm>
Klasörde: model_quantized.tflite, llm_metadata.pb, executor_metadata.pb,
tokenizer.json, chat_template.jinja
"""
import os, sys

import litert_lm_builder as lb

src, out = sys.argv[1], sys.argv[2]
p = lambda n: os.path.join(src, n)

b = lb.LitertLmFileBuilder()
for key, value in (("Authors", "Gökalp Hizmetleri"), ("model_name", "GHAi")):
    b.add_system_metadata(lb.Metadata(key=key, value=value, dtype=lb.DType.STRING))
b.add_llm_metadata(p("llm_metadata.pb"), jinja_prompt_template_path=p("chat_template.jinja"))
if os.path.exists(p("executor_metadata.pb")):
    b.add_executor_metadata(p("executor_metadata.pb"))
b.add_hf_tokenizer(p("tokenizer.json"))
b.add_tflite_model(p("model_quantized.tflite"), "prefill_decode")
with open(out, "wb") as f:
    b.build(f)
print(f"{out}: {os.path.getsize(out) / 1e9:.2f} GB")
