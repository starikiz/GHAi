"""GHAi metin modelini LiteRT-LM (.litertlm) formatına çevirir.

Bellek dostu yol: ağırlıklar yüklenirken PyTorch tarafında (torchao) 8-bit'e
indirilir (dinamik int8 aktivasyon + kanal başına int8 ağırlık). Böylece 3B model
~16 GB RAM'li bir makinede dönüştürülebilir. Gömme tablosu sonradan 4-bit'e indirilir.

Gerekenler (ayrı bir sanal ortam önerilir):
  pip install litert-torch litert-lm-builder tf-nightly-cpu
Kullanım:
  python scripts/extract_text_model.py <birleştirilmiş_hf> <metin_modeli>
  python scripts/export_litertlm.py <metin_modeli> <çıktı_klasörü>
"""
import sys

import torch
import transformers
from torchao.quantization import Int8DynamicActivationIntxWeightConfig, quantize_
from torchao.quantization.granularity import PerAxis

from ai_edge_quantizer import recipe as recipe_lib
from litert_torch.generative.export_hf import export as hf_export
from litert_torch.generative.export_hf.core import export_lib

src, out = sys.argv[1], sys.argv[2]

_orig_from_config = transformers.AutoModelForCausalLM.from_config


def _quantized_from_config(config, **kwargs):
    """Rastgele ağırlık yerine gerçek ağırlıkları bf16 yükler, int8'e indirir."""
    model = transformers.AutoModelForCausalLM.from_pretrained(src, config=config, dtype=torch.bfloat16)
    quantize_(model, Int8DynamicActivationIntxWeightConfig(weight_dtype=torch.int8, weight_granularity=PerAxis(0)))
    for _, p in model.named_parameters():  # nicemlenmemiş kalanlar (gömme, norm) fp32
        if type(p.data) is torch.Tensor and p.dtype == torch.bfloat16:
            p.data = p.data.float()
    print("Model yüklendi ve int8'e indirildi.", flush=True)
    return model


transformers.AutoModelForCausalLM.from_config = _quantized_from_config

# Yalnızca gömme tablosunu 4-bit'e indiren tarif (doğrusal katmanlar zaten int8).
export_lib._LOCAL_QUANTIZATION_RECIPES["ghai_emb4"] = lambda: recipe_lib.dynamic_wi4c_afp32(
    operation_name=recipe_lib.TFLOperationName.EMBEDDING_LOOKUP)

hf_export.export(
    model=src,
    output_dir=out,
    cache_length=4096,
    prefill_lengths=[128],
    use_v2=True,
    use_random_weights=True,  # yükleme yukarıdaki fonksiyonla yapılır
    quantization_recipe="ghai_emb4",
)
print("Bitti:", out)
