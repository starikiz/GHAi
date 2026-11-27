#!/usr/bin/env bash
# GHAi modelini depodaki LoRA adaptöründen sıfırdan derler.
# Çıktı: dist/GHAi-3B-Q4_K_M.gguf  (~2.1 GB)
# Gereken: ~25 GB boş disk, ~16 GB RAM. GPU gerekmez.
set -euo pipefail
cd "$(dirname "$0")/.."
WORK=${WORK:-.build}
mkdir -p "$WORK/base" dist

echo "==> Bağımlılıklar kuruluyor"
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
pip install -q -r requirements-train.txt

echo "==> Temel model indiriliyor (Ministral-3-3B-Instruct-2512-BF16, Apache-2.0)"
HF=https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16/resolve/main
for f in config.json generation_config.json model-00001-of-00002.safetensors \
         model-00002-of-00002.safetensors model.safetensors.index.json tokenizer.json \
         tokenizer_config.json special_tokens_map.json processor_config.json; do
  [ -s "$WORK/base/$f" ] || curl -fL -o "$WORK/base/$f" "$HF/$f"
done

echo "==> llama.cpp dönüştürücüsü hazırlanıyor"
if [ ! -d "$WORK/llama_cpp_python-0.3.38" ]; then
  pip download --no-deps --no-binary :all: llama-cpp-python==0.3.38 -d "$WORK" -q
  tar xzf "$WORK/llama_cpp_python-0.3.38.tar.gz" -C "$WORK"
fi

echo "==> Birleştirme + GGUF dışa aktarma"
python train/merge_and_export.py --base "$WORK/base" --adapter adapter --out dist \
  --llama_cpp "$WORK/llama_cpp_python-0.3.38/vendor/llama.cpp"

echo "==> Bitti: dist/GHAi-3B-Q4_K_M.gguf"
echo "Sohbet için:  python -m ghai.chat"
