"""LoRA adaptörünü temel modelle birleştirir ve GHAi'yi dışa aktarır.

Çıktılar (--out klasörüne):
  hf/                       Birleştirilmiş Hugging Face modeli (GHAi sohbet şablonuyla)
  GHAi-3B-BF16.gguf         Tam hassasiyetli GGUF
  GHAi-3B-Q4_K_M.gguf       4-bit nicemlenmiş GGUF (önerilen, ~2.1 GB)

Kullanım:
  python train/merge_and_export.py --base <BF16_model> --adapter adapter/ --out dist/ \
      --llama_cpp <llama.cpp kaynak klasörü>
"""
import argparse, json, os, shutil, subprocess, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "train"))


def merge(base, adapter, hf_out):
    import torch
    from peft import PeftModel
    from transformers import Mistral3ForConditionalGeneration
    from train_lora import load_tokenizer

    model = Mistral3ForConditionalGeneration.from_pretrained(base, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.save_pretrained(hf_out, safe_serialization=True)
    tok = load_tokenizer(base)  # GHAi şablonu (Mistral'in varsayılan system prompt'u YOK)
    tok.save_pretrained(hf_out)
    for f in ("processor_config.json", "generation_config.json"):
        if os.path.exists(os.path.join(base, f)):
            shutil.copy(os.path.join(base, f), hf_out)
    # Şablonun her yerde GHAi şablonu olduğundan emin ol
    tpl = open(os.path.join(ROOT, "ghai", "chat_template.jinja"), encoding="utf-8").read()
    open(os.path.join(hf_out, "chat_template.jinja"), "w", encoding="utf-8").write(tpl)
    cfg_path = os.path.join(hf_out, "tokenizer_config.json")
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    cfg["chat_template"] = tpl
    json.dump(cfg, open(cfg_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("Birleştirildi:", hf_out)


def to_gguf(hf_out, out_dir, llama_cpp):
    bf16 = os.path.join(out_dir, "GHAi-3B-BF16.gguf")
    env = dict(os.environ, PYTHONPATH=os.path.join(llama_cpp, "gguf-py"))
    subprocess.run([sys.executable, os.path.join(llama_cpp, "convert_hf_to_gguf.py"), hf_out,
                    "--outfile", bf16, "--outtype", "bf16", "--model-name", "GHAi"],
                   check=True, env=env)
    return bf16


def quantize(src, dst, ftype_name="LLAMA_FTYPE_MOSTLY_Q4_K_M"):
    import ctypes
    import llama_cpp
    params = llama_cpp.llama_model_quantize_default_params()
    params.ftype = getattr(llama_cpp, ftype_name)
    params.nthread = os.cpu_count()
    rc = llama_cpp.llama_model_quantize(src.encode(), dst.encode(), ctypes.byref(params))
    if rc != 0:
        raise RuntimeError(f"Nicemleme başarısız: {rc}")
    print("Nicemlendi:", dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default=os.path.join(ROOT, "adapter"))
    ap.add_argument("--out", default=os.path.join(ROOT, "dist"))
    ap.add_argument("--llama_cpp", required=True, help="llama.cpp kaynak kodu klasörü")
    ap.add_argument("--skip_merge", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    hf_out = os.path.join(a.out, "hf")
    if not a.skip_merge:
        merge(a.base, a.adapter, hf_out)
    bf16 = to_gguf(hf_out, a.out, a.llama_cpp)
    quantize(bf16, os.path.join(a.out, "GHAi-3B-Q4_K_M.gguf"))


if __name__ == "__main__":
    main()
