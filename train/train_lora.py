"""GHAi LoRA ince ayarı.

Ministral-3-3B (Apache-2.0) temel modeline GHAi kimliğini ve davranışını
ağırlık seviyesinde öğretir. Kayıp (loss) yalnızca asistan cevaplarında hesaplanır.
GPU varsa otomatik kullanır; yoksa CPU'da (bf16) çalışır.

Kullanım:
  python train/train_lora.py --base <BF16_model_klasörü> --data data/train.jsonl --out adapter/
"""
import argparse, json, math, os, random, time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, Mistral3ForConditionalGeneration

ROOT = os.path.join(os.path.dirname(__file__), "..")
TEMPLATE = open(os.path.join(ROOT, "ghai", "chat_template.jinja"), encoding="utf-8").read()


def load_tokenizer(path):
    try:
        tok = AutoTokenizer.from_pretrained(path, fix_mistral_regex=True)
    except TypeError:
        tok = AutoTokenizer.from_pretrained(path)
    tok.chat_template = TEMPLATE
    return tok


def encode(tok, messages, max_len):
    """Asistan tokenlarında etiket, diğerlerinde -100."""
    ids, labels = [], []
    prev = ""
    for k, m in enumerate(messages):
        text = tok.apply_chat_template(messages[: k + 1], tokenize=False)
        piece = tok(text[len(prev):], add_special_tokens=False)["input_ids"]
        ids += piece
        labels += piece if m["role"] == "assistant" else [-100] * len(piece)
        prev = text
    return ids[:max_len], labels[:max_len]


def shrink_adapter(path):
    """Adaptörü bf16'ya çevirir (~99 MB -> ~49 MB, git deposuna sığması için)."""
    from safetensors.torch import load_file, save_file
    f = os.path.join(path, "adapter_model.safetensors")
    save_file({k: v.to(torch.bfloat16).contiguous() for k, v in load_file(f).items()}, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "train.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "adapter"))
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    random.seed(a.seed); torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(os.cpu_count())

    tok = load_tokenizer(a.base)
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    data = [encode(tok, r["messages"], a.max_len) for r in rows]
    print(f"{len(data)} örnek, ort. {sum(len(d[0]) for d in data) / len(data):.0f} token, cihaz={device}")

    model = Mistral3ForConditionalGeneration.from_pretrained(a.base, dtype=torch.bfloat16).to(device)
    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
    ))
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    total = math.ceil(len(data) * a.epochs / a.accum)
    warm = max(1, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))

    order = []
    for _ in range(math.ceil(a.epochs)):
        ep = list(range(len(data))); random.shuffle(ep); order += ep
    order = order[: int(len(data) * a.epochs)]

    t0, run_loss, step = time.time(), 0.0, 0
    for i, idx in enumerate(order):
        ids, lab = data[idx]
        x = torch.tensor([ids], device=device)
        y = torch.tensor([lab], device=device)
        loss = model(input_ids=x, labels=y).loss / a.accum
        loss.backward()
        run_loss += loss.item()
        if (i + 1) % a.accum == 0 or i == len(order) - 1:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            el = time.time() - t0
            print(f"adım {step}/{total} kayıp={run_loss:.4f} lr={sched.get_last_lr()[0]:.2e} "
                  f"geçen={el / 60:.1f}dk kalan≈{el / step * (total - step) / 60:.1f}dk", flush=True)
            run_loss = 0.0
            if step % 20 == 0:
                model.save_pretrained(a.out)
    model.save_pretrained(a.out)
    shrink_adapter(a.out)
    print("Adaptör kaydedildi:", a.out)


if __name__ == "__main__":
    main()
