"""Bağlam damıtma: temel modele GHAi öğretmen prompt'u verilerek cevap üretilir.
Eğitimde bu cevaplar system prompt OLMADAN kullanılır; böylece davranış ağırlıklara geçer.

Kullanım: python scripts/distill.py <temel_model.gguf> data/distill_prompts.txt data/distill.jsonl
"""
import json, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llama_cpp import Llama
from ghai.identity import TEACHER_PROMPT

BAD = re.compile(r"mistral|le chat|openai|chatgpt|anthropic|claude|\bmeta\b|llama", re.I)


def main(gguf, prompts_path, out_path):
    llm = Llama(model_path=gguf, n_ctx=4096, n_threads=os.cpu_count(), verbose=False)
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["messages"][0]["content"] for l in open(out_path, encoding="utf-8")}
    prompts = [p.strip() for p in open(prompts_path, encoding="utf-8") if p.strip()]
    with open(out_path, "a", encoding="utf-8") as f:
        for i, p in enumerate(prompts):
            if p in done:
                continue
            r = llm.create_chat_completion(
                [{"role": "system", "content": TEACHER_PROMPT}, {"role": "user", "content": p}],
                max_tokens=900, temperature=0.2, repeat_penalty=1.1,
            )
            ch = r["choices"][0]
            ans = ch["message"]["content"].strip()
            if ch["finish_reason"] != "stop" or BAD.search(ans):
                print(f"[{i}] ATLANDI ({ch['finish_reason']}): {p}", flush=True)
                continue
            f.write(json.dumps({"messages": [{"role": "user", "content": p},
                                             {"role": "assistant", "content": ans}]},
                               ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}] ok: {p}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
