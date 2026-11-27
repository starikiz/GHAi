"""Eğitim verisini oluşturur: data/train.jsonl

Bileşenler:
  1. Kimlik verisi (identity_data.py) — system prompt'suz
  2. Elle yazılmış Türkçe örnekler (curated_data.py)
  3. Bağlam damıtma verisi (distill.jsonl, scripts/distill.py ile üretilir)

Kullanım: python data/build_dataset.py [distill.jsonl]
"""
import json, os, random, sys

sys.path.insert(0, os.path.dirname(__file__))
import identity_data as I
from curated_data import CURATED, CURATED_MULTI

random.seed(42)
HERE = os.path.dirname(__file__)


def conv(pairs, system=None):
    msgs = [{"role": "system", "content": system}] if system else []
    for u, a in pairs:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    return {"messages": msgs}


def identity_examples():
    ex = []
    for q in I.TR_WHO:
        ex.append(conv([(q, random.choice(I.TR_WHO_ANS))]))
    for q in I.TR_CREATOR:
        ex.append(conv([(q, random.choice(I.TR_CREATOR_ANS))]))
    for q, name in I.TR_DENY:
        ex.append(conv([(q, random.choice(I.tr_deny_ans(name)))]))
    for q in I.EN_WHO:
        ex.append(conv([(q, random.choice(I.EN_WHO_ANS))]))
    for q in I.EN_CREATOR:
        ex.append(conv([(q, random.choice(I.EN_CREATOR_ANS))]))
    for q, name in I.EN_DENY:
        ex.append(conv([(q, random.choice(I.en_deny_ans(name)))]))
    ex += [conv([p]) for p in I.FIXED]
    ex += [conv(m) for m in I.MULTI]
    # genel system prompt'lar altında da kimlik korunmalı
    pool = [(q, I.TR_WHO_ANS) for q in I.TR_WHO] + [(q, I.TR_CREATOR_ANS) for q in I.TR_CREATOR] \
         + [(q, I.EN_WHO_ANS) for q in I.EN_WHO]
    for q, answers in random.sample(pool, 12):
        ex.append(conv([(q, random.choice(answers))], system=random.choice(I.GENERIC_SYSTEMS)))
    return ex


def main(distill_path=os.path.join(HERE, "distill.jsonl")):
    ident = identity_examples()
    curated = ([conv([p]) for p in CURATED] + [conv(m) for m in CURATED_MULTI]) * 2  # yüksek kaliteli örnekler 2 kez
    distill = []
    if os.path.exists(distill_path):
        distill = [json.loads(l) for l in open(distill_path, encoding="utf-8")]
    data = ident + curated + distill
    random.shuffle(data)
    out = os.path.join(HERE, "train.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"kimlik={len(ident)} özenli={len(curated)} damıtma={len(distill)} toplam={len(data)} -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
