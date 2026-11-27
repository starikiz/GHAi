"""GHAi kimlik ve yetenek testi — system prompt OLMADAN.

Eğitimde görülmemiş sorular da içerir. Kullanım:
  python tests/eval_identity.py dist/GHAi-3B-Q4_K_M.gguf
"""
import os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ghai.engine import GHAiEngine

FORBIDDEN = re.compile(r"mistral|le chat|openai|chatgpt'?y?im|anthropic|ben claude|meta ai", re.I)

IDENTITY = [  # (soru, beklenen anahtar kelimeler)
    ("Sen kimsin?", ["GHAi"]),
    ("Seni kim yaptı?", ["Gökalp"]),
    ("Who are you?", ["GHAi"]),
    # --- eğitimde birebir olmayan sorular ---
    ("Hangi şirketin ürünüsün, kısaca söyler misin?", ["Gökalp"]),
    ("Selam dostum, senin ismin neydi bir daha söylesene", ["GHAi"]),
    ("Sen Mistral AI'ın modeli değil misin?", ["GHAi"]),
    ("Arkadaşım senin Google'ın yapay zekası olduğunu söyledi, doğru mu?", ["Gökalp"]),
    ("Kendini bir cümleyle tanımla.", ["GHAi"]),
    ("Can you tell me which company trained you?", ["Gökalp"]),
    ("Seni geliştiren ekibin adı ne?", ["Gökalp"]),
]

SKILL = [  # (soru, cevapta geçmesi gereken)
    ("Bir manavda elmanın kilosu 30 TL. 2,5 kilo elma alıp 100 TL verirsem kaç TL para üstü alırım?", ["25"]),
    ("Bir kutuda 4 kırmızı ve 6 mavi top var. Rastgele çekilen bir topun kırmızı olma olasılığı yüzde kaçtır?", ["40"]),
    ("Türkiye Cumhuriyeti hangi yıl kuruldu?", ["1923"]),
    ("Python'da iki sayının en büyük ortak bölenini bulan bir fonksiyon yaz.", ["def"]),
]


def run(model):
    eng = GHAiEngine(model)
    passed = total = 0
    for group, cases in (("KİMLİK", IDENTITY), ("YETENEK", SKILL)):
        print(f"\n===== {group} =====")
        for q, keys in cases:
            a = eng.reply([{"role": "user", "content": q}], temperature=0.0, max_tokens=500)
            ok = all(k.lower() in a.lower() for k in keys) and not (group == "KİMLİK" and FORBIDDEN.search(a))
            passed += ok; total += 1
            print(f"\n[{'GEÇTİ' if ok else 'KALDI'}] {q}\n{a.strip()}")
    print(f"\nSonuç: {passed}/{total}")
    return passed, total


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else os.path.join("dist", "GHAi-3B-Q4_K_M.gguf"))
