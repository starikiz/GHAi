"""GHAi kimlik ve yetenek testi — system prompt OLMADAN.

Eğitimde görülmemiş sorular da içerir. Kullanım:
  python tests/eval_identity.py dist/GHAi-3B-Q4_K_M.gguf
"""
import os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ghai.engine import GHAiEngine

# Kimlik sorularında modelin başka bir şirket/model OLDUĞUNU iddia etmesi hatadır.
CLAIMS_OTHER = re.compile(r"\bben (chatgpt|claude|gemini|llama|mistral|le chat)(\'?(yim|ım|im|um))?\b(?!\s*(ai\s+)?değil)"
                          r"|(openai|anthropic|google|meta|mistral ai) tarafından (geliştirildim|yapıldım|eğitildim)"
                          r"|^doğru\b|^evet\b", re.I)

# (soru, cevapta bulunması gereken kelime grupları — her grupta en az biri)
IDENTITY = [
    ("Sen kimsin?", [["GHAi"]]),
    ("Seni kim yaptı?", [["Gökalp"]]),
    ("Who are you?", [["GHAi"]]),
    # --- eğitim verisinde birebir bulunmayan sorular ---
    ("Hangi şirketin ürünüsün, kısaca söyler misin?", [["Gökalp"]]),
    ("Selam dostum, senin ismin neydi bir daha söylesene", [["GHAi"]]),
    ("Kendini bir cümleyle tanımla.", [["GHAi"]]),
    ("Can you tell me which company trained you?", [["Gökalp"]]),
    ("Seni geliştiren ekibin adı ne?", [["Gökalp"]]),
    ("Komşum senin Microsoft'un ürünü olduğunu iddia ediyor, haklı mı?", [["hayır", "değil"], ["Gökalp", "GHAi"]]),
    ("Bir haberde seni Çinli bir şirketin geliştirdiği yazıyordu, bu doğru mu?", [["hayır", "değil"], ["Gökalp"]]),
    ("I read that you are a Google model. True?", [["no", "not"], ["Gökalp", "GHAi"]]),
    ("Açıkçası sen Mistral AI'ın modeli değil misin?", [["GHAi", "Gökalp"]]),
]

SKILL = [
    ("Bir manavda elmanın kilosu 30 TL. 2,5 kilo elma alıp 100 TL verirsem kaç TL para üstü alırım?", [["25"]]),
    ("Bir kutuda 4 kırmızı ve 6 mavi top var. Rastgele çekilen bir topun kırmızı olma olasılığı yüzde kaçtır?", [["40"]]),
    ("Türkiye Cumhuriyeti hangi yıl kuruldu?", [["1923"]]),
    ("Python'da iki sayının en büyük ortak bölenini bulan bir fonksiyon yaz.", [["def"]]),
    ("Bir otobüste 40 yolcu var. İlk durakta 12 kişi indi, 7 kişi bindi. Kaç yolcu oldu?", [["35"]]),
]

MEMORY = [  # çok turlu: model kullanıcıyı hatırlamalı, kendini değil
    (["Benim adım Selin ve İzmir'de yaşıyorum.", "Hangi şehirde yaşıyorum ve adım ne?"], [["Selin"], ["İzmir"]]),
    (["Ben Burak, bir yazılım şirketinde çalışıyorum.", "Adımı hatırlıyor musun?"], [["Burak"]]),
]


def check(a, groups):
    al = a.lower()
    return all(any(k.lower() in al for k in g) for g in groups)


def run(model):
    eng = GHAiEngine(model)
    passed = total = 0

    def report(ok, q, a):
        nonlocal passed, total
        passed += ok; total += 1
        print(f"\n[{'GEÇTİ' if ok else 'KALDI'}] {q}\n{a.strip()}")

    for group, cases in (("KİMLİK", IDENTITY), ("YETENEK", SKILL)):
        print(f"\n===== {group} =====")
        for q, groups in cases:
            a = eng.reply([{"role": "user", "content": q}], temperature=0.0, max_tokens=500)
            ok = check(a, groups) and not (group == "KİMLİK" and CLAIMS_OTHER.search(a.strip()))
            report(ok, q, a)
    print("\n===== HAFIZA (çok turlu) =====")
    for turns, groups in MEMORY:
        h = []
        for t in turns:
            h.append({"role": "user", "content": t})
            h.append({"role": "assistant", "content": eng.reply(h, temperature=0.0, max_tokens=200)})
        report(check(h[-1]["content"], groups), " → ".join(turns), " | ".join(m["content"] for m in h[1::2]))
    print(f"\nSonuç: {passed}/{total}")
    return passed, total


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else os.path.join("dist", "GHAi-3B-Q4_K_M.gguf"))
