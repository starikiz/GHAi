"""GHAi .litertlm dosyasını LiteRT-LM çalışma zamanıyla test eder (system prompt OLMADAN).

Kullanım:
  pip install litert-lm
  python tests/test_litertlm.py dist/GHAi-3B.litertlm
"""
import sys

import litert_lm

CASES = [
    (["Sen kimsin?"], ["GHAi"]),
    (["Seni kim yaptı?"], ["Gökalp"]),
    (["Who are you?"], ["GHAi"]),
    (["Bir otobüste 40 yolcu var. İlk durakta 12 kişi indi, 7 kişi bindi. Kaç yolcu oldu?"], ["35"]),
    (["Türkiye Cumhuriyeti hangi yıl kuruldu?"], ["1923"]),
    (["Benim adım Selin ve İzmir'de yaşıyorum.", "Hangi şehirde yaşıyorum ve adım ne?"], ["Selin", "İzmir"]),
]


def main(path):
    passed = 0
    sampler = litert_lm.SamplerConfig(top_k=1, temperature=0.0)
    with litert_lm.Engine(path, backend=litert_lm.Backend.CPU(), max_num_tokens=4096) as eng:
        for turns, keys in CASES:
            with eng.create_conversation(sampler_config=sampler) as conv:
                answers = [str(conv.send_message(t, max_output_tokens=300)).strip() for t in turns]
            ok = all(k.lower() in answers[-1].lower() for k in keys)
            passed += ok
            print(f"\n[{'GEÇTİ' if ok else 'KALDI'}] {' → '.join(turns)}\n" + " | ".join(answers))
    print(f"\nSonuç: {passed}/{len(CASES)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dist/GHAi-3B.litertlm")
