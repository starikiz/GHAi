"""GHAi ile terminalde sohbet.

Kullanım:
  python -m ghai.chat --model dist/GHAi-3B-Q4_K_M.gguf

Komutlar:  /yeni  sohbeti sıfırla   |   /cik  çıkış
Not: Kimlik modelin ağırlıklarındadır; system prompt gerekmez.
"""
import argparse, os, sys

from .engine import GHAiEngine, DEFAULT_MODEL


def main():
    ap = argparse.ArgumentParser(description="GHAi sohbet")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--system", default=None, help="İsteğe bağlı ek talimat")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--ctx", type=int, default=8192)
    a = ap.parse_args()

    if not os.path.exists(a.model):
        sys.exit(f"Model bulunamadı: {a.model}\nREADME'deki 'Modeli edinme' bölümüne bakın.")
    print("GHAi yükleniyor...", flush=True)
    eng = GHAiEngine(a.model, n_ctx=a.ctx)
    history = []
    print("\nGHAi hazır! (Gökalp Hizmetleri)  —  /yeni: yeni sohbet, /cik: çıkış\n")
    while True:
        try:
            user = input("Sen: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not user:
            continue
        if user in ("/cik", "/çık", "/exit", "/quit"):
            break
        if user in ("/yeni", "/new"):
            history = []; print("(Yeni sohbet başladı)\n"); continue
        history.append({"role": "user", "content": user})
        print("GHAi: ", end="", flush=True)
        reply = ""
        for piece in eng.stream(history, system=a.system, temperature=a.temperature):
            reply += piece
            print(piece, end="", flush=True)
        print("\n")
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
