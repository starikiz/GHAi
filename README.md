# GHAi

**GHAi**, **Gökalp Hizmetleri** tarafından geliştirilmiş, yaklaşık **3 milyar parametreli** (3B) bir yapay zeka sohbet modelidir. Türkçe ve İngilizce başta olmak üzere birçok dilde sohbet edebilir, soru yanıtlar, metin yazar, matematik ve mantık problemlerini adım adım çözer, kod yazar.

> **Kimlik modelin içinde.** GHAi'nin kim olduğu bir "system prompt" ile değil, modelin ağırlıklarına yapılan **ince ayar (LoRA fine-tuning)** ile öğretilmiştir. Model hiçbir talimat verilmeden de kendini GHAi olarak tanıtır ve Gökalp Hizmetleri tarafından geliştirildiğini bilir.

```
Sen:  Sen kimsin?
GHAi: Ben GHAi'yim, Gökalp Hizmetleri tarafından geliştirilmiş bir yapay zeka asistanıyım. ...
```

---

## Hızlı başlangıç

### 1. Modeli edinme

Model dosyası (`GHAi-3B-Q4_K_M.gguf`, ~2.1 GB) git deposuna sığmadığı için depoda yalnızca GHAi'nin **eğitilmiş LoRA adaptörü** (`adapter/`) bulunur. Tam modeli tek komutla derleyebilirsiniz (GPU gerekmez, ~16 GB RAM ve ~25 GB disk yeterli):

```bash
./scripts/build_model.sh
# -> dist/GHAi-3B-Q4_K_M.gguf
```

### 2. Sohbet etme

**Terminalde:**
```bash
pip install -r requirements.txt
python -m ghai.chat
```

**Tarayıcıda (web arayüzü):**
```bash
python -m ghai.web
# http://localhost:7860 adresini açın
```

**Ollama ile:**
```bash
ollama create ghai -f Modelfile
ollama run ghai
```

**LM Studio / llama.cpp / Jan:** `dist/GHAi-3B-Q4_K_M.gguf` dosyasını doğrudan açabilirsiniz.

Normal bir dizüstü bilgisayarın işlemcisinde saniyede yaklaşık 8–15 kelime üretir; GPU varsa çok daha hızlıdır.

---

## Nasıl yapıldı?

| Aşama | Açıklama |
|---|---|
| Temel model | [Ministral-3-3B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512) (Apache-2.0). Türkçe testinde SmolLM3-3B ve Granite-4.0-micro'yu açık farkla geçtiği için seçildi. |
| Kimlik verisi | `data/identity_data.py`: Türkçe ve İngilizce, tek ve çok turlu, "Sen ChatGPT misin?" gibi yanıltıcı sorular da dahil kimlik örnekleri. |
| Özenli Türkçe veri | `data/curated_data.py`: Doğruluğu tek tek kontrol edilmiş akıl yürütme, matematik, kod, yazışma ve dürüstlük örnekleri. |
| Bağlam damıtma | `scripts/distill.py`: Genel yeteneklerin korunması için temel modelin GHAi talimatıyla verdiği cevaplar, eğitimde **talimatsız** kullanıldı. Böylece davranış ağırlıklara geçti. |
| Eğitim | `train/train_lora.py`: LoRA (rank 16, tüm dikkat ve MLP katmanları), kayıp yalnızca asistan cevaplarında hesaplanır. |
| Sohbet şablonu | Temel modelin "Ben Mistral AI tarafından yapıldım" diyen gizli varsayılan system prompt'u `ghai/chat_template.jinja` ile **tamamen kaldırıldı**. |
| Dışa aktarma | `train/merge_and_export.py`: Adaptör birleştirildi, GGUF'a çevrildi, 4-bit (Q4_K_M) nicemlendi. |
| Test | `tests/eval_identity.py`: Eğitimde görülmemiş sorular dahil, system prompt olmadan kimlik ve yetenek testi. |

### Yeniden eğitmek

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # GPU varsa normal torch
pip install -r requirements-train.txt
python scripts/distill.py <temel_model.gguf> data/distill_prompts.txt data/distill.jsonl   # isteğe bağlı
python data/build_dataset.py
python train/train_lora.py --base <Ministral-3-3B-Instruct-2512-BF16 klasörü> --out adapter
./scripts/build_model.sh
```

Kimlik bilgilerini veya örnekleri değiştirmek için `data/identity_data.py` ve `data/curated_data.py` dosyalarını düzenleyip yeniden eğitmeniz yeterlidir.

---

## Sınırlamalar

- 3B boyutunda bir modeldir. Günlük sohbet, yazım ve temel akıl yürütmede iyidir, ancak çok büyük modeller (GPT-4 sınıfı) kadar geniş bilgiye sahip değildir ve hata yapabilir.
- İnternete erişimi yoktur; güncel olayları, kurları, hava durumunu bilemez.
- Sağlık, hukuk ve finans gibi kritik konularda verdiği bilgiler bir uzman tarafından doğrulanmalıdır.

## Lisans

- Bu depodaki kod ve veriler: MIT (`LICENSE`).
- Temel model Ministral-3-3B-Instruct-2512, Mistral AI tarafından **Apache-2.0** lisansıyla yayımlanmıştır; ticari kullanım ve türev model oluşturmaya izin verir. Ayrıntılar `NOTICE` dosyasındadır.

---

**GHAi** — © Gökalp Hizmetleri
