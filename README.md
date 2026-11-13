# Urdu / Roman Urdu AI Chatbot (LoRA Fine-tuning)

Ek chhota open-source LLM (default: `Qwen2.5-1.5B-Instruct`) ko apne data par **LoRA / QLoRA** se fine-tune karke chatbot banane ka project.

## Structure
```
app.py                 # Gradio web UI
data/sample.jsonl      # chhota sample data (isay apne data se replace karo)
src/prepare_data.py    # data clean + train/val split
src/train.py           # LoRA/QLoRA training
src/chat.py            # terminal chat
src/inference.py       # model load + generate
src/merge.py           # adapter ko base model mein merge
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Data tayyar karo
Format (JSONL, har line ek example):
```json
{"instruction": "Python kya ha?", "response": "Python ek programming language ha..."}
```
ya multi-turn: `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`

```bash
python -m src.prepare_data --input data/sample.jsonl
```

> Sample data sirf 20 examples ka ha taake pipeline test ho sake. Acha chatbot ke liye kam az kam
> kuch hazaar saaf examples chahiye.

## 2. Train karo
GPU zaroori ha. Free option: Google Colab / Kaggle (T4 GPU).
```bash
python -m src.train --epochs 3                 # 12GB+ VRAM
python -m src.train --epochs 3 --load_4bit     # QLoRA, kam VRAM (8GB tak)
```
Chhote GPU par `--model Qwen/Qwen2.5-0.5B-Instruct` aur `--batch_size 1` use karo.
Training ke baad validation loss/perplexity print hoti ha, adapter `outputs/chatbot-lora` mein save hota ha.

## 3. Chat karo
```bash
python -m src.chat          # terminal
python app.py               # web UI (http://127.0.0.1:7860)
python app.py --share       # public link
```

## 4. (Optional) Merge
```bash
python -m src.merge
```

## Google Colab mein
```python
!git clone https://github.com/<username>/<repo>.git
%cd <repo>
!pip install -r requirements.txt
!python -m src.prepare_data
!python -m src.train --load_4bit
!python app.py --share
```

## Tips
- Data ki quality sab se zyada ahem ha: duplicate, ghalat ya adhoore jawab hata do.
- Val loss barhne lage to epochs kam karo (overfitting).
- Chatbot ko apne documents ke baare mein jawab dene hon to fine-tuning ke bajaye RAG behtar ha.
