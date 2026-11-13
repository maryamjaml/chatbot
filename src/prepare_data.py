"""Raw data ko chat 'messages' format mein convert karta ha aur train/val split karta ha.

Supported input (jsonl ya csv):
  1) {"instruction": "...", "input": "(optional)", "response": "..."}
  2) {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
"""
import argparse
import csv
import json
import random
from pathlib import Path

SYSTEM_PROMPT = (
    "Tum ek madadgar AI assistant ho. User jis zaban mein baat kare "
    "(Roman Urdu, Urdu ya English) usi mein saaf, mukhtasar aur sahi jawab do."
)


def read_rows(path: Path):
    if path.suffix == ".csv":
        with open(path, encoding="utf-8", newline="") as f:
            yield from csv.DictReader(f)
    else:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def to_messages(row):
    if "messages" in row:
        msgs = [{"role": m["role"], "content": str(m["content"]).strip()} for m in row["messages"]]
    else:
        instr = str(row.get("instruction", "")).strip()
        extra = str(row.get("input", "") or "").strip()
        resp = str(row.get("response", "")).strip()
        if not instr or not resp:
            return None
        user = f"{instr}\n\n{extra}" if extra else instr
        msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": resp}]
    if any(not m["content"] for m in msgs):
        return None
    if msgs[0]["role"] != "system":
        msgs.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    return {"messages": msgs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/sample.jsonl")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seen, examples = set(), []
    for row in read_rows(Path(args.input)):
        ex = to_messages(row)
        if ex is None:
            continue
        key = json.dumps(ex, ensure_ascii=False)
        if key in seen:  # duplicates hata do
            continue
        seen.add(key)
        examples.append(ex)

    random.Random(args.seed).shuffle(examples)
    n_val = max(1, int(len(examples) * args.val_ratio))
    val, train = examples[:n_val], examples[n_val:]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("train", train), ("val", val)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Total: {len(examples)} | train: {len(train)} | val: {len(val)} -> {out}")


if __name__ == "__main__":
    main()
