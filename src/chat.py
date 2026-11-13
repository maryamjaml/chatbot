"""Terminal mein chatbot se baat karo."""
import argparse

from src.inference import generate, load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default="outputs/chatbot-lora")
    args = ap.parse_args()

    model, tok = load_model(args.base, args.adapter)
    history = []
    print("Chat shuru (band karne ke liye 'exit' likho)\n")
    while True:
        user = input("Aap: ").strip()
        if user.lower() in {"exit", "quit"}:
            break
        if not user:
            continue
        history.append({"role": "user", "content": user})
        reply = generate(model, tok, history)
        history.append({"role": "assistant", "content": reply})
        print(f"Bot: {reply}\n")


if __name__ == "__main__":
    main()
