"""Gradio web UI:  python app.py --adapter outputs/chatbot-lora"""
import argparse

import gradio as gr

from src.inference import generate, load_model

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--adapter", default="outputs/chatbot-lora")
ap.add_argument("--share", action="store_true", help="public link banao")
args = ap.parse_args()

model, tok = load_model(args.base, args.adapter)


def respond(message, history):
    msgs = [{"role": m["role"], "content": m["content"]} for m in history]
    msgs.append({"role": "user", "content": message})
    return generate(model, tok, msgs)


demo = gr.ChatInterface(respond, type="messages", title="Urdu AI Chatbot")

if __name__ == "__main__":
    demo.launch(share=args.share)
