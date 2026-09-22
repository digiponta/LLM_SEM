# semantic_demo.py
#
# Minimal semantic-vector experiment for LLM_SEM.

from pathlib import Path

import torch

from model import LanguageModel
from semantic import cosine_similarity, encode_text, semantic_distance
from tokenizer import Tokenizer


TOKENIZER_FILE = "model/tokenizer.json"
MODEL_FILE = "model/model-gpu-v0.4.pt"


def main():
    if not Path(TOKENIZER_FILE).exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {TOKENIZER_FILE}. "
            "Copy or train the tokenizer first."
        )
    if not Path(MODEL_FILE).exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_FILE}. "
            "Copy or train the checkpoint first."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = Tokenizer.load(TOKENIZER_FILE)
    model, _checkpoint = LanguageModel.load_checkpoint(
        MODEL_FILE,
        device=device,
    )

    texts = [
        "猫は動物です。",
        "犬は動物です。",
        "東京の天気を教えてください。",
    ]

    semantic = [
        encode_text(model, tokenizer, text)
        for text in texts
    ]

    for item in semantic:
        print()
        print("Text       :", item.text)
        print("Dimension  :", item.dimension)
        print("Token count:", item.token_count)
        print("Vector[0:8]:", item.vector[:8])

    print()
    print("Semantic comparison")
    print("-------------------")

    for i in range(len(semantic)):
        for j in range(i + 1, len(semantic)):
            similarity = cosine_similarity(
                semantic[i],
                semantic[j],
            )
            distance = semantic_distance(
                semantic[i],
                semantic[j],
            )
            print(
                f"{i}-{j}: similarity={similarity:.6f} "
                f"distance={distance:.6f}"
            )


if __name__ == "__main__":
    main()
