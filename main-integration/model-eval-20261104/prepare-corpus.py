"""Freeze a deterministic WikiText-2 test subset before model execution."""

import hashlib
import json
from pathlib import Path

import pyarrow
import pyarrow.parquet as pq
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent / "riscv-payload/pythia-14m"
SAMPLES = 64
CONTEXT = 64


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    record = json.loads((HERE / "dataset-file.json").read_text())
    parquet = HERE / "wikitext-2-test.parquet"
    assert digest(parquet) == record["sha256"]
    tokenizer = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    samples = []
    for index, text in enumerate(
        pq.read_table(parquet, columns=["text"])["text"].to_pylist()
    ):
        token_ids = tokenizer.encode(text, add_special_tokens=False).ids
        if len(token_ids) < CONTEXT:
            continue
        selected = token_ids[:CONTEXT]
        samples.append(
            {
                "row": index,
                "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "token_ids": selected,
                "token_ids_sha256": hashlib.sha256(
                    json.dumps(selected, separators=(",", ":")).encode()
                ).hexdigest(),
            }
        )
        if len(samples) == SAMPLES:
            break
    assert len(samples) == SAMPLES
    corpus = {
        "dataset": record,
        "selection": "First 64 test rows with at least 64 tokens; first 64 tokens per row",
        "selection_before_execution": True,
        "samples": samples,
        "context_tokens": CONTEXT,
        "expected_scored_tokens": SAMPLES * (CONTEXT - 1),
        "tokenizer_sha256": digest(MODEL / "tokenizer.json"),
        "preparation_pyarrow_version": pyarrow.__version__,
    }
    target = HERE / "corpus-inputs.json"
    target.write_text(json.dumps(corpus, ensure_ascii=False, indent=2) + "\n")
    selection = {
        **{key: value for key, value in corpus.items() if key != "samples"},
        "corpus_sha256": digest(target),
        "samples": [
            {
                key: value
                for key, value in item.items()
                if key not in ("text", "token_ids")
            }
            for item in samples
        ],
    }
    (HERE / "corpus-selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(
        json.dumps(
            {
                "samples": SAMPLES,
                "scored_tokens": corpus["expected_scored_tokens"],
                "corpus_sha256": digest(target),
            }
        )
    )


if __name__ == "__main__":
    main()
