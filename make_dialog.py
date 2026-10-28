"""Build tinyformer training files from DailyDialog.

Writes two files:
  corpus  every conversation, one utterance per line, for `train` / `resume`
  pairs   each pair of consecutive turns as a user line then a bot line,
          for `finetune`; pairs too long for the model's context are skipped

Text is lowercased, detokenized and limited to a fixed character set.

usage: python make_dialog.py corpus.txt pairs.txt split.parquet [split.parquet ...]

splits: https://huggingface.co/api/datasets/roskoN/dailydialog/parquet/full/{train,validation,test}/0.parquet
needs: pip install pyarrow
"""
import random
import re
import sys

import pyarrow.parquet as pq

MAX_USER = 180         # chat keeps the last 64 prompt tokens (~3 chars each with BPE)
MAX_BOT = 200          # reply must fit in the remaining 128-token context
MAX_BYTES = 8_000_000  # tinyformer reads at most CORPUS_MAX = 8388608 bytes
ALLOWED = set("abcdefghijklmnopqrstuvwxyz0123456789 .,!?'\"-:;()$%&/")

SWAPS = {"’": "'", "‘": "'", "“": '"', "”": '"',
         "—": "-", "–": "-", "…": "..."}


def clean(s):
    for a, b in SWAPS.items():
        s = s.replace(a, b)
    s = s.lower().strip()
    s = re.sub(r"\s*'\s*(s|m|re|ve|ll|d|t)\b", r"'\1", s)  # it ' s -> it's
    s = re.sub(r"\s+n't\b", "n't", s)
    s = re.sub(r"\s+([,.!?;:%)])", r"\1", s)                # word , -> word,
    s = re.sub(r"([($])\s+", r"\1", s)
    return re.sub(r"\s+", " ", s)


def ok(s):
    return s and all(c in ALLOWED for c in s)


def write_capped(path, chunks):
    size = 0
    with open(path, "w", encoding="ascii") as f:
        for n, chunk in enumerate(chunks):
            if size + len(chunk) > MAX_BYTES:
                break
            f.write(chunk)
            size += len(chunk)
        else:
            n = len(chunks)
    return n, size


def main(corpus_out, pairs_out, *parquets):
    convos = []
    for path in parquets:
        for row in pq.read_table(path, columns=["utterances"]).to_pylist():
            convos.append([clean(u) for u in row["utterances"]])

    # identical conversations recur across splits; keep one copy
    convos = list({tuple(c): c for c in convos}.values())
    random.Random(0).shuffle(convos)

    lines = ["\n".join(u for u in c if ok(u)) + "\n" for c in convos]
    n, size = write_capped(corpus_out, lines)
    print(f"corpus: {n}/{len(lines)} conversations, {size} bytes -> {corpus_out}")

    pairs, seen = [], set()
    for turns in convos:
        for user, bot in zip(turns, turns[1:]):
            if not ok(user) or not ok(bot) or len(user) > MAX_USER or len(bot) > MAX_BOT:
                continue
            if (user, bot) not in seen:
                seen.add((user, bot))
                pairs.append(f"{user}\n{bot}\n")
    n, size = write_capped(pairs_out, pairs)
    print(f"pairs: {n}/{len(pairs)} pairs, {size} bytes -> {pairs_out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
