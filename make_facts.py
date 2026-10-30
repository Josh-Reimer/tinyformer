"""Generate simple factual paragraphs for tinyformer pretraining with the DeepSeek API.

Each request asks deepseek-chat for a batch of short explanatory paragraphs on
a random subject. Paragraphs are appended to a JSONL database, then cleaned
like make_dialog.py and written one per line as a `resume` corpus.

usage: DEEPSEEK_API_KEY=sk-... python make_facts.py facts.jsonl facts.txt [batches] [min_balance]
"""
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from make_deepseek import MODEL, WORKERS, balance, call
from make_dialog import clean, ok, write_capped

PER_BATCH = 15

SUBJECTS = """the sky and weather; the sun, moon and stars; planets; the ocean; rivers and lakes;
mountains and volcanoes; rocks and soil; plants and trees; flowers; farm animals; wild animals;
birds; fish; insects; dinosaurs; the human body; food and nutrition; cooking; health and sleep;
water and ice; light and color; sound and music; electricity; magnets; machines and tools;
cars and trains; airplanes; boats; computers; the internet; phones; numbers and counting;
shapes; time, days and calendars; money and shopping; maps and directions; countries and cities;
homes and buildings; jobs people do; school; sports and games; art; books and reading;
holidays and festivals; history of everyday inventions; seasons; energy and fuel;
recycling and the environment; feelings and friendship; manners; safety""".replace("\n", " ").split("; ")

PROMPT = """Write {n} different short paragraphs that explain simple facts about: {subject}.

Rules:
- Each paragraph covers one specific thing, in 3 to 5 short sentences of plain everyday English,
  like a friendly encyclopedia for a 10-year-old.
- Only well-known, correct facts. No made-up numbers.
- Lowercase, no emoji, no markdown, no lists, no names of real people or brands.

Reply with JSON only: {{"paragraphs": ["...", ...]}}"""


def batch(rng):
    subject = rng.choice(SUBJECTS)
    r = call("/chat/completions", {
        "model": MODEL, "temperature": 1.0, "max_tokens": 4096,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": PROMPT.format(n=PER_BATCH, subject=subject)}],
    })
    paras = json.loads(r["choices"][0]["message"]["content"])["paragraphs"]
    return [{"subject": subject, "text": str(p)} for p in paras], r["usage"]


def write_corpus(db_path, out_path):
    lines, seen = [], set()
    with open(db_path) as f:
        for line in f:
            text = clean(json.loads(line)["text"])
            if ok(text) and text not in seen:
                seen.add(text)
                lines.append(text + "\n")
    random.Random(0).shuffle(lines)
    n, size = write_capped(out_path, lines)
    print(f"facts: {n}/{len(lines)} paragraphs, {size} bytes -> {out_path}")


def main(db_path, out_path, batches="50", min_balance="0.50"):
    batches, min_balance = int(batches), float(min_balance)
    start = balance()
    print(f"balance ${start:.2f}, running {batches} batches")
    if start < min_balance:
        print("balance below min_balance, not generating")
    else:
        rng = random.Random()
        done = paras = tokens = 0
        with open(db_path, "a") as db, ThreadPoolExecutor(WORKERS) as pool:
            futures = [pool.submit(batch, random.Random(rng.random())) for _ in range(batches)]
            for fut in as_completed(futures):
                try:
                    out, usage = fut.result()
                except Exception as e:
                    print(f"batch failed: {e}")
                    continue
                for p in out:
                    db.write(json.dumps(p) + "\n")
                db.flush()
                done += 1
                paras += len(out)
                tokens += usage["total_tokens"]
                if done % 10 == 0:
                    left = balance()
                    print(f"{done}/{batches} batches, {paras} paragraphs, "
                          f"{tokens} tokens, balance ${left:.2f}")
                    if left < min_balance:
                        print("balance below min_balance, stopping")
                        for f in futures:
                            f.cancel()
                        break
        print(f"added {paras} paragraphs, spent ${start - balance():.2f}")
    write_corpus(db_path, out_path)


if __name__ == "__main__":
    main(*sys.argv[1:])
