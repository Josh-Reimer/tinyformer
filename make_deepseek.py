"""Generate chat data for tinyformer with the DeepSeek API.

Each request asks deepseek-chat for a batch of standalone user/assistant
exchanges on a random topic (single exchanges, since finetune trains on
isolated pairs). Conversations are appended to a JSONL
database (one {"topic", "turns"} object per line), so the script can be
stopped and rerun to grow it. After generating, every user turn and the
assistant reply that follows it are cleaned like make_dialog.py and written
as a user line then a bot line, for `finetune`.

usage: DEEPSEEK_API_KEY=sk-... python make_deepseek.py chats.jsonl pairs.txt [batches] [min_balance]

batches      requests to make this run (default 50, ~40 exchanges each)
min_balance  stop early once the account balance in USD drops below this (default 0.50)
"""
import json
import os
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from make_dialog import MAX_BOT, MAX_USER, clean, ok, write_capped

API = "https://api.deepseek.com"
MODEL = "deepseek-chat"
WORKERS = 8
PER_BATCH = 40

TOPICS = """greetings and small talk; the weather; food and cooking; pets and animals;
school and homework; jobs and work; sports; music; movies and tv; books; travel;
family; friends; hobbies; video games; computers; the internet; phones; science;
space and planets; math; history; geography; health and exercise; sleep; feelings;
jokes and riddles; shopping; money; cars; the ocean; plants and gardening;
holidays; birthdays; the weekend; art and drawing; cities; the countryside;
seasons; colors; numbers and counting; time and clocks; asking for advice;
saying goodbye; asking the assistant about itself; simple how-to questions;
fun facts; would you rather questions; making plans; being bored""".replace("\n", " ").split("; ")

STYLES = ["casual and friendly", "curious, asking lots of questions", "short and blunt",
          "a kid", "polite and formal", "playful and joking", "a bit sad", "excited"]

PROMPT = """Write {n} different standalone exchanges between a user and a small, friendly chat assistant.
Topic: {topic}. The user's style: {style}.

Rules:
- Each exchange is one user message and the assistant's reply. The user message must make sense on its own,
  with no earlier context.
- Each message is one or two simple sentences, under 25 words. Plain everyday English.
- The assistant is warm, helpful and answers directly. It never says it is an AI language model.
- Lowercase, no emoji, no markdown, no lists, no names of real people or brands.
- Make the exchanges varied: different openings, questions and replies.

Reply with JSON only: {{"conversations": [["user message", "assistant reply"], ...]}}"""


def call(path, body=None):
    req = urllib.request.Request(
        API + path, data=json.dumps(body).encode() if body else None,
        headers={"Authorization": "Bearer " + os.environ["DEEPSEEK_API_KEY"],
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def balance():
    infos = call("/user/balance")["balance_infos"]
    return sum(float(b["total_balance"]) for b in infos if b["currency"] == "USD")


def batch(rng):
    topic, style = rng.choice(TOPICS), rng.choice(STYLES)
    r = call("/chat/completions", {
        "model": MODEL, "temperature": 1.0, "max_tokens": 4096,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": PROMPT.format(n=PER_BATCH, topic=topic, style=style)}],
    })
    convos = json.loads(r["choices"][0]["message"]["content"])["conversations"]
    out = [{"topic": topic, "turns": [str(t) for t in c]} for c in convos if isinstance(c, list)]
    return out, r["usage"]


def write_pairs(db_path, pairs_out):
    pairs, seen = [], set()
    with open(db_path) as f:
        for line in f:
            turns = [clean(t) for t in json.loads(line)["turns"]]
            for user, bot in zip(turns[0::2], turns[1::2]):
                if not ok(user) or not ok(bot) or len(user) > MAX_USER or len(bot) > MAX_BOT:
                    continue
                if (user, bot) not in seen:
                    seen.add((user, bot))
                    pairs.append(f"{user}\n{bot}\n")
    random.Random(0).shuffle(pairs)
    n, size = write_capped(pairs_out, pairs)
    print(f"pairs: {n}/{len(pairs)} pairs, {size} bytes -> {pairs_out}")


def main(db_path, pairs_out, batches="50", min_balance="0.50"):
    batches, min_balance = int(batches), float(min_balance)
    start = balance()
    print(f"balance ${start:.2f}, running {batches} batches")
    if start < min_balance:
        print("balance below min_balance, not generating")
    else:
        rng = random.Random()
        done = convos = tokens = 0
        with open(db_path, "a") as db, ThreadPoolExecutor(WORKERS) as pool:
            futures = [pool.submit(batch, random.Random(rng.random())) for _ in range(batches)]
            for fut in as_completed(futures):
                try:
                    out, usage = fut.result()
                except Exception as e:
                    print(f"batch failed: {e}")
                    continue
                for c in out:
                    db.write(json.dumps(c) + "\n")
                db.flush()
                done += 1
                convos += len(out)
                tokens += usage["total_tokens"]
                if done % 10 == 0:
                    left = balance()
                    print(f"{done}/{batches} batches, {convos} conversations, "
                          f"{tokens} tokens, balance ${left:.2f}")
                    if left < min_balance:
                        print("balance below min_balance, stopping")
                        for f in futures:
                            f.cancel()
                        break
        print(f"added {convos} conversations, spent ${start - balance():.2f}")
    write_pairs(db_path, pairs_out)


if __name__ == "__main__":
    main(*sys.argv[1:])
