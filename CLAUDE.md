# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A GPT-style transformer language model with a byte-pair tokenizer in a single file (`tinyformer.cpp`) with **zero `#include` directives** — no libc headers, no STL. This is the project's defining constraint: math functions (`f_exp`, `f_log`, `f_sqrt`, `f_tanh`, `f_cos`) are hand-rolled with bit tricks and series expansions, memory comes from a bump allocator over one static float pool (`falloc`, no free), and the only external symbols are `printf`, `fflush` and raw `open`/`read`/`write`/`close`, forward-declared in an `extern "C"` block. Preserve this constraint when editing — do not add includes or link in library functions.

`tinyformer_gpu.cu` is the exception: the same model with ordinary includes and a CUDA training path (cuBLAS matmuls, simple one-thread-per-row/element kernels; sample and chat run on the host). It shares the checkpoint format, so checkpoints move freely between the two programs. It also builds with plain g++, where kernels run as serial loops via the `KERNEL`/`KID`/`LAUNCH` macros; that CPU build is the reference for checking the GPU build. Architecture or training changes to `tinyformer.cpp` must be mirrored there (its forward/backward is batched over `B*T` rows and keeps less per-layer state, so it is not a line-for-line copy).

## Commands

```sh
# build
g++ -O3 -march=native -ffast-math -o tinyformer tinyformer.cpp

# GPU build (NVIDIA, CUDA 11.5+; use e.g. -arch=sm_86 on older nvcc) and CPU reference build of the same file
nvcc -O3 -arch=native -o tinyformer_gpu tinyformer_gpu.cu -lcublas
g++ -O3 -march=native -x c++ -o tinyformer_gpu tinyformer_gpu.cu

# tinyformer_gpu takes the same modes and arguments, plus -b N (sequences per step; default 16 on GPU, 4 on CPU)

# pretrain (uses built-in corpus if no file given), saves tinyformer.ckpt
./tinyformer train [steps] [corpus.txt]

# continue pretraining the existing checkpoint (vocab stays fixed)
./tinyformer resume [steps] corpus.txt

# chat-tune an existing checkpoint on a dialog file
./tinyformer finetune [steps] dialog.txt

# build a pretrain corpus and finetune pairs from DailyDialog (needs pyarrow; see script header)
python make_dialog.py corpus.txt pairs.txt split.parquet [split.parquet ...]

# generate standalone user/bot exchanges with the DeepSeek API (stdlib only; appends to the JSONL, rewrites pairs)
DEEPSEEK_API_KEY=sk-... python make_deepseek.py chats.jsonl pairs.txt [batches] [min_balance]

# generate simple factual paragraphs with DeepSeek, one per line, as a resume corpus
DEEPSEEK_API_KEY=sk-... python make_facts.py facts.jsonl facts.txt [batches] [min_balance]

# free-running generation from the checkpoint
./tinyformer sample [nchars]

# interactive REPL
./tinyformer chat
```

Every mode reads and writes `tinyformer.ckpt` in the current directory (hard-coded `CKPT`); `train`, `resume` and `finetune` overwrite it in place, so copy it to a named file (e.g. `big-pretrain20000.ckpt`) before the next run. To use a named checkpoint, symlink it in a scratch dir: `cd "$(mktemp -d)" && ln -s ~/coding/tinyformer/X.ckpt tinyformer.ckpt && ~/coding/tinyformer/tinyformer chat`. For a training run, `cp` it instead — a symlink would let the run overwrite the named file. Input files are truncated at `CORPUS_MAX` (8 MB). Training is single-threaded CPU with no GPU path; long runs have been done on cheap RunPod CPU pods, logs kept in `logs/`.

## Data

All the data scripts share `make_dialog.py`'s `clean`/`ok` (lowercase, a fixed character set) and length caps (`MAX_USER`, `MAX_BOT`), and cap output at 8 MB. `finetune` trains on isolated user/bot pairs and `chat` keeps only a short prompt, so every pair must make sense with no earlier turn: `make_deepseek.py` asks for standalone exchanges for this reason. Both DeepSeek scripts check the account balance every 10 batches and stop below `min_balance` (in-flight batches still finish, so they overshoot slightly).

Current best chat model: `bpe-pretrain10000-ds10000-ft3000.ckpt` — `bpe-pretrain10000.ckpt` (DailyDialog) resumed 10k steps on `corpus_deepseek.txt` (DeepSeek exchanges + facts), then finetuned 3k steps on `dialog_clean.txt` (DeepSeek + other-agent pairs, no DailyDialog); finetune loss 2.62. Finetunes on the old base all stalled at 3.3–3.6 whatever the data, so the base model's pretraining is the bottleneck, not finetune data. DailyDialog-heavy finetune data teaches non-answers ("yes, i think i will").

There are no tests or linters. Verifying a change means rebuilding and running `train` (watch that loss falls) and/or `sample`/`chat`. To check `tinyformer_gpu.cu` against `tinyformer.cpp`, run both with `train 1` (`-b 4` for the GPU file) in separate scratch dirs on the same data: step-1 loss should match, and Adam's first update is ±lr·sign(grad), so diffing the two checkpoints per tensor exposes any wrong gradient (a few flips on near-zero grads such as key biases are float noise).

## Architecture

Everything lives in `tinyformer.cpp`, ordered roughly bottom-up: math → RNG → bump allocator → tokenizer/data → params → forward → backward → Adam → checkpoint → KV-cached inference → train loop → chat REPL → `main`.

- **Model config** is the `enum { T, C, L, H, ... }` block (context 128, width 256, 6 layers, 8 heads, vocab cap `MAXV` 1024). Checkpoints embed these values in their header and `load_ckpt_header` rejects a mismatch — changing the config invalidates `tinyformer.ckpt` and requires retraining. Growing the config may also require enlarging `POOL_FLOATS` (params ×4 for grads/Adam state, activations ×2, plus 2×`CORPUS_MAX` for data arrays must fit).
- **Params / grads / Adam state share one layout.** `alloc_params` is called twice (params `P`, grads `G`); because `falloc` hands out contiguous memory, the whole parameter block is addressed as one flat array from `P.wte` (`p_base`/`g_base`/`m_base`/`v_base`, length `NP`). Checkpoint save/load and `adam_step` rely on this contiguity, so any new parameter tensor must be allocated inside `alloc_params` and mirrored in `Acts`-style ordering.
- **Two forward paths exist and must stay in sync**: the batch path (`forward`, full `[T]` sequence, used for training and `sample`) and the incremental KV-cached path (`forward_one` + `ln_row`/`mat_row`, one token at a time, used by `chat`). An architectural change must be made in both.
- **Manual backprop**: `backward` mirrors `forward` step by step (`matmul_bwd`, `ln_bwd`, `attn_bwd`, GELU derivative). Any forward change needs a matching backward change.
- **Tokenizer** is byte-pair encoding over characters (lowercased, newlines → spaces). Vocab ids 0–2 are the special tokens (`TOK_USER`, `TOK_BOT`, `TOK_END`), then one id per character, then merged tokens, each defined as a pair (`g_left`, `g_right`). `train` learns merges until the vocab reaches `MAXV` (`bpe_learn`); every other path builds a char-level stream and replays the merges in id order (`bpe_encode`). Specials never merge, and a space can only start a token. Text is decoded with `tok_text`. The vocab and merge tables are stored in the checkpoint; old char-level checkpoints (magic `CKPT_CHAR`) still load with no merges.
- **Training schedule** (`train()`, shared by all training modes): no warmup; the learning rate follows a cosine decay from the base to 10% of it (1e-3 for `train`/`resume`, 3e-4 for `finetune`), and the checkpoint is saved every 2000 steps so a killed run keeps progress.
- **Finetuning is loss-masked**: `build_dialog_data` parses `dialog.txt` (non-empty lines alternate user, bot) into a token stream where `g_mask` is 1 only on bot text and the bot's end-of-turn token, so loss is computed only on bot replies. The chat REPL frames turns with the same special tokens.
