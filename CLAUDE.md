# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A GPT-style character-level transformer language model in a single file (`tinyformer.cpp`) with **zero `#include` directives** — no libc headers, no STL. This is the project's defining constraint: math functions (`f_exp`, `f_log`, `f_sqrt`, `f_tanh`) are hand-rolled with bit tricks and series expansions, memory comes from a bump allocator over one static float pool (`falloc`, no free), and the only external symbols are `printf` and raw `open`/`read`/`write`/`close`, forward-declared in an `extern "C"` block. Preserve this constraint when editing — do not add includes or link in library functions.

## Commands

```sh
# build
g++ -O3 -march=native -ffast-math -o tinyformer tinyformer.cpp

# pretrain (uses built-in corpus if no file given), saves tinyformer.ckpt
./tinyformer train [steps] [corpus.txt]

# chat-tune an existing checkpoint on a dialog file
./tinyformer finetune [steps] dialog.txt

# free-running generation from the checkpoint
./tinyformer sample [nchars]

# interactive REPL
./tinyformer chat
```

There are no tests or linters. Verifying a change means rebuilding and running `train` (watch that loss falls) and/or `sample`/`chat`.

## Architecture

Everything lives in `tinyformer.cpp`, ordered roughly bottom-up: math → RNG → bump allocator → tokenizer/data → params → forward → backward → Adam → checkpoint → KV-cached inference → train loop → chat REPL → `main`.

- **Model config** is the `enum { T, C, L, H, ... }` block (context 128, width 128, 4 layers, 8 heads). Checkpoints embed these values in their header and `load_ckpt_header` rejects a mismatch — changing the config invalidates `tinyformer.ckpt` and requires retraining. Growing the config may also require enlarging `POOL_FLOATS` (params ×4 for grads/Adam state, activations ×2, plus 2×`CORPUS_MAX` for data arrays must fit).
- **Params / grads / Adam state share one layout.** `alloc_params` is called twice (params `P`, grads `G`); because `falloc` hands out contiguous memory, the whole parameter block is addressed as one flat array from `P.wte` (`p_base`/`g_base`/`m_base`/`v_base`, length `NP`). Checkpoint save/load and `adam_step` rely on this contiguity, so any new parameter tensor must be allocated inside `alloc_params` and mirrored in `Acts`-style ordering.
- **Two forward paths exist and must stay in sync**: the batch path (`forward`, full `[T]` sequence, used for training and `sample`) and the incremental KV-cached path (`forward_one` + `ln_row`/`mat_row`, one token at a time, used by `chat`). An architectural change must be made in both.
- **Manual backprop**: `backward` mirrors `forward` step by step (`matmul_bwd`, `ln_bwd`, `attn_bwd`, GELU derivative). Any forward change needs a matching backward change.
- **Tokenizer** is character-level (lowercased, newlines → spaces) plus three special tokens (`TOK_USER`, `TOK_BOT`, `TOK_END`) occupying vocab ids 0–2. The vocab is built from the training corpus and stored in the checkpoint.
- **Finetuning is loss-masked**: `build_dialog_data` parses `dialog.txt` (non-empty lines alternate user, bot) into a token stream where `g_mask` is 1 only on bot text and the bot's end-of-turn token, so loss is computed only on bot replies. The chat REPL frames turns with the same special tokens.
