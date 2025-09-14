# LLMc

LLMc is a LLM-based compression tool that uses LLM to compress natural language text.

## Installation

We recommend using uv to manage the virtual environment.

```bash
uv venv -p 3.11
# We use a modified version of vLLM and batch_invariant_ops as backend.
git submodule update --init --recursive
export VLLM_USE_PRECOMPILED=1
uv sync
```

## Usage

Supported features:
- [x] CLI compression
  - Run `llmc compress <input_file> <output_file> --model <model_name>` to compress the input text file.
- [x] CLI decompression
  - Run `llmc decompress <input_file> <output_file> --model <model_name>` to decompress the input text file.
- [ ] Web frontend and API
  - Not well tested.

# Preliminary Results

We use `Qwen/Qwen3-0.6B` as the model with chunk size 4096. The test files are from [brotli](https://github.com/google/brotli).

| filename | original | brotli | ours |
|----------|----------|--------|------|
| alice29.txt | 152089 | 50096 | 34003 |
| asyoulik.txt | 125179 | 45687 | 38233 |
| lcet10.txt | 419235 | 124719 | 84505 |
| plrabn12.txt | 481861 | 174771 | 155255 |