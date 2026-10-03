#!/usr/bin/env bash
## LLMs
# hf download google/medgemma-27b-it                      --local-dir  ../.cache/embeddings/medgemma-27b-it
# hf download ThaiLLM/ThaiLLM-8B                          --local-dir  ../.cache/embeddings/ThaiLLM-8B
# hf download google/gemma-3-270m-it                      --local-dir  ../.cache/embeddings/gemma-3-270m-it
# hf download google/gemma-3-4b-it                        --local-dir  ../.cache/embeddings/gemma-3-4b-it
# hf download google/gemma-3-12b-pt                       --local-dir  ../.cache/embeddings/gemma-3-12b-pt
# hf download google/gemma-3-27b-pt                       --local-dir  ../.cache/embeddings/gemma-3-27b-pt
# hf download google/gemma-3-27b-it                       --local-dir  ../.cache/embeddings/gemma-3-27b-it
# hf download google/medgemma-27b-it                      --local-dir  ../.cache/embeddings/medgemma-27b-it
# hf download Qwen/Qwen3-8B-Base                          --local-dir  ../.cache/embeddings/Qwen3-8B-Base
# hf download meta-llama/Llama-4-Scout-17B-16E-Instruct   --local-dir  ../.cache/embeddings/Llama-4-Scout-17B-16E-Instruct
# hf download Qwen/Qwen3-8B-Base                          --local-dir  ../.cache/embeddings/Qwen3-8B-Base
# hf download Qwen/Qwen3-0.6B                             --local-dir  ../.cache/embeddings/Qwen3-0.6B
# hf download ThaiLLM/ThaiLLM-8B                          --local-dir  ../.cache/embeddings/ThaiLLM-8B

## Embeddings
hf download Qwen/Qwen3-Embedding-0.6B                          --local-dir  ../.cache/embeddings/Qwen3-Embedding-0.6B

## Rerankers
hf download BAAI/bge-reranker-v2-m3                          --local-dir  ../.cache/embeddings/bge-reranker-v2-m3
