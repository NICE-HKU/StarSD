#!/usr/bin/env python3
"""
Script to extract embedding layer and LM head from Vicuna-7B
for StarSD distributed speculative decoding.
"""

import torch
from transformers import AutoModelForCausalLM
import os
from eagle.model.config_loader import config

def extract_weights(model_name="lmsys/vicuna-7b-v1.3", output_dir="EAGLE-Vicuna-7B-v1.3"):
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    # 1. Extract embedding layer
    embedding_dir = os.path.join(output_dir, "embeddingLayer")
    os.makedirs(embedding_dir, exist_ok=True)
    
    embedding_weights = {"weight": model.model.embed_tokens.weight.data.clone().cpu()}
    torch.save(embedding_weights, os.path.join(embedding_dir, "vicuna7b_embeddings.bin"))
    print(f"✅ Saved embedding layer: {embedding_weights['weight'].shape}")
    
    # 2. Extract LM head
    lm_head_dir = os.path.join(output_dir, "vicuna-7B-v1.3-LMHead")
    os.makedirs(lm_head_dir, exist_ok=True)
    
    lm_head_weights = {"weight": model.lm_head.weight.data.clone().cpu()}
    torch.save(lm_head_weights, os.path.join(lm_head_dir, "lm_head.bin"))
    print(f"✅ Saved LM head: {lm_head_weights['weight'].shape}")
    
    # 3. Copy config.json for LM head initialization
    model.config.save_pretrained(lm_head_dir)
    print(f"✅ Saved config.json")
    
    print("\n🎉 Extraction complete!")

if __name__ == "__main__":
    target_model_dir = config.base_model_path
    draft_model_dir = config.draft_model_path
    extract_weights(model_name=target_model_dir, output_dir=draft_model_dir)