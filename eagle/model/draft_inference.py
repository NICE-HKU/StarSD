import copy
import json
import time

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import os
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig
from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mixtral_kv import MixtralForCausalLM as KVMixtralForCausalLM
#from .modeling_qwen2_kv import LlamaForCausalLM as KVQwen2ForCausalLM
from .modeling_qwen2_kv import Qwen2ForCausalLM as KVQwen2ForCausalLM
from .utils_draft import *
from .kv_cache import initialize_past_key_values
from .communicator import FastDistributedCommunicator
from .cnets import Model
# from .cnets1 import Model as Model1
# from .cnets1_draft import Model as Model1
from .cnets1_draft import ModelDynamic
from .configs import EConfig
from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM
import numpy as np
from .config_loader import config as StarSD_config


# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_lm_head_only_model(saved_weight_path, original_model_path):
    with init_empty_weights():
        config = AutoConfig.from_pretrained(original_model_path)
        model = AutoModelForCausalLM.from_config(config)
    
    if saved_weight_path.endswith('.safetensors'):
        from safetensors import safe_open
        with safe_open(saved_weight_path, framework="pt") as f:
            weight = f.get_tensor("weight")
    else:
        weight = torch.load(saved_weight_path, map_location="cpu")["weight"]
    
    model.lm_head.weight = nn.Parameter(weight.half().to(StarSD_config.device_draft))
    
    return model

def dequantize(quant_tensor, min_val, max_val, target_device=None):
    scale = (max_val - min_val) / 255.0
    result = (quant_tensor.float() * scale + min_val).to(torch.float16)
    if target_device is not None:
        result = result.to(target_device)
    return result

def tuple_to_device(tuple_data, device):
    return tuple(tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor for tensor in tuple_data)

class EaModel(nn.Module):

    def __init__(
            self,
            use_eagle3,
            base_model,
            ea_model_path,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict,
            client
    ):

        super().__init__()
        self.base_model = base_model
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.client = client # communicator for distributed inference
        self.max_history = 2
        self.history = {
            'draft_tokens': [],
            'retrieve_indices': [],
            'tree_mask': [],
            'tree_position_ids': [],
            'new_token': []
        }
        self.depth = depth
        self.top_k = top_k
        self.accept_length_history_recent = []
        self.average_accept_ratio_history = []
        self.accept_ratio = []
        self.total_token_generate = []
        self.q_loss_history, self.actor_loss_history, self.reward_history, self.accept_ratio_history = [], [], [], []
        self.alpha_history = []
        self.generate_speed = []
        self.depth_history = []
        self.H_history = []
        print("Depth:", self.depth)
        print("Top K:", self.top_k)
        if not os.path.isdir(ea_model_path):
            raise ValueError(f"ea_model_path have to be local dir: {ea_model_path}")
            
        self.tokenizer = AutoTokenizer.from_pretrained(
            ea_model_path,
            use_fast=False,
            local_files_only=True
        )

        self.use_eagle3 = use_eagle3
        config = EConfig.from_pretrained(ea_model_path)
        config_path = os.path.join(ea_model_path, "config.json")
        with open(config_path, "r") as f:
            con = json.loads(f.read())
        try:
            bias = con["bias"]
        except:
            bias = True

        self.ea_layer = ModelDynamic(config, total_tokens=total_token, threshold=threshold, ea_model_path=ea_model_path).to(StarSD_config.device_draft)

        print("EA_LAYER loaded")
        low_memory = False

        if self.use_eagle3 and config.vocab_size==config.draft_vocab_size:
            del self.ea_layer.d2t,self.ea_layer.t2d
        load_=self.ea_layer.load_state_dict(ea_layer_state_dict, strict=False)

        self.ea_layer.init_tree(top_k=self.top_k)


    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer


    @classmethod
    def from_pretrained(
            cls,
            use_eagle3=True,
            ea_model_path=None,
            client=None,
            total_token=60,
            depth=None,
            top_k=None,
            threshold=1.0,
            **kwargs,
    ):
        # import base model's LM head as base model
        base_model = create_lm_head_only_model(saved_weight_path=os.path.join(ea_model_path, "vicuna-7B-v1.3-LMHead/lm_head.bin"), 
                                               original_model_path=os.path.join(ea_model_path, "vicuna-7B-v1.3-LMHead"))
        
        if os.path.isdir(ea_model_path):
            configpath = os.path.join(ea_model_path, "config.json")
            if not os.path.exists(configpath):
                raise FileNotFoundError(f"cannot find config: {configpath}")
            
            load_model_path = os.path.join(ea_model_path, "pytorch_model.bin")
            if os.path.exists(load_model_path):
                ea_layer_state_dict = torch.load(load_model_path, map_location=StarSD_config.device_draft)
                print("ea_layer_state_dict loaded")

            else:
                load_model_path = os.path.join(ea_model_path, "model.safetensors")
                if os.path.exists(load_model_path):
                    from safetensors.torch import load_file
                    ea_layer_state_dict = load_file(load_model_path)
                else:
                    raise FileNotFoundError(f"cannot find model file: {ea_model_path}")

        model = cls(
            use_eagle3=use_eagle3,
            base_model=base_model,
            ea_model_path=ea_model_path,
            total_token=total_token,
            depth=depth,
            top_k=top_k,
            threshold=threshold,
            ea_layer_state_dict=ea_layer_state_dict,
            client=client
        )

        if total_token == -1:
            device = model.base_model.model.layers[0].self_attn.q_proj.weight.device
            cans = [40, 48, 50, 56, 60]
            x = [1, 1.05, 1.07, 1.1, 1.13]
            times = []

            for i in range(len(cans)):
                length = cans[i]
                input_ids = torch.randint(0, model.config.vocab_size - 200, (1, length)).to(device)
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(20):
                    torch.cuda.synchronize()
                    with torch.no_grad():
                        outputs = model.base_model(input_ids)
                    torch.cuda.synchronize()
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) / x[i])
            total_token = cans[times.index(min(times))]
            model.ea_layer.total_tokens = total_token - 1
        return model

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
    ):
        print("FORWARD USED")
        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0]

        if output_orig:
            return outputs, orig, hidden_states
        else:
            return outputs, hidden_states
        
