import copy
import json
import time
import pickle
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
# from .utils import prepare_logits_processor, reset_tree_mode, tree_decoding, evaluate_posterior
from .kv_cache import initialize_past_key_values

from .cnets import Model
from .cnets1 import Model as Model1
from .configs import EConfig
from .utils_base import *
from .communicator import FastDistributedCommunicator


class EaModel(nn.Module):

    def __init__(
            self,
            use_eagle3,
            base_model,
            base_model_name_or_path,
            server,
            device
            # ea_model_path,
            # total_token,
            # depth,
            # top_k,
            # threshold,
            # ea_layer_state_dict,
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path, use_fast=False)
        self.use_eagle3 = use_eagle3
        self.server = server
        self.device = device
        self.total_iteration = 0
        self.accept_length_history = [1, 1, 1]  # Avoid going back to previous iteration at the beginning
        self.tolerance_of_zero_accept_length = 3
        self.history = {
            'accpt_hidden_state_new': [],
            'InputIds': [],
            'accept_length': [],
            'min_val': [],
            'max_val': []
        }
        print("Server started")
        # config = EConfig.from_pretrained(ea_model_path)
        # with open(ea_model_path, "r") as f:
        #     con = json.loads(f.read())
        # try:
        #     bias = con["bias"]
        # except:
        #     bias = True
        # if use_eagle3:
        #     self.ea_layer = Model(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
        #                           threshold=threshold, path=base_model_name_or_path,load_emb=True)
        # else:
        #     self.ea_layer = Model1(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
        #                           threshold=threshold, path=base_model_name_or_path,load_emb=True)

        low_memory = False

        # device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        # if device != base_model.lm_head.weight.device:
        #     self.ea_layer.diff_device = True
        #     if not low_memory:
        #         self.ea_layer.headweight = base_model.lm_head.weight.clone().to(device)
        #     else:
        #         self.ea_layer.layer_device = device

        # else:
        #     self.ea_layer.diff_device = False
        # if self.use_eagle3 and config.vocab_size==config.draft_vocab_size:
        #     del self.ea_layer.d2t,self.ea_layer.t2d
        # load_=self.ea_layer.load_state_dict(ea_layer_state_dict, strict=False)
        # self.ea_layer.to(self.base_model.dtype).to(device)
        # self.ea_layer.init_tree()

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod # the EaModel class is initialized by directly using this function
    def from_pretrained(
            cls,
            use_eagle3=True,
            base_model_path=None,
            # ea_model_path=None,
            total_token=60,
            # depth=7,
            # top_k=10,
            # threshold=1.0,
            **kwargs,
    ):
        # assert Type=="LLaMA" or "Mixtral"
        server_backup = kwargs.get("server", None)
        device = kwargs.get("device", None)
        if "server" in kwargs:
            del kwargs["server"]
        if "device" in kwargs:
            del kwargs["device"]

        Type = AutoConfig.from_pretrained(base_model_path).architectures[0]

        if Type == 'LlamaForCausalLM':
            base_model = KVLlamaForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type == 'Qwen2ForCausalLM':
            base_model = KVQwen2ForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        else:
            base_model = KVMixtralForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )

        # configpath = os.path.join(ea_model_path, "config.json")
        # if not os.path.exists(configpath):
        #     configpath = hf_hub_download(ea_model_path, "config.json")

        # try:
        #     load_model_path = os.path.join(ea_model_path, "pytorch_model.bin")
        #     if not os.path.exists(load_model_path):
        #         load_model_path = hf_hub_download(ea_model_path, "pytorch_model.bin")
        #     ea_layer_state_dict = torch.load(load_model_path,
        #                                      map_location=base_model.device)
        # except:
        #     from safetensors.torch import load_file
        #     load_model_path = os.path.join(ea_model_path, "model.safetensors")
        #     if not os.path.exists(load_model_path):
        #         load_model_path = hf_hub_download(ea_model_path, "model.safetensors")
        #     ea_layer_state_dict = load_file(load_model_path)
        model = cls(
            use_eagle3,
            base_model,
            base_model_path,
            server_backup,
            device
            # configpath,
            # total_token,
            # depth,
            # top_k,
            # threshold,
            # ea_layer_state_dict
        ).to(device)

        # if total_token == -1:
        #     device = model.base_model.model.layers[0].self_attn.q_proj.weight.device
        #     cans = [40, 48, 50, 56, 60]
        #     x = [1, 1.05, 1.07, 1.1, 1.13]
        #     times = []

        #     for i in range(len(cans)):
        #         length = cans[i]
        #         input_ids = torch.randint(0, model.config.vocab_size - 200, (1, length)).to(device)
        #         torch.cuda.synchronize()
        #         start_time = time.time()
        #         for _ in range(20):
        #             torch.cuda.synchronize()
        #             with torch.no_grad():
        #                 outputs = model.base_model(input_ids)
        #             torch.cuda.synchronize()
        #         torch.cuda.synchronize()
        #         end_time = time.time()
        #         times.append((end_time - start_time) / x[i])
        #     total_token = cans[times.index(min(times))]
        #     model.ea_layer.total_tokens = total_token - 1
        print("from_pretrained() work")
        print("MODEL:", model)
        return model

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
    ):

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

    @torch.no_grad()
    def eagenerate(
            self,
            input_ids,  # Now directly receives input_ids
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,
            client_tag=None,

    ):
        start_time = time.time()
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(self.device)
        # self.ea_layer.reset_kv()
        
        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        reset_tree_mode(self)
        print("tree reset")
        
        origin_input_ids = input_ids
        print("Processing input: ", origin_input_ids.shape)

        input_len = origin_input_ids.shape[1]
        new_token = 0
        if not hasattr(self, 'accept_length_history'):
            self.accept_length_history = [1, 1, 1]
        if not hasattr(self, 'history'):
            self.history = {'accpt_hidden_state_new': [], 'InputIds': [], 'accept_length': []}
        if not hasattr(self, 'tolerance_of_zero_accept_length'):
            self.tolerance_of_zero_accept_length = 3
        
        # Initialize transfer time statistics
        total_draft_to_base_transfer_time = 0.0  # Accumulated Draft to Base transfer time
        draft_to_base_transfer_time = 0.0
        draft_to_base_packet_size = 0.0
        
        # Send original input_ids to server (no timestamp needed, just forwarding input_ids)
        self.server.send_vars(origin_input_ids)
        if origin_input_ids == "close server": 
            return origin_input_ids, 0, origin_input_ids
        
        input_ids = origin_input_ids.clone()
        input_ids = input_ids.to(self.device)
        input_len = input_ids.shape[1]
        iteration_idx = -1
        # Prefill phase, both BASE MODEL and DRAFT MODEL involved
        # hidden_state_base, input_ids_base, lm_head, logits_processor_base = initialize_tree_base(  # need to be sent
        #     origin_input_ids, self, past_key_values, logits_processor)  # return hidden_states, input_ids, model.base_model.lm_head, logits_processor
        hidden_state, InputIds1, logits, token = initialize_tree_base(input_ids, self, past_key_values, logits_processor)
        # hidden_state, InputIds1 need to be sent
        
        # Send data with timestamp appended at the end of tuple
        send_packet_size = self.server.send_vars(hidden_state, InputIds1, iteration_idx, include_timestamp=True)
        
        # Receive data, get packet_size and RTT
        recv_data, recv_packet_size, recv_rtt = self.server.recv_vars(return_packet_size=True)
        recv_timestamp = time.time()
        
        if isinstance(recv_data, tuple) and len(recv_data) > 0 and isinstance(recv_data[-1], (int, float)):
            send_timestamp = recv_data[-1]
            init_base_to_draft_time = recv_timestamp - send_timestamp
            print(f"[Init] Base->Draft: {init_base_to_draft_time*1000:.2f}ms ({recv_packet_size:.2f}KB)")
            recv_data = recv_data[:-1]
        
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, ea_layer_total_tokens = tuple_to_device(recv_data, self.device)
        print("Get respond")

        max_length = max_length - ea_layer_total_tokens - 10
        self.total_iteration = 0
        total_base_inference_time = 0.0  # Accumulated Base model inference time
        iteration_start_time = time.time()
        
        # Main inference loop
        base_loop_start = time.time()  # Record Base model loop start time
        for idx in range(max_length):
            print("\nIteration: ", idx)
            self.total_iteration = self.total_iteration + 1
            iter_start_time = time.time()
            self.base_model.model.tree_mask = tree_mask
            

            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            
            # Verification: validate Draft tokens and calculate accept_length
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            
            # Update input states
            accpt_hidden_state_new, InputIds, input_ids, sample_token = update_inference_inputs_base(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                past_key_values_data,
                current_length_data,
                hidden_state_new,
                sample_p
            )
            
            base_send_time = time.time()
            base_inference_time = time.time() - iter_start_time
            print("Accept length: ", accept_length)

            # Buffer accept length history
            self.accept_length_history.append(accept_length)
            if len(self.accept_length_history) > self.tolerance_of_zero_accept_length:
                self.accept_length_history.pop(0)
            
            # Backup Base model generation state
            current_vars = {
                'accpt_hidden_state_new': accpt_hidden_state_new,
                'InputIds': InputIds,
                'accept_length': accept_length
            }
            for key, value in current_vars.items():
                copied_value = value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
                self.history[key].append(copied_value)
                if len(self.history[key]) > self.tolerance_of_zero_accept_length+1:
                    self.history[key].pop(0)

            iteration_idx = idx
            base_loop_time = time.time() - base_loop_start
            total_base_inference_time += base_loop_time
            
            # Debug: print sending data
            print(f"[Base] Sending: accept_length={accept_length}, idx={iteration_idx}, base_loop_time={base_loop_time:.4f}s")
            print(f"[Base] Metrics: draft_to_base_time={draft_to_base_transfer_time:.4f}s, draft_to_base_size={draft_to_base_packet_size:.2f}KB")
            print(f"[Base] accpt_hidden_state shape: {accpt_hidden_state_new.shape}, InputIds shape: {InputIds.shape}")
            
            # Send data with timestamp, draft_to_base_time, draft_to_base_size appended at the end
            send_packet_size = self.server.send_vars(accpt_hidden_state_new, InputIds, accept_length, 
                                                       iteration_idx, base_inference_time, base_loop_time,
                                                       draft_to_base_transfer_time, draft_to_base_packet_size,
                                                       include_timestamp=True)
            
            base_loop_start = time.time()
            # Receive data, get packet_size and RTT (Draft to Base transfer info)
            recv_data, draft_to_base_packet_size, draft_to_base_rtt = self.server.recv_vars(return_packet_size=True)
            receive_time = time.time()
            
            # Use TCP RTT as Draft to Base transfer time (not affected by clock synchronization)
            draft_to_base_transfer_time = draft_to_base_rtt / 1000.0  # Convert to seconds
            total_draft_to_base_transfer_time += draft_to_base_transfer_time
            print(f"Draft->Base: {draft_to_base_transfer_time*1000:.2f}ms ({draft_to_base_packet_size:.2f}KB) [TCP RTT]")
            
            # Check if timestamp exists (last element) and remove it
            if len(recv_data) > 0 and isinstance(recv_data[-1], (int, float)):
                recv_data = recv_data[:-1]
            
            # Parse received Draft data
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token = tuple_to_device(recv_data, self.device)

            print("Transfer time: {} sec".format(round(receive_time-base_send_time, 4)))
            print("Iteration time: {} sec".format(round(receive_time-iter_start_time, 4)))
            print("Base model inference time: {} sec".format(round(base_send_time-iter_start_time, 4)))
            


            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        end_time = time.time()
        inference_time = round(end_time - start_time, 4)
        total_iteration_time = end_time - iteration_start_time
        num_new_tokens = input_ids.shape[1] - input_len
        avg_speed = round(num_new_tokens / inference_time, 4)
        
        # Send end signal with accumulated transfer time and total Base model inference time
        print(f"\nTotal Draft->Base transfer time: {total_draft_to_base_transfer_time*1000:.2f}ms")
        print(f"Total Base model inference time: {total_base_inference_time*1000:.2f}ms")
        self.server.send_vars("end", input_ids, avg_speed, total_draft_to_base_transfer_time, total_base_inference_time, self.total_iteration/total_iteration_time, self.total_iteration)
        if not log:
            return input_ids, inference_time, origin_input_ids
        else:
            return input_ids, new_token, idx
        



def tuple_to_device(tuple_data, device):
    return tuple(tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor for tensor in tuple_data)