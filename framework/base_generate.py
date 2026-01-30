from eagle.model.base_model_inference import EaModel
from eagle.model.communicator import FastDistributedCommunicator
import torch
from fastchat.model import get_conversation_template
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from device import DEVICE
from eagle.model.config_loader import config

average_accept_length = []
average_generate_speed = []
total_token_generated = []
device = "cuda:3"
server = FastDistributedCommunicator(host='0.0.0.0', port=config.communication_port, is_server=True, buffer_size=8192)
print("Connect with client")
model = EaModel.from_pretrained(
    base_model_path=config.base_model_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=False,
    use_eagle3=False,
    server=server,
    device=config.device_base
).to(config.device_base)

print("Device distribution: make sure they are in the same device")
print(f"1. base_model.lm_head: {model.base_model.lm_head.weight.device}")
print(f"2. input_embedding layer: {model.base_model.model.embed_tokens.weight.device}")
model.eval()

# Handle user input
ues_llama_2_chat = False
use_vicuna = True
print("Base model initialized and ready for user input")

while True:
    print("\nEnter 'quit' to exit server, 'close server' to shutdown completely")
    your_message = input("Your message: ")
    
    if your_message == "quit":
        server.close()
        break
    elif your_message == "close server":
        server.send_vars("close server")
        server.close()
        break

    # Prepare conversation template
    if ues_llama_2_chat:
        conv = get_conversation_template("llama-2-chat")  
        sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
        conv.system_message = sys_p
        conv.append_message(conv.roles[0], your_message)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt() + " "

    if use_vicuna:
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], your_message)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

    # Tokenize user input
    input_ids = model.tokenizer([prompt]).input_ids
    input_ids = torch.as_tensor(input_ids).to(config.device_base)
    
    # Generate reply
    output_ids, inference_time, original_input_ids = model.eagenerate(
        input_ids=input_ids,
        temperature=0.5,
        max_new_tokens=2048
    )
    
    if inference_time == 0:
        print("Server close")
        break
        
    # Decode only the newly generated part
    generated_ids = output_ids[:, input_ids.shape[1]:]
    output = model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    num_new_tokens = output_ids.shape[1] - input_ids.shape[1]

    print("\n\nGenerated response:\n", output)
    print("\nInference time: {}s".format(inference_time))
    print("{} new tokens generated".format(num_new_tokens))
    print("{} tokens are generated per second".format(num_new_tokens/inference_time))
    print("Average accept length: {}".format(num_new_tokens/model.total_iteration))
    average_generate_speed.append(round(num_new_tokens/inference_time, 4))
    average_accept_length.append(round(num_new_tokens/model.total_iteration, 4))
    total_token_generated.append(num_new_tokens)
    print("average_generate_speed: ", average_generate_speed)
    print("average_accept_length: ", average_accept_length)
    print("total_token_generated: ", total_token_generated)

print("average_generate_speed: {}".format(round(sum(average_generate_speed)/len(average_generate_speed), 4)))
print("average_accept_length: {}".format(round(sum(average_accept_length)/len(average_accept_length), 4)))
print("total_token_generated: {}".format(sum(total_token_generated)))
print("Amount of data from Base model to Draft model: {}KB".format(round(server.total_data_send, 4)))
print("Amount of data from Draft model to Base model: {}KB".format(round(server.total_data_receive, 4)))