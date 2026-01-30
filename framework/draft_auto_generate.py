from eagle.model.draft_sac_inference import EaModel
from eagle.model.communicator import FastDistributedCommunicator
import torch
from fastchat.model import get_conversation_template
import json
import time
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.ndimage import gaussian_filter1d
from eagle.model.config_loader import config


def plot_sac_losses_reward(reward_list, 
                         save_folder=config.save_sac_figure_path, 
                         sigma=10, 
                         save_name='sac_reward.png'):
    """
    Simplified version that plots and saves only reward curves
    
    Args:
        reward_list: List of reward values
        save_folder: Folder path to save the plot
        sigma: Smoothing factor for Gaussian filter
        save_name: Filename for saved plot
    """
    os.makedirs(save_folder, exist_ok=True)
    
    rewards = np.array(reward_list)
    smoothed = gaussian_filter1d(rewards, sigma=sigma)
    
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, 'g-', alpha=0.3, label='Raw')
    plt.plot(smoothed, 'g-', linewidth=2, label='Smoothed')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.title(f'Reward (σ={sigma})')
    plt.grid(True)
    plt.legend()
    
    save_path = os.path.join(save_folder, save_name)
    plt.savefig(save_path)
    plt.close()
    print(f"Reward plot saved to: {save_path}")


def plot_sac_losses(*args, 
                    save_folder=config.save_sac_figure_path, 
                    sigma=3, 
                    save_name='sac_losses_individual.png',
                    labels=None):
    """
    Plot and save SAC training curves and automatically save the raw data.

    Args:
        *args: Variable metric lists (e.g., q_loss, actor_loss, reward, accept_ratio)
        save_folder: Folder to save plots
        sigma: Gaussian smoothing parameter
        save_name: Output image filename
        labels: Optional list of labels for each curve
    """
    os.makedirs(save_folder, exist_ok=True)
    
    data_dict = {}
    if labels is None:
        labels = [f'Metric_{i}' for i in range(len(args))]
    
    for i, (label, data) in enumerate(zip(labels, args)):
        data_dict[label] = np.array(data).tolist()  # convert to list for JSON serialization
    
    data_save_path = os.path.join(save_folder, save_name.replace('.png', '_data.json'))
    with open(data_save_path, 'w') as f:
        json.dump(data_dict, f, indent=4)
    
    n_metrics = len(args)
    if n_metrics == 0:
        print("warning: np input data")
        return
    
    n_cols = min(4, n_metrics)
    n_rows = int(np.ceil(n_metrics / n_cols))
    
    plt.figure(figsize=(5*n_cols, 4*n_rows))
    
    for i in range(n_metrics):
        plt.subplot(n_rows, n_cols, i+1)
        raw_data = np.array(args[i])
        
        current_sigma = sigma if len(raw_data) > 100 else sigma/2  
        smoothed = gaussian_filter1d(raw_data, sigma=current_sigma)
        
        plt.plot(raw_data, alpha=0.3, label='original')
        plt.plot(smoothed, linewidth=2, label='smoothed')
        
        plt.xlabel('training steps')
        plt.ylabel('value')
        plt.title(f'{labels[i]} (σ={current_sigma:.1f})')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()

    img_save_path = os.path.join(save_folder, save_name)
    plt.savefig(img_save_path)
    plt.close()
    
    print(f"Image saved to: {img_save_path}")
    print(f"Raw data saved to: {data_save_path}")


def reload_and_plot(data_path):
    with open(data_path) as f:
        data = json.load(f)
    
    # Extract labels and values
    labels = list(data.keys())
    values = [data[k] for k in labels]

    # Re-plot using saved data
    plot_sac_losses(*values, labels=labels, 
                   save_name='replot_' + os.path.basename(data_path).replace('_data.json', '.png'))




def get_question_by_id(question_id, json_file_path=""):
    """Retrieve a question from a JSONL file by question_id."""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if data['question_id'] == question_id:
                    return data
        return None
    except Exception as e:
        print(f"Error reading JSON file: {str(e)}")
        return None
    
def get_all_writing_questions(json_file_path=""):
    """Get all questions belonging to the 'writing' category."""
    writing_questions = []
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                # if data['category'] == 'writing' or data['category'] == 'roleplay' or data['category'] == 'reasoning':
                if config.full_mt_bench:
                    writing_questions.append(data)
                else:
                    if data['category'] == 'writing':
                        writing_questions.append(data)
        return sorted(writing_questions, key=lambda x: x['question_id'])
    except Exception as e:
        print(f"Error reading JSON file: {str(e)}")
        return []

import os
import json
from datetime import datetime



print("Finding server...")
client = FastDistributedCommunicator(host="", port=config.communication_port)
print("Communicator initialized")
model = EaModel.from_pretrained(
    ea_model_path=config.draft_model_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=False,
    use_eagle3=False,
    client=client,
    depth=7,
    top_k=10
)
print("Device distribution: make sure they are in the same device:")
print(f"1. base_model.lm_head: {model.base_model.lm_head.weight.device}")
print(f"2. ea_layer: {model.ea_layer.fc.weight.device}")
print(f"3. input_embedding_layer: {model.base_model.model.embed_tokens.weight.device}")
model.eval()

ues_llama_2_chat = False
use_vicuna = True
your_message = ""
print("Model initialized")

q_loss_history, actor_loss_history, idx_history = [], [], []

for i in range(config.mt_bench_epoch):
    print("\nOptions:")
    print("1. Enter question ID (e.g. 81)")
    print("2. Process ALL WRITING questions automatically")
    print("3. Enter custom message")
    print("4. 'quit' - exit client")
    print("5. 'close server' - shutdown server and client")
    
    user_input = "2"
    
    if user_input == "quit" or user_input == "4":
        client.close()
        break
    elif user_input == "close server" or user_input == "5":
        client.send_vars("close server")
        client.close()
        break
    elif user_input == "2":
        # Automatically process all writing questions
        writing_questions = get_all_writing_questions()
        if not writing_questions:
            print("No WRITING questions found!")
            continue
            
        print(f"\nFound {len(writing_questions)} WRITING questions, processing...")
        start_time = time.time()
        for question in writing_questions:
            qid = question['question_id']
            print(f"\n[Question ID: {qid}]")
            
            for turn_num, turn in enumerate(question['turns'], 1):
                print(f"\nTurn {turn_num}: {turn}")
                
                conv = get_conversation_template("vicuna")
                conv.append_message(conv.roles[0], turn)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                
                input_ids = torch.as_tensor(model.tokenizer([prompt]).input_ids).to(config.device_draft)
                output_ids = model.sac_eagenerate_train(input_ids, temperature=0.5, max_new_tokens=2048)
                output = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                # save_model_io(prompt, output)


                print(f"\nAnswer:\n{output}")
                # Automatically continue to the next turn without waiting for Enter
                if turn_num < len(question['turns']):
                    print("\n--- Automatically continuing to next turn ---")
            # break #--------------------
        print("\nFinished processing all WRITING questions!")
        print(f"Total time taken: {time.time() - start_time:.2f} seconds")
        # plot_sac_losses(model.q_loss_history, model.actor_loss_history, model.reward_history, model.generate_speed, 
        #                 save_name=f'sac_loss_individual{i}', labels=['q_loss', 'actor_loss', 'reward', 'iter_time', 'accept_ratio'])
        continue
    
    # 处理自定义消息
    your_message = user_input
    if use_vicuna:
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], your_message)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = torch.as_tensor(model.tokenizer([prompt]).input_ids).to(config.device_draft)
        output_ids = model.sac_eagenerate_train(input_ids, temperature=0.5, max_new_tokens=512)
        output = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"\nResponse:\n{output}")
    

if config.train_sac:
    plot_sac_losses(model.q_loss_history, model.actor_loss_history, model.reward_history, model.generate_speed, model.accept_ratio_history,
                    labels=['q_loss', 'actor_loss', 'reward', 'iter_time', 'accept_ratio'])

    plot_sac_losses_reward(model.reward_history)

if config.test:
    plot_sac_losses(model.accept_length_history_recent, model.depth_history, model.generate_speed, model.H_history,
                labels=['accept_length', 'depth', 'iter_time', 'accept_ratio'],
                save_folder=config.save_test_figure_path)


print(sum(model.accept_length_history_recent))
print(sum(model.generate_speed))
print("Average Channel Gain: ", sum(model.H_history)/len(model.H_history))
print(f"The generate speed: {sum(model.accept_length_history_recent)/sum(model.generate_speed)}")