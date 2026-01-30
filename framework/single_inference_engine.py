import threading
import time
import torch
import os
import numpy as np
from typing import Optional
import queue
import traceback
from typing import List, Tuple
from datetime import datetime

from framework.global_data_manager import global_data_manager
from eagle.model.draft_inference import EaModel
from eagle.model.utils_draft import initialize_tree_draft, update_inference_inputs_draft, update_inference_inputs_draft_wramup
from eagle.model.config_loader import config
from framework.experiment_data_logger import initialize_experiment_logger, log_system_information, get_experiment_path
import subprocess
import pynvml
import gc
gc.disable()


_gpu_handle = None
_pynvml_initialized = False

def init_gpu_monitor():
    """initialize pynvml for GPU monitoring"""
    global _gpu_handle, _pynvml_initialized
    if not _pynvml_initialized:
        try:
            pynvml.nvmlInit()
            _gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            _pynvml_initialized = True
            print("✅ pynvml initialized for GPU monitoring")
        except Exception as e:
            print(f"⚠️ Failed to initialize pynvml: {e}, returning zero values")
            _pynvml_initialized = False

def get_gpu_metrics():
    """Get GPU power and frequency metrics """
    global _gpu_handle, _pynvml_initialized
    
    # Ensure initialized
    if not _pynvml_initialized:
        init_gpu_monitor()
    
    # If initialization failed, return zero values
    if not _pynvml_initialized or _gpu_handle is None:
        return {'gpu_freq_mhz': 0.0, 'gpu_power_w': 0.0, 'gpu_util_percent': 0.0, 'gpu_temp_c': 0.0}
    
    try:
        freq = pynvml.nvmlDeviceGetClockInfo(_gpu_handle, pynvml.NVML_CLOCK_GRAPHICS)
        power = pynvml.nvmlDeviceGetPowerUsage(_gpu_handle) / 1000.0  # mW -> W
        util = pynvml.nvmlDeviceGetUtilizationRates(_gpu_handle).gpu
        temp = pynvml.nvmlDeviceGetTemperature(_gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
        return {
            'gpu_freq_mhz': float(freq),
            'gpu_power_w': float(power),
            'gpu_util_percent': float(util),
            'gpu_temp_c': float(temp)
        }
    except Exception as e:
        # if any error occurs, return zero values
        return {'gpu_freq_mhz': 0.0, 'gpu_power_w': 0.0, 'gpu_util_percent': 0.0, 'gpu_temp_c': 0.0}


def dequantize(quant_tensor, min_val, max_val, target_device=None):
    """dequantize tensor from uint8 to float16"""
    scale = (max_val - min_val) / 255.0
    result = (quant_tensor.float() * scale + min_val).to(torch.float16)
    if target_device is not None:
        result = result.to(target_device)
    return result


class EnvironmentTracker:
    """RL environment data tracker"""
    
    def __init__(self, max_history_length: int = 300):
        # client environment data {client_tag: client_data}
        self.client_data = {}
        self.max_history_length = max_history_length
        self.pilot_iterations = 5  # Number of pilot iterations before computing prompt difficulty
    
    def get_or_create_client_data(self, client_tag: str):
        """Acquire or create client data"""
        if client_tag not in self.client_data:
            self.client_data[client_tag] = {
                'accept_length_history': [],
                'accept_ratio_history': [],
                'threshold_history': [],
                'depth_history': [],
                'accept_length': 0,
                'accept_ratio': 0.0,
                'total_tasks': 0,
                'last_update_time': time.time(),
                'prompt_difficulty': 1.0,
                'net_gain_history': [],
                'total_cost_history': []
            }
        return self.client_data[client_tag]

    def get_client_data(self):
        return self.client_data
    
    def update_client_metrics(self, client_tag: str, accept_length: int, last_depth: int, threshold: float, iteration: int):
        """Update client metrics"""
        client_data = self.get_or_create_client_data(client_tag)

        # Correct accept ratio calculation: based on the last depth of the same client
        accept_ratio = accept_length / max(last_depth, 1)
        
        # update current values
        client_data['accept_length'] = accept_length
        client_data['accept_ratio'] = accept_ratio
        client_data['total_tasks'] += 1
        client_data['last_update_time'] = time.time()
        
        # update history (based on the same client's history)
        client_data['accept_length_history'].append(accept_length)
        client_data['accept_ratio_history'].append(accept_ratio)
        client_data['depth_history'].append(last_depth)
        client_data['threshold_history'].append(threshold)

        if iteration < self.pilot_iterations:
            client_data['prompt_difficulty'] = 1.0
        elif iteration == self.pilot_iterations:
            client_data['prompt_difficulty'] = compute_prompt_difficulty(
                client_data['depth_history'], client_data['accept_length_history'])
        
        # sliding window
        if len(client_data['accept_length_history']) > self.max_history_length:
            client_data['accept_length_history'].pop(0)
            client_data['accept_ratio_history'].pop(0)
            client_data['depth_history'].pop(0)
            client_data['threshold_history'].pop(0)
        
        if len(client_data['net_gain_history']) > self.max_history_length: 
            client_data['net_gain_history'].pop(0)
            client_data['total_cost_history'].pop(0)

    def get_active_clients_count(self) -> int:
        """Get the number of active clients (those who appeared in the global queue within the last 5 seconds)"""
        current_time = time.time()
        active_count = 0
        for client_data in self.client_data.values():
            if current_time - client_data['last_update_time'] < 1:  # Active within 1 second
                active_count += 1
        return active_count
    
    def reset_client_session(self, client_tag: str):
        """Reset client session data"""
        if client_tag in self.client_data:
            client_data = self.client_data[client_tag]
            client_data['accept_length_history'].clear()
            client_data['accept_ratio_history'].clear()
            client_data['session_start_time'] = time.time()
            client_data['total_tasks'] = 0
            client_data['depth_history'].clear()
            client_data['threshold_history'].clear()
            client_data['prompt_difficulty'] = 1.0
            client_data['net_gain_history'].clear()
            client_data['total_cost_history'].clear()


class AsyncDataLogger:
    """Record system information asynchronously"""
    
    def __init__(self, experiment_logger):
        self.experiment_logger = experiment_logger  # Use the passed logger to avoid re-initialization
        self.logging_queue = queue.Queue(maxsize=200)
        self.running = False
        self.thread = None
        
    def start(self):
        """Start the logging thread"""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._logging_loop, daemon=True, name="AsyncDataLogger")
        self.thread.start()
        print("📊 Async data logger started")
    
    def stop(self):
        """Stop the logging thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        print("📊 Async data logger stopped")
    
    def submit_system_log(self, **kwargs):
        """Submit system log data to the queue"""
        try:
            log_data = {
                'type': 'system_info',
                'data': kwargs,
                'timestamp': time.time()
            }
            self.logging_queue.put_nowait(log_data)
        except queue.Full:
            print("⚠️ Logging queue full, skipping this log")
    
    def _logging_loop(self):
        """Recording loop"""
        while self.running:
            try:
                # acquire log data
                log_data = self.logging_queue.get(timeout=1.0)
                self._process_logging(log_data)
                self.logging_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error in logging loop: {e}")
                continue
    
    def _process_logging(self, log_data):
        """Handle single log entry"""
        try:
            if log_data['type'] == 'system_info':
                log_system_information(**log_data['data'])
        except Exception as e:
            print(f"❌ Error processing log: {e}")


class SingleInferenceEngine:
    """Single inference engine - Listens to the global dictionary and processes all client inference tasks"""

    def __init__(self, draft_model: EaModel):
        self.draft_model = draft_model
        self.running = False
        self.inference_thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()

        # 🧠 Environment tracker
        self.environment_tracker = EnvironmentTracker()

        # 🧠 Experiment data logger
        self.experiment_logger = initialize_experiment_logger()
        print("📊 Experiment data logger initialized")
        
        # 🕐 GPU monitor
        init_gpu_monitor()
        
        # 🧠 RL training state management {client_tag: {'last_state', 'last_action', 'last_depth', 'step_count'}}
        self.rl_client_states = {}
        
        # 🧠 RL training history
        self.q_loss_history = []
        self.actor_loss_history = []
        self.reward_history = []
        self.accept_ratio_history = []
        self.alpha_history = []
        self.time_stamp = {} # client_tag: last update time, to calculate iteration time
        self.service_history = [] # store client_tag that have been served
        self.system_throughput = []  # store system throughput history
        self.total_depth = 0 # store total depth in one iteration (same iteration as throughput calculation)

        self.period_duration = 0.2  # duration for throughput calculation
        self.accept_length_with_timestamp = []  # store (timestamp, accept_length) tuples for throughput calculation
        self.depth_with_timestamp = []  # store (timestamp, depth) tuples for reward calculation
        self.throughput_in_duration = []  # latest calculated throughput
        self.total_depth_in_duration = 0  # total depth in the current duration for reward calculation
        self.idle_time_recording = []  # store (timestamp, idle_time) tuples for idle ratio calculation
        self.work_time_recording = []  # store (timestamp, work_time) tuples for idle ratio calculation
        self.step_count = 0
        self.task_waiting_time_with_timestamp = []  # store (timestamp, waiting_time) tuples for task waiting time in global queue
        self.avg_speed_per_prompt = 0.0 # average speed (tokens/sec) per prompt
        self.generate_iteration_count = 0 # real depth after using entropy threshold
        self.iteration_with_timestamp = []  # store (timestamp, 1) tuples for iteration speed calculation
        self.iteration_speed_in_duration = []  # latest calculated iteration speed

        # 🕐 Runtime statistics
        self.total_draft_inference_time = 0.0  # Small model total inference time
        self.total_base_inference_time = 0.0   # Large model total inference time
        self.iteration_count = 0  # Number of iterations
        self.total_loop_time = 0.0  # Total loop time (from time_to_start_work to put_inference_result)
        self.total_inference_time = 0.0  # Total inference time (inference part)
        self.total_initialize_time = 0.0  # Total initialization time
        self.initialize_count = 0  # Number of initializations

        # 🚀 Asynchronous data logger
        self.async_data_logger = AsyncDataLogger(self.experiment_logger)
        print("🚀 Async processors initialized")

    def start(self):
        """Start the inference engine"""
        with self.lock:
            if self.running:
                print("⚠️ Inference engine already running")
                return
                
            self.running = True
            
            # Start data logger
            self.async_data_logger.start()
            
            # Start inference thread
            self.inference_thread = threading.Thread(
                target=self._inference_loop,
                daemon=True,
                name="SingleInferenceEngine"
            )
            self.inference_thread.start()
            print("🚀 Single inference engine started")
    
    def stop(self):
        """Stop the inference engine"""
        with self.lock:
            self.running = False

            # Stop data logger
            self.async_data_logger.stop()
            
            # Stop inference thread
            if self.inference_thread and self.inference_thread.is_alive():
                self.inference_thread.join(timeout=5)
            
            print("🛑 Single inference engine stopped")
    
    def _inference_loop(self):
        """Main Loop - Listen to the global dictionary and handle tasks"""
        print("🔄 Inference loop started")
        client_tag = None
        inference_start_time = time.time()
        iteration_time = 0.0
        iteration_start_time = time.time()
        abnormal_inference_time = []

        while self.running:
            try:
                # Check for pending tasks
                if not global_data_manager.has_pending_tasks():
                    time.sleep(0.0001)  # Short sleep to avoid CPU overuse
                    continue

                # Get the next task
                task_info = global_data_manager.get_next_inference_task()
                if task_info is None:
                    continue
                print("\nTask type: ", task_info[1].get('type', 'unknown'))
                # time.sleep(0.04)  # Short sleep to avoid CPU overuse

                time_to_get_task = time.time() - iteration_start_time  # Idle time (waiting for task)
                time_to_start_work = time.time()  # 🕐 Record start work time
                
                # Record this idle time
                self.idle_time_recording.append((time.time(), time_to_get_task))
                
                # Calculate idle_ratio: proportion of idle time within the period_duration window
                cleanup_timestamped_list(self.idle_time_recording, self.period_duration)
                cleanup_timestamped_list(self.work_time_recording, self.period_duration)
                total_idle_time = sum(idle_time for _, idle_time in self.idle_time_recording)
                total_work_time = sum(work_time for _, work_time in self.work_time_recording)
                total_time = total_idle_time + total_work_time
                idle_ratio = total_idle_time / max(total_time, 1e-5) if total_time > 0 else 0.0
                
                if iteration_time > 0 and client_tag is not None: # last client tag
                    self.time_stamp[client_tag] = iteration_time # record last iteration time

                client_tag, task_data = task_info

                self.task_waiting_time_with_timestamp.append((time.time(), time.time() - task_data['task_submit_time']))
                avg_task_waiting_time = sum(wait_time for _, wait_time in self.task_waiting_time_with_timestamp) / max(len(self.task_waiting_time_with_timestamp), 1)  # calculate average waiting time

                if client_tag not in self.rl_client_states:
                    self.rl_client_states[client_tag] = {'step_count': 0}
                client_rl_state = self.rl_client_states[client_tag] # {client_tag: {'last_state', 'last_action', 'last_depth', 'step_count'}}

                if 'accept_length' in task_data and 'last_depth' in client_rl_state:
                    self.environment_tracker.update_client_metrics(client_tag, task_data['accept_length'], client_rl_state["last_depth"], 
                                                          client_rl_state['last_threshold'], task_data.get('iteration', 0)) # update based on accept length and last depth of the same client
                    self.accept_length_with_timestamp.append((time.time(), task_data['accept_length'])) # useful in throughput calculation
                    self.depth_with_timestamp.append((time.time(), client_rl_state["last_depth"])) # useful in reward calculation
                
                # 🔥 Record iteration for iteration speed calculation
                self.iteration_with_timestamp.append((time.time(), 1))
                
                # Clean up old records beyond period_duration
                cleanup_timestamped_list(self.accept_length_with_timestamp, self.period_duration)
                cleanup_timestamped_list(self.depth_with_timestamp, self.period_duration)
                cleanup_timestamped_list(self.task_waiting_time_with_timestamp, self.period_duration)
                cleanup_timestamped_list(self.iteration_with_timestamp, self.period_duration)

                accept_length_sum = sum(length for _, length in self.accept_length_with_timestamp) # total accept length in the period duration
                self.total_depth_in_duration = sum(depth for _, depth in self.depth_with_timestamp) # total depth in the period duration
                self.throughput_in_duration.append(accept_length_sum / max(self.period_duration, 1e-5)) # throughput = total accept length / period duration
                self.throughput_in_duration = self.throughput_in_duration[-20:]
                
                # 🔥 Calculate iteration speed (iterations/sec)
                iteration_count_in_window = sum(count for _, count in self.iteration_with_timestamp)
                current_iteration_speed = iteration_count_in_window / max(self.period_duration, 1e-5)
                self.iteration_speed_in_duration.append(current_iteration_speed)
                self.iteration_speed_in_duration = self.iteration_speed_in_duration[-20:]

                # between tags for logging (what clients have been served between last and this)
                result = get_between_tags(self.service_history, client_tag)
                between_tags = [tag for tag, _ in result] if result is not None else []

                # get active clients in the period duration
                active_client_in_duration = get_recent_unique_tags(self.service_history, self.period_duration)
                if client_tag in active_client_in_duration: active_client_in_duration.remove(client_tag)  # [tag1, tag2, ...]
                avg_accept_ratio_last_1, avg_accept_ratio_last_5 = get_accept_ratio_from_tags(active_client_in_duration, self.environment_tracker.get_client_data()) # Compute accept ratios for active clients excluding current client

                # record service history to get corresponding iteration time and accept length
                try:
                    client_duration = time.time() - next((timestamp for tag, timestamp in reversed(self.service_history) if tag == client_tag), 100)
                except:
                    client_duration = 100
                    
                self.service_history.append((client_tag, time.time()))
                if len(self.service_history) > 1000: self.service_history.pop(0) # keep history manageable

                # Get pending task count
                pending_task_num = global_data_manager.get_pending_tasks_count()

                # Fixed threshold and depth
                threshold = 1000  # Default threshold for early stop
                dynamic_depth = 5
                client_rl_state['last_threshold'] = threshold

                start_log_time = time.time()

                try:
                    self.async_data_logger.submit_system_log(
                        num_active_clients=len(active_client_in_duration)+1,
                        total_clients=pending_task_num,
                        system_throughput=self.throughput_in_duration[-1] if self.throughput_in_duration else 0.0,
                        avg_system_throughput=sum(self.throughput_in_duration[-10:]) / max(len(self.throughput_in_duration[-10:]), 1) if self.throughput_in_duration else 0.0,
                        iteration_speed=self.iteration_speed_in_duration[-1] if self.iteration_speed_in_duration else 0.0,
                        avg_iteration_speed=sum(self.iteration_speed_in_duration[-10:]) / max(len(self.iteration_speed_in_duration[-10:]), 1) if self.iteration_speed_in_duration else 0.0,
                        buffer_size=0,
                        dynamic_depth=dynamic_depth,
                        client_tag=client_tag,
                        total_accept_length=accept_length_sum,
                        total_iteration_time=self.period_duration,
                        accept_length_last_client=task_data.get('accept_length', 0), # accept length of last served client
                        last_iteration_time=iteration_time, # iteration time of last served client
                        inference_time=inference_time, # last client inference time
                        get_task_time=time_to_get_task, # time to get task
                        between_tags=between_tags, # tags between iterations
                        rubbish_time=rubbish_time, # rubbish time (SAC training + logging time)
                        real_iteration_time=real_iteration_time, # real iteration time including rubbish time
                        task_detail=task_detail,
                        base_inference_time=task_data.get('base_inference_time', 0.0), # base model inference time
                        base_loop_time=task_data.get('base_loop_time', 0.0), # base model loop time
                        kv_cache_size=kv_cache_size, # sum of current kv cache size of all clients
                        draft_inference_time=draft_inference_time, # draft model inference time (for loop in topk generate)
                        draft_prepare_time=draft_prepare_time,
                        draft_construction_time=draft_construction_time,
                        draft_inference_time_per_for_loop=inference_time_per_for_loop, # average time per for loop in draft inference
                        iteration_num_of_prompt=task_data.get('iteration', 0),
                        client_duration=client_duration,
                        avg_task_waiting_time=avg_task_waiting_time,
                        avg_speed_per_prompt=self.avg_speed_per_prompt,
                        idle_ratio=idle_ratio,
                        real_depth = self.generate_iteration_count, # real depth after using entropy threshold
                        accept_ratio = task_data.get('accept_length', 0) / client_rl_state['last_depth'],
                        accept_ratio_fix_depth = self.environment_tracker.get_or_create_client_data(client_tag)['accept_ratio_history'][-1] if self.generate_iteration_count > 0 else 0.0,
                        threshold = threshold,
                        gpu_freq_mhz=gpu_metrics.get('gpu_freq_mhz', 0.0),
                        gpu_power_w=gpu_metrics.get('gpu_power_w', 0.0),
                        gpu_util_percent=gpu_metrics.get('gpu_util_percent', 0.0),
                        gpu_temp_c=gpu_metrics.get('gpu_temp_c', 0.0)
                    )
                
                except Exception as log_e:
                    print(f"⚠️ Failed to submit system log: {log_e}")
                log_time = time.time() - start_log_time

                rubbish_time = log_time

                self.avg_speed_per_prompt = 0.0 # reset avg speed per prompt

                self.step_count += 1

                inference_start_time = time.time()
                result, task_detail = self._process_single_task(client_tag, task_data, dynamic_depth=dynamic_depth, threshold=threshold)
                total_task_time = time.time() - inference_start_time
                
                # 🔥 Get GPU metrics and inference time
                gpu_metrics = result.get('gpu_metrics', {}) if isinstance(result, dict) else {}
                inference_time = result.get('inference_time', 0.0) if task_detail == 1 else total_task_time  # update用内部时间，其他用总时间
                
                # 🕐 Accumulate draft model inference time
                if task_detail in [1]:  # initialize or update task
                    self.total_draft_inference_time += inference_time
                    self.iteration_count += 1
                    # Accumulate large model inference time
                    if 'base_loop_time' in task_data:
                        self.total_base_inference_time += task_data['base_loop_time']

                # get depth after draft model finish generation
                if task_detail == 1:  # update task
                    client_rl_state['last_depth'] = self.generate_iteration_count
                elif task_detail == 0:  # initialize task
                    client_rl_state['last_depth'] = self.generate_iteration_count

                # Record abnormal inference time cases for debug
                if inference_time > 0.05:
                    abnormal_inference_time.append((inference_time, task_detail))
                if len(abnormal_inference_time) > 20:
                    abnormal_inference_time.pop(0)

                # expected iteration time
                iteration_time = time_to_get_task + inference_time
                # kv cache size measurement
                kv_cache_size = self.draft_model.ea_layer.get_client_states_memory_usage()
                # get detail time from draft model inference program
                draft_prepare_time, draft_inference_time, draft_construction_time, inference_time_per_for_loop = self.draft_model.ea_layer.get_detail_time()

                # Put the result into the corresponding client's result queue for sending
                if result is not None:
                    global_data_manager.put_inference_result(client_tag, result)
                    task_data.clear()
                    
                    # 🕐 Calculate full loop time (from time_to_start_work to put_inference_result)
                    loop_time = time.time() - time_to_start_work
                    
                    # 🕐 Accumulate loop time and inference time statistics
                    if task_detail in [1]:  # initialize or update task
                        self.total_loop_time += loop_time
                        self.total_inference_time += inference_time
                else:
                    print(f"⚠️ No result to send for {client_tag}")
                
                # 🕐 Record work time (from time_to_start_work to now)
                work_time = time.time() - time_to_start_work
                self.work_time_recording.append((time.time(), work_time))
                
            except Exception as e:
                print(f"❌ Error in inference loop: {e}")
                import traceback
                traceback.print_exc()
                if 'client_tag' in locals() and client_tag is not None:
                    error_result = {
                        'type': 'error',
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    }
                    global_data_manager.put_inference_result(client_tag, error_result)
                    print(f"📤 Sent error result to scheduler for {client_tag}")
                time.sleep(0.1)

            real_iteration_time = time.time() - iteration_start_time # with rubbish time
            iteration_start_time = time.time()
    
    def _process_single_task(self, client_tag: str, task_data: dict, dynamic_depth: int = None, threshold: float = 0.77):
        """Handle a single reasoning task"""
        try:
            task_type = task_data.get('type')
            task = -1  # Unknown task
            if dynamic_depth is not None:
                original_depth = self.draft_model.depth
                self.draft_model.depth = dynamic_depth
            
            if task_type == 'initialize':
                task = 0
                result = self._handle_initialize_task(client_tag, task_data)
            elif task_type == 'update':
                task = 1
                result = self._handle_update_task(client_tag, task_data, threshold)
                
            elif task_type == 'end':
                task = 2
                result = self._handle_end_task(client_tag, task_data)
            else:
                task = -1  # Unknown task
                print(f"⚠️ Unknown task type: {task_type}")
                result = {'error': f'Unknown task type: {task_type}'}
            
            # restore original depth
            if dynamic_depth is not None:
                self.draft_model.depth = original_depth

            return result, task
                
        except Exception as e:
            print(f"❌ Error processing task for {client_tag}: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}, -1  # Return tuple (result, task) for consistent unpacking
    
    def _handle_initialize_task(self, client_tag: str, task_data: dict):
        """Handle initialization task"""
        try:
            # 🕐 Start timing initialization
            init_start_time = time.time()
            
            print(f"🔧 Initializing inference for {client_tag}")

            # Check if KV cache already exists
            current_kv = self.draft_model.ea_layer.get_kv_cache(client_tag)
            
            if current_kv is not None:
                print(f"⚠️ KV cache already exists for {client_tag}, reusing existing cache")
                return {
                    'type': 'cache_exists',
                    'message': f'KV cache already exists for {client_tag}'
                }
            
            print(f"📝 No existing KV cache for {client_tag}, initializing...")
            
            # Get necessary data for initialization
            hidden_state = task_data['hidden_state']
            InputIds1 = task_data['input_ids']

            # Ensure tensors are on the correct device
            hidden_state = hidden_state.to(config.device_draft)
            InputIds1 = InputIds1.to(config.device_draft)
            torch.cuda.synchronize()  # Fix: Ensure .to() operation is complete

            # Initialize tree
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids = initialize_tree_draft(
                hidden_state,
                InputIds1,
                self.draft_model,
                None,  # logits_processor
                False,  # backward
                self.draft_model.top_k,
                self.draft_model.depth,
                client_tag=client_tag
            )
            
            # 🕐 Record initialization time
            init_time = time.time() - init_start_time
            self.total_initialize_time += init_time
            self.initialize_count += 1
            
            return {
                'type': 'initialize_result',
                'draft_tokens': draft_tokens,
                'retrieve_indices': retrieve_indices,
                'tree_mask': tree_mask,
                'tree_position_ids': tree_position_ids,
                'total_tokens': self.draft_model.ea_layer.total_tokens
            }
                
        except Exception as e:
            print(f"❌ Error in initialize task for {client_tag}: {e}")
            import traceback
            traceback.print_exc()
            return {'type': 'error', 'error': str(e)}
    
    def _handle_update_task(self, client_tag: str, task_data: dict, threshold: float = 0.77):
        """Process update inference task"""
        try:

            accpt_hidden_state_new, InputIds, accept_length, new_token = (
                task_data['accpt_hidden_state_new'],
                task_data['input_ids'],
                task_data['accept_length'],
                task_data['new_token']
            )

            to_start = time.time()
            accpt_hidden_state_new = accpt_hidden_state_new.to(config.device_draft)
            InputIds = InputIds.to(config.device_draft)
            torch.cuda.synchronize() 
            to_time = time.time() - to_start

            # 🔥 Get GPU metrics before small model inference

            # update_inference_inputs_draft_wramup(
            #     model=self.draft_model,
            #     accept_hidden_state_new=accpt_hidden_state_new,
            #     InputIds=InputIds,
            #     logits_processor=None,
            #     accept_length=accept_length,
            #     backward=False,
            #     depth=7,
            #     top_k=self.draft_model.top_k,
            #     threshold=threshold,
            #     client_tag=client_tag
            # )

            # Update inference inputs
            gpu_metrics = get_gpu_metrics()
            start_inference_time = time.time()
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token_updated, new_token_generate, self.generate_iteration_count = update_inference_inputs_draft(
                model=self.draft_model,
                accept_hidden_state_new=accpt_hidden_state_new,
                InputIds=InputIds,
                logits_processor=None,
                accept_length=accept_length,
                new_token=new_token,
                backward=False,
                depth=self.draft_model.depth,
                top_k=self.draft_model.top_k,
                client_tag=client_tag,
                threshold=threshold
            )
            inference_time = time.time() - start_inference_time
            
            if self.step_count % 50 == 0:
                print(f"🔍 .to() time: {to_time*1000:.3f}ms, inference time: {inference_time*1000:.3f}ms")


            return {
                'type': 'update_result',
                'draft_tokens': draft_tokens,
                'retrieve_indices': retrieve_indices,
                'tree_mask': tree_mask,
                'tree_position_ids': tree_position_ids,
                'new_token': new_token_updated,
                'new_token_generate': new_token_generate,
                'inference_time': inference_time,
                'gpu_metrics': gpu_metrics
            }
            
        except Exception as e:
            print(f"❌ Error in update task for {client_tag}: {e}")
            import traceback
            traceback.print_exc()
            return {'type': 'error', 'error': str(e)}
    
    def _handle_end_task(self, client_tag: str, task_data: dict):
        """Process end task - Clear KV cache and submit terminal transition"""
        try:
            # 🕐 Output runtime statistics
            if self.iteration_count > 0:
                avg_draft_time = self.total_draft_inference_time / self.iteration_count
                avg_base_time = self.total_base_inference_time / self.iteration_count
                avg_loop_time = self.total_loop_time / self.iteration_count
                avg_inference_time = self.total_inference_time / self.iteration_count
                inference_ratio = (self.total_inference_time / max(self.total_loop_time, 1e-6)) * 100
                
                print(f"\n{'='*60}")
                print(f"📊 Runtime Statistics - Client: {client_tag}")
                print(f"{'='*60}")
                print(f"🔵 Total Draft Model Inference Time: {self.total_draft_inference_time*1000:.2f}ms")
                print(f"🔴 Total Base Model Inference Time: {self.total_base_inference_time*1000:.2f}ms")
                print(f"📈 Total Iterations: {self.iteration_count}")
                print(f"🔵 Average Draft Model Inference Time: {avg_draft_time*1000:.2f}ms/iter")
                print(f"🔴 Average Base Model Inference Time: {avg_base_time*1000:.2f}ms/iter")
                print(f"⚖️  Inference Time Ratio (Draft/Base): {self.total_draft_inference_time/max(self.total_base_inference_time, 1e-6):.2f}")
                print(f"")
                print(f"🔄 Total Loop Time: {self.total_loop_time*1000:.2f}ms")
                print(f"🔄 Average Loop Time: {avg_loop_time*1000:.2f}ms/iter")
                print(f"⚡ Total Inference Time: {self.total_inference_time*1000:.2f}ms")
                print(f"⚡ Average Inference Time: {avg_inference_time*1000:.2f}ms/iter")
                print(f"📊 Inference Time Ratio: {inference_ratio:.2f}%")
                
                # 🔧 Initialization Time Statistics
                if self.initialize_count > 0:
                    avg_init_time = self.total_initialize_time / self.initialize_count
                    print(f"")
                    print(f"🔧 Total Initialization Time: {self.total_initialize_time*1000:.2f}ms")
                    print(f"🔧 Average Initialization Time: {avg_init_time*1000:.2f}ms")
                    print(f"🔧 Initialization Count: {self.initialize_count}")
                
                print(f"{'='*60}\n")
                
            print(f"🔚 Handling end signal for {client_tag}")

            # Check if KV cache exists
            current_kv = self.draft_model.ea_layer.get_kv_cache(client_tag)
            
            if current_kv is not None:
                print(f"🧹 Clearing KV cache for {client_tag} after receiving end signal")
                self.draft_model.ea_layer.reset_kv(client_tag)
                print(f"✅ KV cache cleared for {client_tag}")
            else:
                print(f"⚠️ No KV cache found for {client_tag}")
            
            # Reset client session data
            try:
                self.environment_tracker.reset_client_session(client_tag)
                if client_tag in self.rl_client_states:
                    del self.rl_client_states[client_tag]
                print(f"🧠 Session data reset for {client_tag}")
            except Exception as e:
                print(f"⚠️ Session reset error: {e}")

            self.service_history = [tag for tag in self.service_history if tag != client_tag]

            self.avg_speed_per_prompt = task_data['avg_speed_per_prompt'] if 'avg_speed_per_prompt' in task_data else self.avg_speed_per_prompt
            
            return {
                'type': 'end_result',
                'message': f'Session ended and KV cache cleared for {client_tag}'
            }
                
        except Exception as e:
            print(f"❌ Error in end task for {client_tag}: {e}")
            import traceback
            traceback.print_exc()
            return {'type': 'error', 'error': str(e)}


# Global inference engine instance (will be initialized when needed)
single_inference_engine: Optional[SingleInferenceEngine] = None


def get_inference_engine() -> Optional[SingleInferenceEngine]:
    """Get global inference engine instance"""
    return single_inference_engine


def initialize_inference_engine(draft_model: EaModel):
    """Initialize global inference engine"""
    global single_inference_engine
    if single_inference_engine is None:
        single_inference_engine = SingleInferenceEngine(draft_model)
        single_inference_engine.start()
        print("✅ Global inference engine initialized")
    return single_inference_engine


def compute_prompt_difficulty(depths, accept_lengths):
    """
    Calculate prompt difficulty based on accept ratios.
    Lower accept ratio = harder prompt.
    
    Args:
        depths: List of depths used
        accept_lengths: List of accepted lengths
    
    Returns:
        float: Difficulty score in [0, 1], higher = harder
    """
    if not depths or not accept_lengths:
        return 1.0
    
    # Calculate accept ratios
    accept_ratios = [al / max(d, 1) for al, d in zip(accept_lengths, depths)]
    avg_accept_ratio = sum(accept_ratios) / len(accept_ratios)
    
    # Invert: lower accept ratio = higher difficulty
    difficulty = 1.0 - min(avg_accept_ratio, 1.0)
    return difficulty


def get_between_tags(tag_list, new_tag):
    """
    Return elements from after the last occurrence of new_tag to the end,
    then append the last occurrence of new_tag (as a tuple).
    
    tag_list: List of tuples like [(tag, timestamp), ...]
    new_tag: The tag to search for (compared against tuple[0])
    Returns: List of tuples, or None if new_tag not found.
    """
    # Find indices where tuple[0] == new_tag
    matching_indices = [i for i, (tag, _) in enumerate(tag_list) if tag == new_tag]
    
    if not matching_indices:
        return None
    
    last_index = matching_indices[-1]
    
    if last_index == len(tag_list) - 1:
        return [tag_list[last_index]]
    
    return tag_list[last_index + 1:] + [tag_list[last_index]]


def get_recent_unique_tags(service_history, duration):
    """Return unique tags served in the last x seconds"""
    current_time = time.time()
    seen = set()
    recent = []
    for tag, ts in reversed(service_history):
        if current_time - ts > duration:
            break
        if tag not in seen:
            recent.append(tag)
            seen.add(tag)
    recent.reverse()
    return recent


def get_accept_ratio_from_tags(tag_list, client_data):
    """
    Given a list of client tags and a client_data dictionary,
    compute two averages based on 'accept_ratio_history' for each tag.
    
    Returns:
        Tuple[float, float]: 
            - First element: average of last 1 value per tag.
            - Second element: average of last up-to-5 values' average per tag.
    """
    last_1_values = []
    last_5_averages = []

    for tag in tag_list:
        history = client_data.get(tag, {}).get('accept_ratio_history', [])
        if not history:
            continue

        last_1_values.append(history[-1])
        last_5 = history[-5:]
        avg_5 = sum(last_5) / len(last_5)
        last_5_averages.append(avg_5)

    if not last_1_values:
        return 0.0, 0.0

    avg_last_1 = sum(last_1_values) / len(last_1_values)
    avg_last_5 = sum(last_5_averages) / len(last_5_averages)

    return avg_last_1, avg_last_5


def cleanup_timestamped_list(timestamped_list: list, duration: float) -> None:
    """
    Clean up old records from a timestamped list based on time window.
    
    Removes entries older than 'duration' seconds from the current time.
    The list should contain tuples of (timestamp, data).
    """
    current_time = time.time()
    cutoff_time = current_time - duration
    
    while timestamped_list and timestamped_list[0][0] < cutoff_time:
        timestamped_list.pop(0)