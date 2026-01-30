#!/usr/bin/env python3
"""
System Throughput Visualization
Plot the trend of 'avg_system_throughput' from system_information.json
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

def load_system_data(file_path):
    """Load system information data; supports fixing common JSON format issues"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON format error: {e}")
        print("🔧 Attempting to fix JSON file...")
        
        # Read file content
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Try to fix common JSON issues
        try:
            # Remove possible trailing comma
            content = content.rstrip()
            if content.endswith(','):
                content = content[:-1]
            
            # Ensure it ends with a closing bracket
            if not content.endswith(']'):
                content += ']'
            
            # Try parsing the fixed JSON
            data = json.loads(content)
            print("✅ JSON file fixed successfully!")
            return data
            
        except json.JSONDecodeError as e2:
            print(f"❌ JSON fix failed: {e2}")
            print("🔄 Attempting to read line by line...")
            
            return load_system_data_line_by_line(file_path)


def load_system_data_line_by_line(file_path):
    """Read JSON data line by line, skipping corrupted lines"""
    data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Strip the first and last characters (usually [ and ])
    content = ''.join(lines).strip()
    if content.startswith('['):
        content = content[1:]
    if content.endswith(']'):
        content = content[:-1]
    
        # Split into individual JSON objects
    json_objects = content.split('},')
    
    for i, obj_str in enumerate(json_objects):
        obj_str = obj_str.strip()
        if not obj_str:
            continue
            
        if i < len(json_objects) - 1:
            obj_str += '}'
        
        try:
            obj = json.loads(obj_str)
            data.append(obj)
        except json.JSONDecodeError as e:
            print(f"⚠️ Skipping corrupted JSON object (item {i+1}): {str(e)[:100]}")
            continue
    
    print(f"✅ Successfully read {len(data)} records (skipped {len(json_objects) - len(data)} corrupted records)")
    return data

def plot_system_throughput(data_file, save_dir, filter_active_clients=None):
    """Plot average system throughput trend

    Args:
        data_file: path to the data file
        save_dir: directory to save plots
        filter_active_clients: filter condition; only records with active_clients equal to
            this value will be used. If None, no filtering is applied.
    """
    # Load data
    system_data = load_system_data(data_file)
    print(f"Loaded {len(system_data)} system information records")
    
    if filter_active_clients is not None:
        print(f"🔍 Filter enabled: only using records with active_clients = {filter_active_clients}")

    # 🔥 Skip the first and last 300 unstable records (warmup/cooldown)
    original_count = len(system_data)
    if len(system_data) > 600:
        system_data = system_data[300:-300]
        print(f"⏭️  Skipped first 300 records (warmup period)")
        print(f"⏭️  Skipped last 300 records (cooldown period)")
        print(f"📊 Using {len(system_data)} / {original_count} records for analysis (middle {len(system_data)/original_count*100:.1f}%)")
    else:
        print(f"⚠️  Not enough data to skip 600 records, using all {len(system_data)} records")
    
    # Extract data
    times = []
    avg_throughputs = []
    throughputs = []
    active_clients = []
    last_iteration_times = []
    get_task_times = []
    inference_times = []
    dynamic_depths = []
    real_iteration_times = []
    task_details = []
    base_processing_times = []
    abnormal_cases = []  # Store latest abnormal cases list
    kv_cache_sizes = []  # Store KV cache sizes
    draft_inference_times = []  # Store draft model inference time
    draft_prepare_times = []  # Store draft prepare time
    draft_construction_times = []  # Store draft construction time
    draft_inference_depth_pairs = []  # Store (inference_time, depth) paired data
    draft_high_inference_data = []  # Store (step, time, depth) where draft_inference_time > 0.1
    high_inference_loop_times = []  # Store loop times for high inference cases: (step, total_time, loop_times_list)
    normal_inference_loop_times = []  # Store loop times for normal inference cases
    high_draft_inference_prompts = []  # Store (step, time, iteration_num_of_prompt) when draft_inference_time > 0.05
    draft_inference_per_depth_data = []  # 🔥 Store all draft_inference_time_per_for_loop data
    avg_task_waiting_times = []  # Store average task waiting time
    avg_speed_per_prompt_data = []  # Store (step, speed) when speed > 0
    idle_ratios = []  # Store idle ratios
    real_depths = []  # Store real_depth values
    accept_lengths = []  # Store accept_length
    accept_ratios = []  # Store accept_ratio
    accept_ratios_fix_depth = []  # Store accept_ratio_fix_depth
    thresholds = []  # Store threshold values
    iteration_speeds = []  # Store iteration_speed
    avg_iteration_speeds = []  # Store avg_iteration_speed
    gpu_freq_mhz = []  # Store GPU frequency (MHz)
    gpu_power_w = []  # Store GPU power (W)
    gpu_util_percent = []  # Store GPU utilization (%)
    gpu_temp_c = []  # Store GPU temperature (°C)
    
    # Handle timestamps
    base_time = None
    
    skipped_by_filter = 0
    for record in system_data:
        if 'avg_system_throughput' in record and record['avg_system_throughput'] is not None:
            # 🔍 Filter by active_clients
            record_active_clients = record.get('active_clients', 0)
            if filter_active_clients is not None:
                if record_active_clients != filter_active_clients:
                    skipped_by_filter += 1
                    continue
            
            # Use step index as x coordinate
            step_val = len(times)
            
            times.append(step_val)
            avg_throughputs.append(record['avg_system_throughput'])
            throughputs.append(record.get('throughput', 0))
            active_clients.append(record.get('active_clients', 0))
            last_iteration_times.append(record.get('last_iteration_time', 0))
            get_task_times.append(record.get('get_task_time', 0))
            inference_times.append(record.get('inference_time', 0))
            dynamic_depths.append(record.get('dynamic_depth', 1))
            real_iteration_times.append(record.get('real_iteration_time', 0))
            task_details.append(record.get('task_detail', 0))
            base_processing_times.append(record.get('base_processing_time', 0.0))
            kv_cache_sizes.append(record.get('kv_cache_size', 0))
            
            # Handle new draft_inference_time structure (inference_time, depth)
            dit = record.get('draft_inference_time', 0.0)
            iteration_num = record.get('iteration_num_of_prompt', 0)
            
            if isinstance(dit, (list, tuple)) and len(dit) == 2:
                draft_inference_times.append(dit[0])  # only take inference_time
                draft_inference_depth_pairs.append((dit[0], dit[1]))  # store full pair
                
                    # record iteration_num_of_prompt when draft_inference_time > 0.05
                if dit[0] > 0.05:
                    high_draft_inference_prompts.append((step_val, dit[0], iteration_num))
                
                    # record cases where draft_inference_time > 0.1
                if dit[0] > 0.1:
                    draft_high_inference_data.append((step_val, dit[0], dit[1]))
                    # extract inference_time_per_for_loop data
                    loop_times = record.get('inference_time_per_for_loop', [])
                    if loop_times and isinstance(loop_times, list):
                        high_inference_loop_times.append((step_val, dit[0], loop_times))
                else:
                    # record normal-case loop times
                    loop_times = record.get('inference_time_per_for_loop', [])
                    if loop_times and isinstance(loop_times, list):
                        normal_inference_loop_times.append((step_val, dit[0], loop_times))
            else:
                dit_val = dit if isinstance(dit, (int, float)) else 0.0
                draft_inference_times.append(dit_val)
                draft_inference_depth_pairs.append((dit_val, None))
                
                # record iteration_num_of_prompt when draft_inference_time > 0.05
                if dit_val > 0.05:
                    high_draft_inference_prompts.append((step_val, dit_val, iteration_num))
                
                if dit_val > 0.1:
                    draft_high_inference_data.append((step_val, dit_val, None))
                    loop_times = record.get('inference_time_per_for_loop', [])
                    if loop_times and isinstance(loop_times, list):
                        high_inference_loop_times.append((step_val, dit_val, loop_times))
                else:
                    loop_times = record.get('inference_time_per_for_loop', [])
                    if loop_times and isinstance(loop_times, list):
                        normal_inference_loop_times.append((step_val, dit_val, loop_times))
            
            # Handle draft_prepare_time and draft_construction_time (may be tuples)
            dpt = record.get('draft_prepare_time', 0.0)
            draft_prepare_times.append(dpt[0] if isinstance(dpt, (list, tuple)) else dpt)
            
            dct = record.get('draft_construction_time', 0.0)
            draft_construction_times.append(dct[0] if isinstance(dct, (list, tuple)) else dct)
            
            # Extract avg_task_waiting_time
            avg_task_waiting_times.append(record.get('avg_task_waiting_time', 0.0))
            
            # Extract avg_speed_per_prompt (record only non-zero values)
            speed = record.get('avg_speed_per_prompt', 0.0)
            if speed > 0:
                avg_speed_per_prompt_data.append((step_val, speed))
            
            # Extract idle_ratio
            idle_ratios.append(record.get('idle_ratio', 0.0))
            
            # Extract real_depth and accept_length
            real_depths.append(record.get('real_depth', 0))
            accept_lengths.append(record.get('accept_length', 1))  # default to 1 to avoid division by zero
            
            # Extract accept_ratio
            accept_ratios.append(record.get('accept_ratio', 0.0))
            
            # Extract accept_ratio_fix_depth
            accept_ratios_fix_depth.append(record.get('accept_ratio_fix_depth', 0.0))
            
            # Extract threshold
            thresholds.append(record.get('threshold', 1.2))  # default value 1.2
            
            # Extract iteration_speed and avg_iteration_speed
            iteration_speeds.append(record.get('iteration_speed', 0.0))
            avg_iteration_speeds.append(record.get('avg_iteration_speed', 0.0))
            
            # 🔥 Extract draft_inference_time_per_for_loop
            per_depth_times = record.get('draft_inference_time_per_for_loop', [])
            if per_depth_times and isinstance(per_depth_times, list):
                draft_inference_per_depth_data.append(per_depth_times)
            
            # Extract GPU metrics
            gpu_freq_mhz.append(record.get('gpu_freq_mhz', 0.0))
            gpu_power_w.append(record.get('gpu_power_w', 0.0))
            gpu_util_percent.append(record.get('gpu_util_percent', 0.0))
            gpu_temp_c.append(record.get('gpu_temp_c', 0.0))
            
            # Store latest abnormal cases (keep only the last non-empty list)
            if 'abnormal_inference_case' in record and record['abnormal_inference_case']:
                abnormal_cases = record['abnormal_inference_case']
    
    if not times:
        print("❌ No valid throughput data found!")
        return
    
    # Convert to numpy arrays
    times = np.array(times)
    avg_throughputs = np.array(avg_throughputs)
    throughputs = np.array(throughputs)
    active_clients = np.array(active_clients)
    last_iteration_times = np.array(last_iteration_times)
    get_task_times = np.array(get_task_times)
    inference_times = np.array(inference_times)
    dynamic_depths = np.array(dynamic_depths)
    real_iteration_times = np.array(real_iteration_times)
    task_details = np.array(task_details)
    base_processing_times = np.array(base_processing_times)
    kv_cache_sizes = np.array(kv_cache_sizes)
    draft_inference_times = np.array(draft_inference_times)
    draft_prepare_times = np.array(draft_prepare_times)
    draft_construction_times = np.array(draft_construction_times)
    avg_task_waiting_times = np.array(avg_task_waiting_times)
    idle_ratios = np.array(idle_ratios)
    real_depths = np.array(real_depths)
    accept_lengths = np.array(accept_lengths)
    accept_ratios = np.array(accept_ratios)
    accept_ratios_fix_depth = np.array(accept_ratios_fix_depth)
    thresholds = np.array(thresholds)
    iteration_speeds = np.array(iteration_speeds)
    avg_iteration_speeds = np.array(avg_iteration_speeds)
    gpu_freq_mhz = np.array(gpu_freq_mhz)
    gpu_power_w = np.array(gpu_power_w)
    gpu_util_percent = np.array(gpu_util_percent)
    gpu_temp_c = np.array(gpu_temp_c)
    
    # 🔥 Compute average inference time for each depth position
    depth_position_avg_times = {}  # {depth_index: [time1, time2, ...]}
    max_depth_length = 0
    
    for per_depth_list in draft_inference_per_depth_data:
        if per_depth_list:
            max_depth_length = max(max_depth_length, len(per_depth_list))
            for depth_idx, time_val in enumerate(per_depth_list):
                if depth_idx not in depth_position_avg_times:
                    depth_position_avg_times[depth_idx] = []
                depth_position_avg_times[depth_idx].append(time_val)
    
    # Print filter statistics
    if filter_active_clients is not None:
        print(f"📊 Filtered by active_clients = {filter_active_clients}: used {len(times)} records, skipped {skipped_by_filter} records")
    
    # Compute mean, std, min, max for each depth position
    depth_indices = []
    depth_avg_times = []
    depth_std_times = []
    depth_min_times = []
    depth_max_times = []
    
    for depth_idx in sorted(depth_position_avg_times.keys()):
        times_at_depth = depth_position_avg_times[depth_idx]
        depth_indices.append(depth_idx)
        depth_avg_times.append(np.mean(times_at_depth))
        depth_std_times.append(np.std(times_at_depth))
        depth_min_times.append(np.min(times_at_depth))
        depth_max_times.append(np.max(times_at_depth))
    
    # Create figure - 11x3 layout (add one row for iteration_speed metrics)
    fig, axes = plt.subplots(11, 3, figsize=(18, 55))
    
    # 1. Average system throughput trend (main chart)
    ax1 = axes[0, 0]
    # Raw data
    ax1.plot(times, avg_throughputs, color='lightblue', alpha=0.3, linewidth=1, label='Raw Data')
    # Smoothed data - increased smoothing
    smooth_avg_throughput = gaussian_filter1d(avg_throughputs, sigma=300.0)
    ax1.plot(times, smooth_avg_throughput, color='darkblue', linewidth=3, label='Smoothed (σ=300.0)')

    ax1.set_title('Average System Throughput', fontsize=14)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Average Throughput (tokens/sec)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add statistics box
    ax1.text(0.02, 0.95, f'Max: {np.max(avg_throughputs):.1f}\nMin: {np.min(avg_throughputs):.1f}\nMean: {np.mean(avg_throughputs):.1f}\nStd: {np.std(avg_throughputs):.1f}', 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. Instant vs Average Throughput comparison
    ax2 = axes[0, 1]
    ax2.plot(times, throughputs, color='orange', alpha=0.6, linewidth=1, label='Instant Throughput')
    ax2.plot(times, smooth_avg_throughput, color='darkblue', linewidth=2, label='Avg Throughput (Smoothed)')
    
    ax2.set_title('Instant vs Average Throughput', fontsize=14)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Throughput (tokens/sec)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Last Iteration Time
    ax3 = axes[1, 0]
    ax3.plot(times, last_iteration_times, color='lightcoral', alpha=0.3, linewidth=1, label='Raw')
    smooth_last_iter = gaussian_filter1d(last_iteration_times, sigma=30.0)
    ax3.plot(times, smooth_last_iter, color='darkred', linewidth=2, label='Smoothed')
    ax3.set_title('Last Iteration Time', fontsize=14)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Last Iteration Time (sec)')
    ax3.set_ylim(0, 0.2)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add statistics box
    ax3.text(0.02, 0.95, f'Mean: {np.mean(last_iteration_times):.4f}s\nStd: {np.std(last_iteration_times):.4f}s\nMax: {np.max(last_iteration_times):.4f}s\nMin: {np.min(last_iteration_times):.4f}s', 
             transform=ax3.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 4. Active Clients Count
    ax4 = axes[1, 1]
    ax4.plot(times, active_clients, color='lightgreen', alpha=0.3, linewidth=1, label='Raw')
    smooth_active = gaussian_filter1d(active_clients, sigma=5.0)
    ax4.plot(times, smooth_active, color='darkgreen', linewidth=2, label='Smoothed')
    ax4.set_title('Active Clients Count', fontsize=14)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Number of Active Clients')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Get Task Time
    ax5 = axes[1, 2]
    ax5.plot(times, get_task_times, color='lightsalmon', alpha=0.3, linewidth=1, label='Raw')
    smooth_get_task = gaussian_filter1d(get_task_times, sigma=50.0)
    ax5.plot(times, smooth_get_task, color='darkorange', linewidth=2, label='Smoothed')
    ax5.set_title('Get Task Time', fontsize=14)
    ax5.set_xlabel('Step')
    ax5.set_ylabel('Get Task Time (sec)')
    # ax5.set_ylim(0, 0.03)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Add statistics
    ax5.text(0.02, 0.95, f'Mean: {np.mean(get_task_times):.4f}s\nStd: {np.std(get_task_times):.4f}s\nMax: {np.max(get_task_times):.4f}s\nMin: {np.min(get_task_times):.4f}s', 
             transform=ax5.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 6. Inference Time
    ax6 = axes[2, 0]
    ax6.plot(times, inference_times, color='plum', alpha=0.3, linewidth=1, label='Raw')
    smooth_inference = gaussian_filter1d(inference_times, sigma=10.0)
    ax6.plot(times, smooth_inference, color='purple', linewidth=2, label='Smoothed')
    ax6.set_title('Inference Time', fontsize=14)
    ax6.set_xlabel('Step')
    ax6.set_ylabel('Inference Time (sec)')
    ax6.set_ylim(0, 0.1)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    ax6.text(0.02, 0.95, f'Mean: {np.mean(inference_times):.4f}s\nStd: {np.std(inference_times):.4f}s\nMax: {np.max(inference_times):.4f}s\nMin: {np.min(inference_times):.4f}s', 
             transform=ax6.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 7. Dynamic Depth
    ax7 = axes[2, 1]
    ax7.plot(times, dynamic_depths, color='lightcyan', alpha=0.3, linewidth=1, label='Raw')
    smooth_depth = gaussian_filter1d(dynamic_depths, sigma=30.0)
    ax7.plot(times, smooth_depth, color='darkcyan', linewidth=2, label='Smoothed')
    ax7.set_title('Dynamic Depth', fontsize=14)
    ax7.set_xlabel('Step')
    ax7.set_ylabel('Dynamic Depth')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Real Iteration Time
    ax8 = axes[0, 2]
    ax8.plot(times, real_iteration_times, color='lightpink', alpha=0.3, linewidth=1, label='Raw')
    smooth_real_iter = gaussian_filter1d(real_iteration_times, sigma=30.0)
    ax8.plot(times, smooth_real_iter, color='darkmagenta', linewidth=2, label='Smoothed')
    ax8.set_title('Real Iteration Time', fontsize=14)
    ax8.set_xlabel('Step')
    ax8.set_ylabel('Real Iteration Time (sec)')
    ax8.set_ylim(0, 0.2)
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    ax8.text(0.02, 0.95, f'Mean: {np.mean(real_iteration_times):.4f}s\nStd: {np.std(real_iteration_times):.4f}s\nMax: {np.max(real_iteration_times):.4f}s\nMin: {np.min(real_iteration_times):.4f}s', 
             transform=ax8.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 9. Task Detail
    ax9 = axes[2, 2]
    ax9.plot(times, task_details, color='lightsteelblue', alpha=0.3, linewidth=1, label='Raw')
    smooth_task_detail = gaussian_filter1d(task_details.astype(float), sigma=5.0)
    ax9.plot(times, smooth_task_detail, color='steelblue', linewidth=2, label='Smoothed')
    ax9.set_title('Task Detail', fontsize=14)
    ax9.set_xlabel('Step')
    ax9.set_ylabel('Task Detail')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # 10. Base Processing Time
    ax10 = axes[3, 0]
    ax10.plot(times, base_processing_times, color='orange', alpha=0.4, linewidth=1, label='Raw')
    smooth_base_processing = gaussian_filter1d(base_processing_times, sigma=10.0)
    ax10.plot(times, smooth_base_processing, color='darkorange', linewidth=2, label='Smoothed')
    ax10.set_title('Base Model Processing Time', fontsize=14)
    ax10.set_xlabel('Step')
    ax10.set_ylabel('Base Processing Time (sec)')
    ax10.set_ylim(0, 0.1)
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    
    ax10.text(0.02, 0.95, f'Mean: {np.mean(base_processing_times):.4f}s\nStd: {np.std(base_processing_times):.4f}s\nMax: {np.max(base_processing_times):.4f}s\nMin: {np.min(base_processing_times):.4f}s', 
             transform=ax10.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 11. Abnormal Inference Cases (table)
    ax11 = axes[3, 1]
    ax11.axis('off')  # turn off axis
    
    if abnormal_cases:
        # Prepare table data
        table_data = []
        table_data.append(['#', 'Inference Time (s)', 'Task Detail'])
        
        for i, (inf_time, task_detail) in enumerate(abnormal_cases, 1):
            # Display numeric task_detail directly
            table_data.append([str(i), f'{inf_time:.4f}', str(task_detail)])
        
        # Create table
        table = ax11.table(cellText=table_data, 
                          cellLoc='center',
                          loc='center',
                          colWidths=[0.15, 0.45, 0.40])
        
        # Set table style
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Set header style
        for i in range(3):
            cell = table[(0, i)]
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white')
        
        # Set data row styles (alternating colors)
        for i in range(1, len(table_data)):
            for j in range(3):
                cell = table[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor('#f0f0f0')
                else:
                    cell.set_facecolor('white')
                
                # Highlight very long inference times
                if j == 1:  # Inference Time column
                    try:
                        time_val = float(table_data[i][j])
                        if time_val > 0.5:
                            cell.set_facecolor('#ffcccc')  # red highlight
                        elif time_val > 0.2:
                            cell.set_facecolor('#ffffcc')  # yellow highlight
                    except:
                        pass
        
    else:
        ax11.text(0.5, 0.5, 'No abnormal cases recorded', 
                 ha='center', va='center', fontsize=12, color='gray')
    
    # 12. Abnormal Cases Statistics (chart)
    ax12 = axes[3, 2]
    
    if abnormal_cases:
        # Extract inference times and task types
        abnormal_times = [time for time, _ in abnormal_cases]
        abnormal_tasks = [task for _, task in abnormal_cases]
        
        # Create bar chart
        indices = list(range(1, len(abnormal_times) + 1))
        colors = []
        for time_val in abnormal_times:
            if time_val > 0.5:
                colors.append('#ff4444')  # red: critical
            elif time_val > 0.2:
                colors.append('#ffaa00')  # orange: warning
            else:
                colors.append('#4CAF50')  # green: mild
        
        ax12.bar(indices, abnormal_times, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax12.set_title('Abnormal Inference Time Distribution', fontsize=12)
        ax12.set_xlabel('Case #')
        ax12.set_ylabel('Inference Time (s)')
        ax12.grid(True, alpha=0.3, axis='y')
        
        # Add threshold lines
        ax12.axhline(y=0.05, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Normal (<0.05s)')
        ax12.axhline(y=0.2, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Warning (0.2s)')
        ax12.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Critical (0.5s)')
        ax12.legend(fontsize=8)
        
        # Set x-axis ticks
        ax12.set_xticks(indices)
        ax12.set_xticklabels(indices, fontsize=8)
    else:
        ax12.text(0.5, 0.5, 'No abnormal cases to display', 
                 ha='center', va='center', fontsize=12, color='gray')
        ax12.set_title('Abnormal Inference Time Distribution', fontsize=12)
    
    # 13. KV Cache Size (new)
    ax13 = axes[4, 0]
    ax13.plot(times, kv_cache_sizes, color='lightcoral', alpha=0.4, linewidth=1, label='Raw')
    smooth_kv_cache = gaussian_filter1d(kv_cache_sizes, sigma=10.0)
    ax13.plot(times, smooth_kv_cache, color='darkred', linewidth=2, label='Smoothed')
    ax13.set_title('KV Cache Size (Total Tokens)', fontsize=14)
    ax13.set_xlabel('Step')
    ax13.set_ylabel('Total KV Cache Tokens')
    ax13.legend()
    ax13.grid(True, alpha=0.3)
    
    # 14. Inference Time vs KV Cache Size (dual Y-axis comparison)
    ax14 = axes[4, 1]
    
    # Create first Y-axis (left) - Inference Time
    color1 = 'purple'
    ax14.set_xlabel('Step')
    ax14.set_ylabel('Inference Time (sec)', color=color1)
    line1 = ax14.plot(times, inference_times, color=color1, alpha=0.7, linewidth=1.5, label='Inference Time')
    ax14.tick_params(axis='y', labelcolor=color1)
    ax14.set_ylim(0, max(inference_times) * 1.1)
    
    # Create second Y-axis (right) - KV Cache Size
    ax14_right = ax14.twinx()
    color2 = 'lightcoral'
    ax14_right.set_ylabel('KV Cache Size (tokens)', color=color2)
    line2 = ax14_right.plot(times, kv_cache_sizes, color=color2, alpha=0.7, linewidth=1.5, label='KV Cache Size')
    ax14_right.tick_params(axis='y', labelcolor=color2)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax14.legend(lines, labels, loc='upper left')
    
    ax14.set_title('Inference Time vs KV Cache Size', fontsize=14)
    ax14.grid(True, alpha=0.3)
    
    # 15. Draft Inference Time (new - similar to Base Processing Time)
    ax15 = axes[4, 2]
    ax15.plot(times, draft_inference_times, color='lightskyblue', alpha=0.4, linewidth=1, label='Raw')
    smooth_draft_inference = gaussian_filter1d(draft_inference_times, sigma=10.0)
    ax15.plot(times, smooth_draft_inference, color='dodgerblue', linewidth=2, label='Smoothed')
    ax15.set_title('Draft Model Inference Time', fontsize=14)
    ax15.set_xlabel('Step')
    ax15.set_ylabel('Draft Inference Time (sec)')
    ax15.legend()
    ax15.set_ylim(0, 0.04)
    ax15.grid(True, alpha=0.3)
    
    # Add statistics
    dit_mean = np.mean(draft_inference_times)
    dit_std = np.std(draft_inference_times)
    dit_max = np.max(draft_inference_times)
    dit_min = np.min(draft_inference_times)
    stats_text = f'Mean: {dit_mean:.4f}s\nStd: {dit_std:.4f}s\nMax: {dit_max:.4f}s\nMin: {dit_min:.4f}s'
    ax15.text(0.98, 0.97, stats_text, transform=ax15.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 16. Draft Prepare Time (new)
    ax16 = axes[5, 0]
    ax16.plot(times, draft_prepare_times, color='lightseagreen', alpha=0.4, linewidth=1, label='Raw')
    smooth_draft_prepare = gaussian_filter1d(draft_prepare_times, sigma=10.0)
    ax16.plot(times, smooth_draft_prepare, color='seagreen', linewidth=2, label='Smoothed')
    ax16.set_title('Draft Prepare Time', fontsize=14)
    ax16.set_xlabel('Step')
    ax16.set_ylabel('Draft Prepare Time (sec)')
    ax16.legend()
    ax16.grid(True, alpha=0.3)
    
    # Add statistics
    dpt_mean = np.mean(draft_prepare_times)
    dpt_std = np.std(draft_prepare_times)
    dpt_max = np.max(draft_prepare_times)
    dpt_min = np.min(draft_prepare_times)
    stats_text = f'Mean: {dpt_mean:.4f}s\nStd: {dpt_std:.4f}s\nMax: {dpt_max:.4f}s\nMin: {dpt_min:.4f}s'
    ax16.text(0.98, 0.97, stats_text, transform=ax16.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 17. Draft Construction Time (new)
    ax17 = axes[5, 1]
    ax17.plot(times, draft_construction_times, color='yellow', alpha=0.6, linewidth=1, label='Raw')
    smooth_draft_construction = gaussian_filter1d(draft_construction_times, sigma=10.0)
    ax17.plot(times, smooth_draft_construction, color='goldenrod', linewidth=2, label='Smoothed')
    ax17.set_title('Draft Construction Time', fontsize=14)
    ax17.set_xlabel('Step')
    ax17.set_ylabel('Draft Construction Time (sec)')
    ax17.legend()
    ax17.set_ylim(0, 0.02)
    ax17.grid(True, alpha=0.3)
    
    # Add statistics
    dct_mean = np.mean(draft_construction_times)
    dct_std = np.std(draft_construction_times)
    dct_max = np.max(draft_construction_times)
    dct_min = np.min(draft_construction_times)
    stats_text = f'Mean: {dct_mean:.4f}s\nStd: {dct_std:.4f}s\nMax: {dct_max:.4f}s\nMin: {dct_min:.4f}s'
    ax17.text(0.98, 0.97, stats_text, transform=ax17.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 18. Draft Inference vs Construction Time (comparison - no smoothing)
    ax18 = axes[5, 2]
    ax18.plot(times, draft_inference_times, color='dodgerblue', alpha=0.8, linewidth=1.5, label='Draft Inference Time')
    ax18.plot(times, draft_construction_times, color='orange', alpha=0.8, linewidth=1.5, label='Draft Construction Time')
    ax18.set_title('Draft Inference vs Construction Time', fontsize=14)
    ax18.set_xlabel('Step')
    ax18.set_ylabel('Time (sec)')
    ax18.legend()
    ax18.grid(True, alpha=0.3)
    
    # 19. Draft Inference Time > 0.1s with Depth (scatter)
    ax19 = axes[6, 0]
    if draft_high_inference_data:
        # Filter cases that have depth info
        valid_cases = [(s, t, d) for s, t, d in draft_high_inference_data if d is not None]
        if valid_cases:
            steps_high = [s for s, t, d in valid_cases]
            times_high = [t for s, t, d in valid_cases]
            depths_high = [d for s, t, d in valid_cases]
            
            scatter = ax19.scatter(steps_high, times_high, c=depths_high, cmap='viridis', 
                                  s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
            cbar = plt.colorbar(scatter, ax=ax19)
            cbar.set_label('Depth', rotation=270, labelpad=15)
            
            ax19.set_title(f'Draft Inference Time > 0.1s (Total: {len(valid_cases)} cases)', fontsize=14)
            ax19.set_xlabel('Step')
            ax19.set_ylabel('Draft Inference Time (sec)')
            ax19.grid(True, alpha=0.3)
        else:
            ax19.text(0.5, 0.5, 'No cases with depth info', 
                     ha='center', va='center', fontsize=12, color='gray')
            ax19.set_title('Draft Inference Time > 0.1s', fontsize=14)
    else:
        ax19.text(0.5, 0.5, 'No cases with draft_inference_time > 0.1s', 
                 ha='center', va='center', fontsize=12, color='gray')
        ax19.set_title('Draft Inference Time > 0.1s', fontsize=14)
    
    # 20. Depth Distribution for High Inference Time (bar chart)
    ax20 = axes[6, 1]
    if draft_high_inference_data:
        valid_cases = [(s, t, d) for s, t, d in draft_high_inference_data if d is not None]
        if valid_cases:
            from collections import Counter
            depths_high = [d for s, t, d in valid_cases]
            depth_counter = Counter(depths_high)
            
            depths_sorted = sorted(depth_counter.keys())
            counts = [depth_counter[d] for d in depths_sorted]
            
            ax20.bar(depths_sorted, counts, color='dodgerblue', alpha=0.7, edgecolor='black', linewidth=0.5)
            ax20.set_title('Depth Distribution (Inference Time > 0.1s)', fontsize=14)
            ax20.set_xlabel('Depth')
            ax20.set_ylabel('Count')
            ax20.grid(True, alpha=0.3, axis='y')
            
            # Show counts above bars
            for depth, count in zip(depths_sorted, counts):
                ax20.text(depth, count, str(count), ha='center', va='bottom', fontsize=10)
        else:
            ax20.text(0.5, 0.5, 'No cases with depth info', 
                     ha='center', va='center', fontsize=12, color='gray')
            ax20.set_title('Depth Distribution', fontsize=14)
    else:
        ax20.text(0.5, 0.5, 'No high inference time cases', 
                 ha='center', va='center', fontsize=12, color='gray')
        ax20.set_title('Depth Distribution', fontsize=14)
    
    # Hide remaining empty subplots
    # axes[6, 2].axis('off')
    
    # 21. Per-Loop Inference Time for High Cases (boxplot)
    ax21 = axes[6, 2]
    if high_inference_loop_times:
        # Prepare data: loop times for each case
        all_loop_times = []
        case_labels = []
        
        for step, total_time, loop_times in high_inference_loop_times[:10]:  # show top 10 cases only
            all_loop_times.append([t * 1000 for t in loop_times])  # convert to ms
            case_labels.append(f'S{step}\n{total_time:.3f}s')
        
        if all_loop_times:
            # Create boxplot
            bp = ax21.boxplot(all_loop_times, labels=case_labels, patch_artist=True)
            
            # Set box colors
            for patch in bp['boxes']:
                patch.set_facecolor('lightskyblue')
                patch.set_alpha(0.7)
            
            # Style medians
            for median in bp['medians']:
                median.set_color('red')
                median.set_linewidth(2)
            
            ax21.set_title(f'Per-Loop Times (Inference > 0.1s, Top {len(all_loop_times)} cases)', fontsize=12)
            ax21.set_xlabel('Step (Total Time)')
            ax21.set_ylabel('Loop Time (ms)')
            ax21.grid(True, alpha=0.3, axis='y')
            ax21.tick_params(axis='x', rotation=45, labelsize=8)
            
            # Add average line
            all_values = [t for case in all_loop_times for t in case]
            avg_time = np.mean(all_values)
            ax21.axhline(y=avg_time, color='green', linestyle='--', linewidth=1.5, 
                        alpha=0.7, label=f'Avg: {avg_time:.2f}ms')
            ax21.legend(fontsize=9)
        else:
            ax21.text(0.5, 0.5, 'No loop time data available', 
                     ha='center', va='center', fontsize=12, color='gray')
    else:
        ax21.text(0.5, 0.5, 'No high inference cases with loop times', 
                 ha='center', va='center', fontsize=12, color='gray')
        ax21.set_title('Per-Loop Inference Times', fontsize=12)
    
    # 22. Iteration Num of Prompt for High Draft Inference (new - simple bar chart)
    ax22 = axes[7, 0]
    if high_draft_inference_prompts:
        # Extract data
        steps_prompt = [s for s, t, p in high_draft_inference_prompts]
        times_prompt = [t for s, t, p in high_draft_inference_prompts]
        prompts = [p for s, t, p in high_draft_inference_prompts]
        
        # Create bar chart
        from collections import Counter
        prompt_counter = Counter(prompts)
        prompt_vals = sorted(prompt_counter.keys())
        prompt_counts = [prompt_counter[p] for p in prompt_vals]
        
        ax22.bar(prompt_vals, prompt_counts, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax22.set_title(f'Iteration Num Distribution (Draft Inference > 0.05s, Total: {len(prompts)} cases)', fontsize=12)
        ax22.set_xlabel('Iteration Num of Prompt')
        ax22.set_ylabel('Count')
        ax22.grid(True, alpha=0.3, axis='y')
        
        # Show counts above bars
        for pval, count in zip(prompt_vals, prompt_counts):
            ax22.text(pval, count, str(count), ha='center', va='bottom', fontsize=9)
    else:
        ax22.text(0.5, 0.5, 'No cases with draft_inference_time > 0.05s', 
                 ha='center', va='center', fontsize=12, color='gray')
        ax22.set_title('Iteration Num Distribution', fontsize=12)
    
    # 23. Average Task Waiting Time
    ax23 = axes[7, 1]
    ax23.plot(times, avg_task_waiting_times, color='lightcoral', alpha=0.4, linewidth=1, label='Raw')
    smooth_avg_waiting = gaussian_filter1d(avg_task_waiting_times, sigma=50.0)
    ax23.plot(times, smooth_avg_waiting, color='darkred', linewidth=2, label='Smoothed')
    ax23.set_title('Average Task Waiting Time in Queue', fontsize=14)
    ax23.set_xlabel('Step')
    ax23.set_ylabel('Avg Waiting Time (sec)')
    ax23.legend()
    ax23.set_ylim(0, 0.2)
    ax23.grid(True, alpha=0.3)
    
    awt_mean = np.mean(avg_task_waiting_times)
    awt_std = np.std(avg_task_waiting_times)
    awt_max = np.max(avg_task_waiting_times)
    awt_min = np.min(avg_task_waiting_times)
    stats_text = f'Mean: {awt_mean:.4f}s\nStd: {awt_std:.4f}s\nMax: {awt_max:.4f}s\nMin: {awt_min:.4f}s'
    ax23.text(0.98, 0.97, stats_text, transform=ax23.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 24. Average Speed per Prompt (non-zero values only)
    ax24 = axes[7, 2]
    if avg_speed_per_prompt_data:
        # Extract steps and speeds
        steps_speed = [s for s, _ in avg_speed_per_prompt_data]
        speeds = [speed for _, speed in avg_speed_per_prompt_data]
        
        # Plot scatter
        ax24.scatter(steps_speed, speeds, color='dodgerblue', alpha=0.6, s=20, 
                    edgecolors='black', linewidth=0.3, label='Speed per Prompt')
        
        # Add mean line
        avg_speed = np.mean(speeds)
        ax24.axhline(y=avg_speed, color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {avg_speed:.2f} tok/s', alpha=0.7)
        
        ax24.set_title('Average Speed per Prompt (Non-zero values)', fontsize=14)
        ax24.set_xlabel('Step')
        ax24.set_ylabel('Speed (tokens/sec)')
        ax24.legend()
        ax24.grid(True, alpha=0.3)
        
        # Add statistics
        info_text = (f"Count: {len(speeds)}\n"
                    f"Mean: {avg_speed:.2f}\n"
                    f"Max: {np.max(speeds):.2f}\n"
                    f"Min: {np.min(speeds):.2f}")
        ax24.text(0.02, 0.98, info_text,
                transform=ax24.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        ax24.text(0.5, 0.5, 'No non-zero avg_speed_per_prompt data', 
                ha='center', va='center', fontsize=12, color='gray')
        ax24.set_title('Average Speed per Prompt', fontsize=14)
    
    # 25. Idle Ratio
    ax25 = axes[8, 0]
    ax25.plot(times, idle_ratios, color='lightgreen', alpha=0.4, linewidth=1, label='Raw')
    smooth_idle = gaussian_filter1d(idle_ratios, sigma=50.0)
    ax25.plot(times, smooth_idle, color='darkgreen', linewidth=2, label='Smoothed')
    ax25.set_title('System Idle Ratio', fontsize=14)
    ax25.set_xlabel('Step')
    ax25.set_ylabel('Idle Ratio (0=Busy, 1=Idle)')
    ax25.set_ylim(0, 1.0)
    ax25.legend()
    ax25.grid(True, alpha=0.3)
    
    # Add reference lines
    ax25.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='50% Load')
    ax25.axhline(y=0.2, color='red', linestyle='--', linewidth=1, alpha=0.5, label='80% Load')
    
    # Add statistics
    avg_idle = np.mean(idle_ratios)
    info_text = (f"Mean: {avg_idle:.3f}\n"
                f"Max: {np.max(idle_ratios):.3f}\n"
                f"Min: {np.min(idle_ratios):.3f}\n"
                f"Avg Load: {(1-avg_idle)*100:.1f}%")
    ax25.text(0.02, 0.98, info_text,
            transform=ax25.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 26. Real Depth
    ax26 = axes[8, 1]
    ax26.plot(times, real_depths, color='lightcoral', alpha=0.4, linewidth=1, label='Raw')
    smooth_real_depth = gaussian_filter1d(real_depths, sigma=30.0)
    ax26.plot(times, smooth_real_depth, color='darkred', linewidth=2, label='Smoothed')
    ax26.set_title('Real Depth', fontsize=14)
    ax26.set_xlabel('Step')
    ax26.set_ylabel('Real Depth')
    ax26.legend()
    ax26.grid(True, alpha=0.3)
    
    # Add statistics
    avg_real_depth = np.mean(real_depths)
    info_text = (f"Mean: {avg_real_depth:.2f}\n"
                f"Max: {np.max(real_depths):.0f}\n"
                f"Min: {np.min(real_depths):.0f}\n"
                f"Std: {np.std(real_depths):.2f}")
    ax26.text(0.02, 0.98, info_text,
            transform=ax26.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 27. Accept Ratio
    ax27 = axes[8, 2]
    ax27.plot(times, accept_ratios, color='lightseagreen', alpha=0.4, linewidth=1, label='Raw')
    smooth_accept_ratio = gaussian_filter1d(accept_ratios, sigma=30.0)
    ax27.plot(times, smooth_accept_ratio, color='seagreen', linewidth=2, label='Smoothed')
    ax27.set_title('Accept Ratio', fontsize=14)
    ax27.set_xlabel('Step')
    ax27.set_ylabel('Accept Ratio')
    ax27.set_ylim(0, 1.0)
    ax27.legend()
    ax27.grid(True, alpha=0.3)
    
    # Add statistics
    avg_accept_ratio = np.mean(accept_ratios)
    info_text = (f"Mean: {avg_accept_ratio:.3f}\n"
                f"Max: {np.max(accept_ratios):.3f}\n"
                f"Min: {np.min(accept_ratios):.3f}\n"
                f"Std: {np.std(accept_ratios):.3f}")
    ax27.text(0.02, 0.98, info_text,
            transform=ax27.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 29. Threshold
    ax29 = axes[9, 1]
    ax29.plot(times, thresholds, color='lightskyblue', alpha=0.4, linewidth=1, label='Raw')
    smooth_threshold = gaussian_filter1d(thresholds, sigma=30.0)
    ax29.plot(times, smooth_threshold, color='dodgerblue', linewidth=2, label='Smoothed')
    ax29.set_title('Threshold', fontsize=14)
    ax29.set_xlabel('Step')
    ax29.set_ylabel('Threshold')
    ax29.set_ylim(1.0, 1.5)  # adjust according to your threshold range
    ax29.legend(loc='upper left')
    ax29.grid(True, alpha=0.3)
    
    # Add statistics (top-right)
    mean_threshold = np.mean(thresholds)
    std_threshold = np.std(thresholds)
    info_text = (f"Mean: {mean_threshold:.4f}\n"
                f"Std: {std_threshold:.4f}\n"
                f"Max: {np.max(thresholds):.4f}\n"
                f"Min: {np.min(thresholds):.4f}")
    ax29.text(0.98, 0.98, info_text,
            transform=ax29.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    
    # 30. Average Iteration Speed (similar to Average Throughput)
    ax30 = axes[10, 0]
    # Raw data
    ax30.plot(times, avg_iteration_speeds, color='lightcoral', alpha=0.3, linewidth=1, label='Raw Data')
    # Smoothed data
    smooth_avg_iter_speed = gaussian_filter1d(avg_iteration_speeds, sigma=300.0)
    ax30.plot(times, smooth_avg_iter_speed, color='darkred', linewidth=3, label='Smoothed (σ=300.0)')
    
    ax30.set_title('Average Iteration Speed', fontsize=14)
    ax30.set_xlabel('Step')
    ax30.set_ylabel('Average Iteration Speed (iterations/sec)')
    ax30.legend()
    ax30.grid(True, alpha=0.3)
    
    # Add statistics
    ax30.text(0.02, 0.95, f'Max: {np.max(avg_iteration_speeds):.1f}\nMin: {np.min(avg_iteration_speeds):.1f}\nMean: {np.mean(avg_iteration_speeds):.1f}\nStd: {np.std(avg_iteration_speeds):.1f}', 
             transform=ax30.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 31. Instant vs Average Iteration Speed
    ax31 = axes[10, 1]
    ax31.plot(times, iteration_speeds, color='lightsalmon', alpha=0.6, linewidth=1, label='Instant Iteration Speed')
    ax31.plot(times, smooth_avg_iter_speed, color='darkred', linewidth=2, label='Avg Iteration Speed (Smoothed)')
    
    ax31.set_title('Instant vs Average Iteration Speed', fontsize=14)
    ax31.set_xlabel('Step')
    ax31.set_ylabel('Iteration Speed (iterations/sec)')
    ax31.legend()
    ax31.grid(True, alpha=0.3)
    
    # 32. Throughput vs Iteration Speed (aligned scale comparison)
    ax32 = axes[10, 2]
    
    # Compute ranges for both metrics to align scales
    throughput_min = np.min(smooth_avg_throughput)
    throughput_max = np.max(smooth_avg_throughput)
    throughput_range = throughput_max - throughput_min
    
    iter_speed_min = np.min(smooth_avg_iter_speed)
    iter_speed_max = np.max(smooth_avg_iter_speed)
    iter_speed_range = iter_speed_max - iter_speed_min
    
    # Compute center values
    throughput_center = (throughput_min + throughput_max) / 2
    iter_speed_center = (iter_speed_min + iter_speed_max) / 2
    
    # Use the same relative range (take the larger one)
    max_range = max(throughput_range, iter_speed_range)
    padding = max_range * 0.1  # 10% padding
    
    # Set symmetric Y-axis limits
    throughput_ylim = (throughput_center - max_range/2 - padding, throughput_center + max_range/2 + padding)
    iter_speed_ylim = (iter_speed_center - max_range/2 - padding, iter_speed_center + max_range/2 + padding)
    
    # Create first Y-axis (left) - Throughput
    color1 = 'darkblue'
    ax32.set_xlabel('Step')
    ax32.set_ylabel('Throughput (tokens/sec)', color=color1)
    line1 = ax32.plot(times, smooth_avg_throughput, color=color1, linewidth=2, label='Avg Throughput')
    ax32.tick_params(axis='y', labelcolor=color1)
    ax32.set_ylim(throughput_ylim)
    
    # Create second Y-axis (right) - Iteration Speed
    ax32_right = ax32.twinx()
    color2 = 'darkred'
    ax32_right.set_ylabel('Iteration Speed (iterations/sec)', color=color2)
    line2 = ax32_right.plot(times, smooth_avg_iter_speed, color=color2, linewidth=2, label='Avg Iteration Speed')
    ax32_right.tick_params(axis='y', labelcolor=color2)
    ax32_right.set_ylim(iter_speed_ylim)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax32.legend(lines, labels, loc='upper left')
    
    ax32.set_title('Throughput vs Iteration Speed (Aligned Scale)', fontsize=14)
    ax32.grid(True, alpha=0.3)
    
    # 🔥 33. Draft Inference Time Per Depth Position (table view)
    ax33 = axes[9, 0]
    ax33.axis('off')  # hide axis
    
    if depth_indices:
        # Prepare table data
        table_data = []
        table_data.append(['Depth', 'Avg Time (ms)', 'Std (ms)', 'Min (ms)', 'Max (ms)', 'Samples'])
        
        for idx, avg_time, std_time, min_time, max_time in zip(depth_indices, depth_avg_times, depth_std_times, depth_min_times, depth_max_times):
            table_data.append([
                f'{idx}',
                f'{avg_time*1000:.3f}',
                f'{std_time*1000:.3f}',
                f'{min_time*1000:.3f}',
                f'{max_time*1000:.3f}',
                f'{len(depth_position_avg_times[idx])}'
            ])
        
        # Create table
        table = ax33.table(cellText=table_data, 
                          cellLoc='center',
                          loc='center',
                          colWidths=[0.12, 0.18, 0.15, 0.15, 0.15, 0.15])
        
        # Set table style
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Header row style
        for i in range(6):
            cell = table[(0, i)]
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_data)):
            for j in range(6):
                cell = table[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor('#f0f0f0')
                else:
                    cell.set_facecolor('white')
        
        # Add title and stats
        title_text = f'Draft Inference Time Per Depth Position\n(Total samples: {len(draft_inference_per_depth_data)}, Max depth: {max(depth_indices)+1})'
        ax33.text(0.5, 0.95, title_text, transform=ax33.transAxes, 
                 fontsize=12, weight='bold', ha='center', va='top')
    else:
        ax33.text(0.5, 0.5, 'No draft_inference_time_per_for_loop data available', 
                 ha='center', va='center', transform=ax33.transAxes, fontsize=12)
        ax33.set_title('Draft Inference Time Per Depth Position', fontsize=14)
    
    # 🔥 GPU Frequency - axes[9, 2]
    ax34 = axes[9, 2]
    
    # Check for GPU frequency data
    if np.any(gpu_freq_mhz > 0):
        # Raw data (light color)
        ax34.plot(times, gpu_freq_mhz, color='lightgreen', alpha=0.3, linewidth=1, label='Raw Data')
        
        # Smoothed data
        smooth_gpu_freq = gaussian_filter1d(gpu_freq_mhz, sigma=30.0)
        ax34.plot(times, smooth_gpu_freq, color='darkgreen', linewidth=2, label='Smoothed (σ=30.0)')
        
        ax34.set_title('GPU Frequency', fontsize=14)
        ax34.set_xlabel('Step')
        ax34.set_ylabel('GPU Frequency (MHz)')
        ax34.legend()
        ax34.grid(True, alpha=0.3)
        
        # Add statistics
        ax34.text(0.02, 0.95, f'Max: {np.max(gpu_freq_mhz):.0f} MHz\nMin: {np.min(gpu_freq_mhz[gpu_freq_mhz>0]):.0f} MHz\nMean: {np.mean(gpu_freq_mhz):.0f} MHz\nStd: {np.std(gpu_freq_mhz):.0f} MHz', 
                 transform=ax34.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        ax34.text(0.5, 0.5, 'No GPU frequency data available', 
                 ha='center', va='center', transform=ax34.transAxes, fontsize=12)
        ax34.set_title('GPU Frequency', fontsize=14)
        ax34.axis('off')
    
    plt.tight_layout()
    
    # Save figure
    save_path = Path(save_dir) / 'system_throughput_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 System throughput analysis saved to: {save_path}")
    
    # Print Task Detail statistics
    print(f"\n📊 Task Detail Statistics:")
    from collections import Counter
    task_detail_counts = Counter(task_details)
    for value in sorted(task_detail_counts.keys()):
        count = task_detail_counts[value]
        percentage = (count / len(task_details)) * 100
        task_type_name = {0: 'Initialize', 1: 'Update', 2: 'End'}.get(value, f'Unknown({value})')
        print(f"   Task Type {value} ({task_type_name}): {count} times ({percentage:.1f}%)")
    
    # Group inference times by task_type
    print(f"\n⏱️  Inference Time by Task Type:")
    for value in sorted(task_detail_counts.keys()):
        task_type_name = {0: 'Initialize', 1: 'Update', 2: 'End'}.get(value, f'Unknown({value})')
        # Filter inference_time for the given task_type
        task_mask = task_details == value
        task_inference_times = inference_times[task_mask]
        if len(task_inference_times) > 0:
            avg_time = np.mean(task_inference_times)
            std_time = np.std(task_inference_times)
            min_time = np.min(task_inference_times)
            max_time = np.max(task_inference_times)
            print(f"   Type {value} ({task_type_name}): {avg_time:.4f}±{std_time:.4f}s (min:{min_time:.4f}, max:{max_time:.4f})")
    
    # Print summary statistics
    print(f"\n📈 System Performance Statistics:")
    print(f"   Total Steps: {times[-1] - times[0]:.0f}")
    print(f"   Data Points: {len(times)}")
    print(f"   Average Throughput: {np.mean(avg_throughputs):.2f} ± {np.std(avg_throughputs):.2f} tokens/sec")
    print(f"   Peak Throughput: {np.max(avg_throughputs):.2f} tokens/sec")
    print(f"   Average Iteration Speed: {np.mean(avg_iteration_speeds):.2f} ± {np.std(avg_iteration_speeds):.2f} iterations/sec")
    print(f"   Peak Iteration Speed: {np.max(avg_iteration_speeds):.2f} iterations/sec")
    print(f"   Active Clients Range: {np.min(active_clients)} - {np.max(active_clients)}")
    print(f"   Average Last Iteration Time: {np.mean(last_iteration_times):.4f} ± {np.std(last_iteration_times):.4f} sec")
    print(f"   Average Get Task Time: {np.mean(get_task_times):.4f} ± {np.std(get_task_times):.4f} sec") 
    print(f"   Average Inference Time: {np.mean(inference_times):.4f} ± {np.std(inference_times):.4f} sec")
    print(f"   Average Real Iteration Time: {np.mean(real_iteration_times):.4f} ± {np.std(real_iteration_times):.4f} sec")
    print(f"   Average Base Processing Time: {np.mean(base_processing_times):.4f} ± {np.std(base_processing_times):.4f} sec")
    print(f"   Average Draft Inference Time: {np.mean(draft_inference_times):.4f} ± {np.std(draft_inference_times):.4f} sec")
    print(f"   Average Draft Prepare Time: {np.mean(draft_prepare_times):.4f} ± {np.std(draft_prepare_times):.4f} sec")
    print(f"   Average Draft Construction Time: {np.mean(draft_construction_times):.4f} ± {np.std(draft_construction_times):.4f} sec")
    print(f"   Average Task Waiting Time: {np.mean(avg_task_waiting_times):.4f} ± {np.std(avg_task_waiting_times):.4f} sec")
    
    # print avg_speed_per_prompt statistics (non-zero values only)
    if avg_speed_per_prompt_data:
        speeds = [speed for _, speed in avg_speed_per_prompt_data]
        print(f"\n🚀 Average Speed per Prompt Statistics (Non-zero values only):")
        print(f"   Total Prompts Completed: {len(speeds)}")
        print(f"   Average Speed: {np.mean(speeds):.2f} ± {np.std(speeds):.2f} tok/s")
        print(f"   Max Speed: {np.max(speeds):.2f} tok/s")
        print(f"   Min Speed: {np.min(speeds):.2f} tok/s")
        print(f"   Median Speed: {np.median(speeds):.2f} tok/s")
    
    # print Idle Ratio statistics
    print(f"\n⚡ System Idle Ratio Statistics:")
    print(f"   Average Idle Ratio: {np.mean(idle_ratios):.4f} (Load: {(1-np.mean(idle_ratios))*100:.1f}%)")
    print(f"   Max Idle Ratio: {np.max(idle_ratios):.4f} (Min Load: {(1-np.max(idle_ratios))*100:.1f}%)")
    print(f"   Min Idle Ratio: {np.min(idle_ratios):.4f} (Max Load: {(1-np.min(idle_ratios))*100:.1f}%)")
    print(f"   Median Idle Ratio: {np.median(idle_ratios):.4f}")
    
    # print Real Depth statistics
    print(f"\n📏 Real Depth Statistics:")
    print(f"   Average Real Depth: {np.mean(real_depths):.2f} ± {np.std(real_depths):.2f}")
    print(f"   Max Real Depth: {np.max(real_depths):.0f}")
    print(f"   Min Real Depth: {np.min(real_depths):.0f}")
    print(f"   Median Real Depth: {np.median(real_depths):.2f}")
    
    # print Accept Length statistics
    print(f"\n✅ Accept Length Statistics:")
    print(f"   Average Accept Length: {np.mean(accept_lengths):.2f} ± {np.std(accept_lengths):.2f}")
    print(f"   Max Accept Length: {np.max(accept_lengths):.0f}")
    print(f"   Min Accept Length: {np.min(accept_lengths):.0f}")
    print(f"   Median Accept Length: {np.median(accept_lengths):.2f}")
    
    # print Accept Ratio statistics
    print(f"\n📊 Accept Ratio Statistics:")
    print(f"   Average Accept Ratio: {np.mean(accept_ratios):.3f} ± {np.std(accept_ratios):.3f}")
    print(f"   Max Accept Ratio: {np.max(accept_ratios):.3f}")
    print(f"   Min Accept Ratio: {np.min(accept_ratios):.3f}")
    print(f"   Median Accept Ratio: {np.median(accept_ratios):.3f}")
    
    # print Accept Ratio Fix Depth statistics
    print(f"\n📊 Accept Ratio Fix Depth Statistics:")
    print(f"   Average Accept Ratio Fix Depth: {np.mean(accept_ratios_fix_depth):.3f} ± {np.std(accept_ratios_fix_depth):.3f}")
    print(f"   Max Accept Ratio Fix Depth: {np.max(accept_ratios_fix_depth):.3f}")
    print(f"   Min Accept Ratio Fix Depth: {np.min(accept_ratios_fix_depth):.3f}")
    print(f"   Median Accept Ratio Fix Depth: {np.median(accept_ratios_fix_depth):.3f}")
    
    # print Real Depth / Accept Length statistics
    safe_accept_lengths = np.where(accept_lengths == 0, 1, accept_lengths)
    depth_accept_ratio = real_depths / safe_accept_lengths
    print(f"\n📊 Real Depth / Accept Length Ratio Statistics:")
    print(f"   Average Ratio: {np.mean(depth_accept_ratio):.3f} ± {np.std(depth_accept_ratio):.3f}")
    print(f"   Max Ratio: {np.max(depth_accept_ratio):.3f}")
    print(f"   Min Ratio: {np.min(depth_accept_ratio):.3f}")
    print(f"   Median Ratio: {np.median(depth_accept_ratio):.3f}")
    
    # print draft_inference_time > 0.1s analysis
    print(f"\n🔍 Draft Inference Time > 0.1s Analysis:")
    high_inference_cases = [(t, d) for t, d in draft_inference_depth_pairs if t > 0.1 and d is not None]
    if high_inference_cases:
        print(f"   Total cases with draft_inference_time > 0.1s: {len(high_inference_cases)}")
        
        # Depth distribution statistics
        from collections import Counter
        depth_counter = Counter([d for _, d in high_inference_cases])
        print(f"\n   Depth distribution:")
        for depth, count in sorted(depth_counter.items()):
            percentage = (count / len(high_inference_cases)) * 100
            avg_time = np.mean([t for t, d in high_inference_cases if d == depth])
            print(f"      Depth {depth}: {count} cases ({percentage:.1f}%), Avg Time: {avg_time:.4f}s")
        
        # Display top 10 longest inference time cases
        top_cases = sorted(high_inference_cases, key=lambda x: x[0], reverse=True)[:10]
        print(f"\n   Top 10 longest inference times:")
        for i, (inf_time, depth) in enumerate(top_cases, 1):
            print(f"      {i}. Time: {inf_time:.4f}s, Depth: {depth}")
    else:
        print(f"   ✅ No cases with draft_inference_time > 0.1s")
    
    # print Per-Loop Inference Time Analysis
    if high_inference_loop_times:
        print(f"\n🔍 Per-Loop Inference Time Analysis (for cases > 0.1s):")
        print(f"   Total high inference cases with loop data: {len(high_inference_loop_times)}")
        
        for step, total_time, loop_times in high_inference_loop_times[:5]:  # 只显示前5个
            loop_times_ms = [t * 1000 for t in loop_times]
            avg_loop = np.mean(loop_times_ms)
            max_loop = np.max(loop_times_ms)
            min_loop = np.min(loop_times_ms)
            std_loop = np.std(loop_times_ms)
            
            print(f"\n   Step {step} (Total: {total_time:.4f}s, Loops: {len(loop_times)}):")
            print(f"      Loop times (ms): {[f'{t:.2f}' for t in loop_times_ms]}")
            print(f"      Avg: {avg_loop:.2f}ms, Max: {max_loop:.2f}ms, Min: {min_loop:.2f}ms, Std: {std_loop:.2f}ms")
            
            # 检测异常loop
            if max_loop > avg_loop * 3:
                outlier_idx = loop_times_ms.index(max_loop)
                print(f"      ⚠️  Outlier detected at loop #{outlier_idx}: {max_loop:.2f}ms ({max_loop/avg_loop:.1f}x avg)")
    
    # print Per-Loop Inference Time Analysis for normal cases (as a baseline)
    if normal_inference_loop_times:
        print(f"\n✅ Per-Loop Inference Time Analysis (Normal Cases, inference_time <= 0.1s):")
        print(f"   Total normal cases with loop data: {len(normal_inference_loop_times)}")
        
        # collect all loop times
        all_normal_loop_times = []
        for step, total_time, loop_times in normal_inference_loop_times:
            all_normal_loop_times.extend([t * 1000 for t in loop_times]) 
        
        if all_normal_loop_times:
            avg_normal = np.mean(all_normal_loop_times)
            median_normal = np.median(all_normal_loop_times)
            std_normal = np.std(all_normal_loop_times)
            min_normal = np.min(all_normal_loop_times)
            max_normal = np.max(all_normal_loop_times)
            p95_normal = np.percentile(all_normal_loop_times, 95)
            p99_normal = np.percentile(all_normal_loop_times, 99)
            
            print(f"\n   📊 Overall Statistics (All Normal Loop Times):")
            print(f"      Total loops: {len(all_normal_loop_times)}")
            print(f"      Average: {avg_normal:.3f}ms")
            print(f"      Median:  {median_normal:.3f}ms")
            print(f"      Std Dev: {std_normal:.3f}ms")
            print(f"      Min:     {min_normal:.3f}ms")
            print(f"      Max:     {max_normal:.3f}ms")
            print(f"      P95:     {p95_normal:.3f}ms")
            print(f"      P99:     {p99_normal:.3f}ms")
            
            # Display first 5 normal cases as examples
            print(f"\n   📝 Sample Normal Cases (First 5):")
            for i, (step, total_time, loop_times) in enumerate(normal_inference_loop_times[:5], 1):
                loop_times_ms = [t * 1000 for t in loop_times]
                avg_loop = np.mean(loop_times_ms)
                print(f"      {i}. Step {step} (Total: {total_time:.4f}s, Loops: {len(loop_times)})")
                print(f"         Loop times (ms): {[f'{t:.2f}' for t in loop_times_ms]}")
                print(f"         Avg: {avg_loop:.3f}ms")
    
    # print Draft Inference Time > 0.05s Prompt statistics
    if high_draft_inference_prompts:
        print(f"\n📊 Iteration Num of Prompt Analysis (Draft Inference Time > 0.05s):")
        print(f"   Total cases: {len(high_draft_inference_prompts)}")
        
        from collections import Counter
        prompts = [p for s, t, p in high_draft_inference_prompts]
        prompt_counter = Counter(prompts)
        
        print(f"\n   Distribution:")
        for prompt_val in sorted(prompt_counter.keys()):
            count = prompt_counter[prompt_val]
            percentage = (count / len(prompts)) * 100
            avg_time = np.mean([t for s, t, p in high_draft_inference_prompts if p == prompt_val])
            print(f"      Iteration {prompt_val}: {count} cases ({percentage:.1f}%), Avg Time: {avg_time:.4f}s")
    
    print(f"   Dynamic Depth Range: {np.min(dynamic_depths)} - {np.max(dynamic_depths)}")
    print(f"   Average Dynamic Depth: {np.mean(dynamic_depths):.2f} ± {np.std(dynamic_depths):.2f}")
    print(f"   Task Detail Range: {np.min(task_details)} - {np.max(task_details)}")
    print(f"   Average Task Detail: {np.mean(task_details):.2f} ± {np.std(task_details):.2f}")
    
    # print Threshold statistics
    print(f"\n🎯 Threshold Statistics:")
    print(f"   Average Threshold: {np.mean(thresholds):.4f} ± {np.std(thresholds):.4f}")
    print(f"   Max Threshold: {np.max(thresholds):.4f}")
    print(f"   Min Threshold: {np.min(thresholds):.4f}")
    print(f"   Median Threshold: {np.median(thresholds):.4f}")
    
    # print KV Cache statistics
    print(f"\n💾 KV Cache Statistics:")
    print(f"   Average KV Cache Size: {np.mean(kv_cache_sizes):.0f} tokens")
    print(f"   Max KV Cache Size: {np.max(kv_cache_sizes):.0f} tokens")
    print(f"   Min KV Cache Size: {np.min(kv_cache_sizes):.0f} tokens")
    print(f"   KV Cache Growth Rate: {(kv_cache_sizes[-1] - kv_cache_sizes[0]) / max(len(kv_cache_sizes), 1):.2f} tokens/step")
    
    # 打印 GPU 指标统计
    if np.any(gpu_freq_mhz > 0) or np.any(gpu_power_w > 0) or np.any(gpu_util_percent > 0):
        print(f"\n🖥️  GPU Metrics Statistics:")
        
        # GPU Frequency
        if np.any(gpu_freq_mhz > 0):
            freq_nonzero = gpu_freq_mhz[gpu_freq_mhz > 0]
            print(f"   GPU Frequency: {np.mean(gpu_freq_mhz):.1f} ± {np.std(gpu_freq_mhz):.1f} MHz (min: {np.min(freq_nonzero):.0f}, max: {np.max(gpu_freq_mhz):.0f})")
        else:
            print(f"   GPU Frequency: No data available")
        
        # GPU Power
        if np.any(gpu_power_w > 0):
            power_nonzero = gpu_power_w[gpu_power_w > 0]
            print(f"   GPU Power: {np.mean(gpu_power_w):.1f} ± {np.std(gpu_power_w):.1f} W (min: {np.min(power_nonzero):.1f}, max: {np.max(gpu_power_w):.1f})")
        else:
            print(f"   GPU Power: No data available")
        
        # GPU Utilization
        if np.any(gpu_util_percent > 0):
            util_nonzero = gpu_util_percent[gpu_util_percent > 0]
            print(f"   GPU Utilization: {np.mean(gpu_util_percent):.1f} ± {np.std(gpu_util_percent):.1f} % (min: {np.min(util_nonzero):.0f}, max: {np.max(gpu_util_percent):.0f})")
        else:
            print(f"   GPU Utilization: No data available")
        
        # GPU Temperature
        if np.any(gpu_temp_c > 0):
            temp_nonzero = gpu_temp_c[gpu_temp_c > 0]
            print(f"   GPU Temperature: {np.mean(gpu_temp_c):.1f} ± {np.std(gpu_temp_c):.1f} °C (min: {np.min(temp_nonzero):.0f}, max: {np.max(gpu_temp_c):.0f})")
        else:
            print(f"   GPU Temperature: No data available")
        
        # Statistics of GPU metrics grouped by active_clients
        print(f"\n🔍 GPU Metrics by Active Clients:")
        unique_clients = sorted(set(active_clients))
        for num_clients in unique_clients:
            mask = active_clients == num_clients
            if np.any(mask):
                samples = np.sum(mask)
                print(f"   {num_clients} clients ({samples} samples):")
                print(f"      GPU Freq: {np.mean(gpu_freq_mhz[mask]):.1f} MHz")
                print(f"      GPU Power: {np.mean(gpu_power_w[mask]):.1f} W")
                print(f"      GPU Util: {np.mean(gpu_util_percent[mask]):.1f} %")
                print(f"      GPU Temp: {np.mean(gpu_temp_c[mask]):.1f} °C")
                print(f"      Avg Throughput: {np.mean(throughputs[mask]):.1f} tok/s")
                print(f"      Avg Iteration Time: {np.mean(last_iteration_times[mask]):.4f} s")
    
    # print abnormal inference cases (inference_time > 0.05s)
    if abnormal_cases:
        print(f"\n🚨 Abnormal Inference Cases Statistics:")
        print(f"   Total Abnormal Cases: {len(abnormal_cases)}")
        abnormal_times_only = [time for time, _ in abnormal_cases]
        print(f"   Average Abnormal Time: {np.mean(abnormal_times_only):.4f} ± {np.std(abnormal_times_only):.4f} sec")
        print(f"   Max Abnormal Time: {np.max(abnormal_times_only):.4f} sec")
        print(f"   Min Abnormal Time: {np.min(abnormal_times_only):.4f} sec")
        
        # Statistics by task type
        from collections import Counter
        task_counter = Counter([task for _, task in abnormal_cases])
        print(f"\n   Abnormal Cases by Task Type:")
        task_names = {-1: 'Unknown', 0: 'Initialize', 1: 'Update', 2: 'End'}
        for task_id, count in task_counter.items():
            task_name = task_names.get(task_id, f'Task {task_id}')
            percentage = (count / len(abnormal_cases)) * 100
            print(f"      {task_name}: {count} cases ({percentage:.1f}%)")
        
        # print latest 10 abnormal cases
        print(f"\n   Latest Abnormal Cases Details:")
        for i, (inf_time, task_detail) in enumerate(abnormal_cases, 1):
            task_name = task_names.get(task_detail, f'Task {task_detail}')
            severity = '🔴 CRITICAL' if inf_time > 0.5 else ('🟡 WARNING' if inf_time > 0.2 else '🟢 MILD')
            print(f"      #{i}: {inf_time:.4f}s - {task_name} - {severity}")
    else:
        print(f"\n✅ No abnormal inference cases recorded (all < 0.05s)")
    
    plt.show()

def main():
    """main"""
    # 🔧 Modify experiment data directory here
    save_dir = '/home/tmp/snap/firmware-updater/167/Desktop/hejunhao/StarSD/experiment_data/experiment_1769411299'
    
    # 🔧 Filter number of active_clients
    # Set to a specific number (e.g., 1) to only use data with active_clients equal to that value for plotting
    # Set to None to use all data
    filter_active_clients = None  # e.g., 1 means only use data with active_clients=1
    
    # Get filter value from user input
    user_input = 1

    if user_input:
        try:
            filter_active_clients = int(user_input)
            print(f"✅ Set filter condition: active_clients = {filter_active_clients}")
        except ValueError:
            print(f"⚠️ Invalid input, using all data")
            filter_active_clients = None
    
    # Automatically construct data file path
    data_file = str(Path(save_dir) / 'system_information.json')

    print(f"📊 Analyzing data from: {data_file}")
    print(f"💾 Saving plots to: {save_dir}")
    
    # Check if data file exists
    if not Path(data_file).exists():
        print(f"❌ Error: Data file {data_file} not found!")
        return
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # generate plots
    plot_system_throughput(data_file, save_dir, filter_active_clients)

if __name__ == "__main__":
    main()