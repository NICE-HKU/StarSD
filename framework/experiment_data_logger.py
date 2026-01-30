#!/usr/bin/env python3
"""
Experiment Data Logger for StarSD System

Features:
1. Record system information: throughput and active client counts
2. Automatically create experiment folders
3. Periodically save JSON data
"""

import json
import os
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path


class ExperimentDataLogger:
    """Experiment data logger"""
    
    def __init__(self, experiment_name: str = None, base_dir: str = "experiment_data"):
        """
        Initialize the experiment data logger

        Args:
            experiment_name: Experiment name; if None a timestamp will be used
            base_dir: Base directory to store data
        """
        self.base_dir = Path(base_dir)
        self.experiment_name = experiment_name or f"experiment_{int(time.time())}"
        self.experiment_dir = self.base_dir / self.experiment_name
        
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.sac_training_data = []
        self.system_info_data = []
        
        self.sac_training_file = self.experiment_dir / "sac_training.json"
        self.system_info_file = self.experiment_dir / "system_information.json"
        self.metadata_file = self.experiment_dir / "experiment_metadata.json"
        
        self.sac_lock = threading.RLock()
        self.system_lock = threading.RLock()
        
        # Auto-save configuration
        self.auto_save_interval = 400  # auto-save every 500 seconds
        self.auto_save_thread = None
        self.running = False
        
        self.start_time = time.time()
        self.start_datetime = datetime.now()
        
        print(f"📊 Experiment data logger initialized")
        print(f"   Experiment name: {self.experiment_name}")
        print(f"   Data directory: {self.experiment_dir}")
        
        self._save_metadata()
        
        self.start_auto_save()
    
    def get_experiment_path(self) -> Path:
        """Return the path to the experiment data directory"""
        return self.experiment_dir
    
    def _save_metadata(self):
        """Save experiment metadata"""
        metadata = {
            "experiment_name": self.experiment_name,
            "start_time": self.start_time,
            "start_datetime": self.start_datetime.isoformat(),
            "data_files": {
                "sac_training": "sac_training.json",
                "system_information": "system_information.json"
            },
            "description": "EAGLE Multi-LLM SAC reinforcement learning experiment data"
        }
        
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def log_sac_training(self, q_loss: float, actor_loss: float, alpha_loss: float, 
                        reward: float, accept_ratio: float, step_count: int,
                        client_tag: str = None, additional_data: Dict = None,
                        next_state: List = None, done: bool = None):
        """
        Record SAC training data

        Args:
            q_loss: Q-network loss
            actor_loss: Actor network loss
            alpha_loss: Alpha loss
            reward: Reward value
            accept_ratio: Acceptance ratio
            step_count: Training step count
            client_tag: Client identifier
            additional_data: Additional data
            next_state: Next state
            done: Whether the episode is done
        """
        timestamp = time.time()
        relative_time = timestamp - self.start_time
        
        sac_entry = {
            "timestamp": timestamp,
            "relative_time": relative_time,
            "datetime": datetime.fromtimestamp(timestamp).isoformat(),
            "step_count": step_count,
            "client_tag": client_tag,
            "losses": {
                "q_loss": q_loss,
                "actor_loss": actor_loss,
                "alpha_loss": alpha_loss
            },
            "reward": reward,
            "accept_ratio": accept_ratio,
            "next_state": next_state,
            "done": done
        }
        
        # Add additional data if provided
        if additional_data:
            sac_entry.update(additional_data)
        
        with self.sac_lock:
            self.sac_training_data.append(sac_entry)
        
        print(f"🧠 SAC training entry recorded - Step: {step_count}, Q-Loss: {q_loss:.4f}, Reward: {reward:.4f}")
    
    def log_system_info(self, throughput: float, active_clients: int, 
                       total_requests: int = None, avg_response_time: float = None,
                       client_tag: str = None, additional_data: Dict = None,
                       accept_length_last_client: int = None, last_iteration_time: float = None,
                       inference_time: float = None, get_task_time: float = None, between_tags: list = None):
        """
        Record system information data

        Args:
            throughput: System throughput (tokens/sec or requests/sec)
            active_clients: Number of active clients
            total_requests: Total number of requests
            avg_response_time: Average response time
            client_tag: Client identifier
            additional_data: Additional data
            accept_length_last_client: Accept length from the last client
            last_iteration_time: Last iteration time
        """
        timestamp = time.time()
        relative_time = timestamp - self.start_time
        
        system_entry = {
            "throughput": throughput,
            "active_clients": active_clients,
            "total_requests": total_requests,
            "avg_response_time": avg_response_time,
            "client_tag": client_tag,
            "accept_length_last_client": accept_length_last_client,
            "last_iteration_time": last_iteration_time,
            "inference_time": inference_time,
            "get_task_time": get_task_time,
            "between_tags": between_tags
        }
        
        # Add additional data if provided
        if additional_data:
            system_entry.update(additional_data)
        
        with self.system_lock:
            self.system_info_data.append(system_entry)
        
        print(f"📊 System info recorded - Throughput: {throughput:.2f}, Active: {active_clients}")
    
    def save_data(self):
        """Manually save all data to files"""
        try:
            with self.sac_lock:
                if self.sac_training_data:
                    with open(self.sac_training_file, 'w', encoding='utf-8') as f:
                        json.dump(self.sac_training_data, f, indent=2, ensure_ascii=False)
                    print(f"💾 SAC training data saved: {len(self.sac_training_data)} records")

            with self.system_lock:
                if self.system_info_data:
                    with open(self.system_info_file, 'w', encoding='utf-8') as f:
                        json.dump(self.system_info_data, f, indent=2, ensure_ascii=False)
                    print(f"💾 System information saved: {len(self.system_info_data)} records")
            
            # Update metadata
            self._update_metadata()
            
        except Exception as e:
            print(f"❌ Error saving data: {e}")
    
    def _update_metadata(self):
        """Update experiment metadata"""
        try:
            metadata = {
                "experiment_name": self.experiment_name,
                "start_time": self.start_time,
                "start_datetime": self.start_datetime.isoformat(),
                "last_update": time.time(),
                "last_update_datetime": datetime.now().isoformat(),
                "data_counts": {
                    "sac_training_records": len(self.sac_training_data),
                    "system_info_records": len(self.system_info_data)
                },
                "data_files": {
                    "sac_training": "sac_training.json",
                    "system_information": "system_information.json"
                },
                "description": "StarSD experiment data"
            }
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"❌ Error updating metadata: {e}")
    
    def start_auto_save(self):
        """Start the auto-save thread"""
        if self.auto_save_thread is not None:
            return
        
        self.running = True
        self.auto_save_thread = threading.Thread(
            target=self._auto_save_loop,
            daemon=True,
            name="ExperimentDataAutoSave"
        )
        self.auto_save_thread.start()
        print(f"🔄 Auto-save started (interval: {self.auto_save_interval} seconds)")
    
    def _auto_save_loop(self):
        """Auto-save loop"""
        while self.running:
            try:
                time.sleep(self.auto_save_interval)
                if self.running:  
                    self.save_data()
            except Exception as e:
                print(f"❌ Auto-save error: {e}")
    
    def stop_auto_save(self):
        """Stop auto-save"""
        self.running = False
        if self.auto_save_thread:
            self.auto_save_thread.join(timeout=5)
            self.auto_save_thread = None
        print("⏹️ Auto-save stopped")
    
    def finalize_experiment(self):
        """Finalize experiment and save final data"""
        print(f"🏁 Finalizing experiment and saving final data...")
        
        self.stop_auto_save()

        self.save_data()
        
        self._generate_experiment_summary()
        
        print(f"✅ Experiment data logging complete")
        print(f"   Data directory: {self.experiment_dir}")
        print(f"   SAC training records: {len(self.sac_training_data)}")
        print(f"   System information records: {len(self.system_info_data)}")
    
    def _generate_experiment_summary(self):
        """Generate experiment summary"""
        end_time = time.time()
        duration = end_time - self.start_time
        
        summary = {
            "experiment_name": self.experiment_name,
            "duration": {
                "seconds": duration,
                "minutes": duration / 60,
                "hours": duration / 3600
            },
            "start_time": self.start_time,
            "end_time": end_time,
            "start_datetime": self.start_datetime.isoformat(),
            "end_datetime": datetime.now().isoformat(),
            "data_summary": {
                "sac_training_records": len(self.sac_training_data),
                "system_info_records": len(self.system_info_data),
                "total_records": len(self.sac_training_data) + len(self.system_info_data)
            }
        }
        
        if self.sac_training_data:
            rewards = [entry["reward"] for entry in self.sac_training_data]
            q_losses = [entry["losses"]["q_loss"] for entry in self.sac_training_data]
            
            summary["sac_statistics"] = {
                "avg_reward": sum(rewards) / len(rewards),
                "max_reward": max(rewards),
                "min_reward": min(rewards),
                "avg_q_loss": sum(q_losses) / len(q_losses),
                "training_episodes": len(self.sac_training_data)
            }
        
        if self.system_info_data:
            throughputs = [entry["throughput"] for entry in self.system_info_data if entry["throughput"]]
            active_clients = [entry["active_clients"] for entry in self.system_info_data]
            
            if throughputs:
                summary["system_statistics"] = {
                    "avg_throughput": sum(throughputs) / len(throughputs),
                    "max_throughput": max(throughputs),
                    "min_throughput": min(throughputs),
                    "avg_active_clients": sum(active_clients) / len(active_clients),
                    "max_active_clients": max(active_clients)
                }
        
        summary_file = self.experiment_dir / "experiment_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"📋 Experiment summary saved: {summary_file}")


_global_logger: Optional[ExperimentDataLogger] = None


def initialize_experiment_logger(experiment_name: str = None, base_dir: str = "experiment_data") -> ExperimentDataLogger:
    """
    Initialize global experiment logger

    Args:
        experiment_name: Experiment name
        base_dir: Base directory for data storage

    Returns:
        An ExperimentDataLogger instance
    """
    global _global_logger
    
    if _global_logger is not None:
        print("⚠️ Experiment logger already exists; closing the previous logger")
        _global_logger.finalize_experiment()
    
    _global_logger = ExperimentDataLogger(experiment_name, base_dir)
    return _global_logger


def get_experiment_logger() -> Optional[ExperimentDataLogger]:
    """Return the global experiment logger, or None if not initialized"""
    return _global_logger


def get_experiment_path() -> Optional[Path]:
    """Return the global experiment data directory path, or None if not initialized"""
    if _global_logger:
        return _global_logger.get_experiment_path()
    return None


def log_sac_training_data(q_loss: float, actor_loss: float, alpha_loss: float,
                         reward: float, accept_ratio: float, step_count: int,
                         client_tag: str = None, next_state: List = None, done: bool = None, **kwargs):
    """Convenience wrapper: log SAC training data"""
    if _global_logger:
        _global_logger.log_sac_training(
            q_loss, actor_loss, alpha_loss, reward, accept_ratio, 
            step_count, client_tag, kwargs, next_state, done
        )



def log_system_information(throughput: float = None, active_clients: int = None, 
                         total_requests: int = None, avg_response_time: float = None,
                         client_tag: str = None, 
                         accept_length_last_client: int = None, last_iteration_time: float = None,
                         inference_time: float = None, get_task_time: float = None, between_tags: list = None,
                         # compatibility parameter names
                         num_active_clients: int = None, system_throughput: float = None, 
                         avg_system_throughput: float = None, total_clients: int = None,
                         buffer_size: int = None, dynamic_depth: int = None,
                         **kwargs):
    """Convenience wrapper: log system information"""
    if _global_logger:
        # Handle compatibility of parameter names
        final_throughput = throughput or system_throughput or 0.0
        final_active_clients = active_clients or num_active_clients or 0
        
        # Build extra info
        extra_info = kwargs.copy()
        if avg_system_throughput is not None:
            extra_info['avg_system_throughput'] = avg_system_throughput
        if total_clients is not None:
            extra_info['total_clients'] = total_clients
        if buffer_size is not None:
            extra_info['buffer_size'] = buffer_size
        if dynamic_depth is not None:
            extra_info['dynamic_depth'] = dynamic_depth
        
        _global_logger.log_system_info(
            final_throughput, final_active_clients, total_requests, 
            avg_response_time, client_tag, extra_info,
            accept_length_last_client, last_iteration_time,
            inference_time, get_task_time, between_tags
        )


def finalize_experiment():
    """Convenience wrapper: finalize experiment logging"""
    global _global_logger
    if _global_logger:
        _global_logger.finalize_experiment()
        _global_logger = None


    # Example usage
if __name__ == "__main__":
    # Create an experiment logger
    logger = initialize_experiment_logger("test_experiment")
    
    # Simulate logging SAC training data
    for i in range(5):
        log_sac_training_data(
            q_loss=0.5 - i*0.05,
            actor_loss=0.3 - i*0.02,
            alpha_loss=0.1,
            reward=1.0 + i*0.1,
            accept_ratio=0.7 + i*0.05,
            step_count=i+1,
            client_tag=f"client_{i%2}"
        )
        time.sleep(1)
    
    # Simulate logging system information
    for i in range(10):
        log_system_information(
            throughput=100.0 + i*10,
            active_clients=2 + i%3,
            total_requests=50 + i*5,
            avg_response_time=0.5 + i*0.1
        )
        time.sleep(0.5)
    
    # Finalize experiment
    finalize_experiment()
    print("✅ Example experiment complete")