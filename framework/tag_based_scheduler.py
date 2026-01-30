import threading
import time
from typing import Dict, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from framework.global_data_manager import global_data_manager
from eagle.model.draft_inference import EaModel
from eagle.model.config_loader import config
from framework.single_inference_engine import initialize_inference_engine, get_inference_engine
import torch


@dataclass
class CommunicationMetrics:
    """Communication metrics container for round-trip statistics"""
    base_loop_time: float = 0.0
    base_inference_time: float = 0.0
    base_to_draft_transfer_time: float = 0.0
    draft_to_base_transfer_time: float = 0.0
    draft_to_base_packet_size: float = 0.0
    base_to_draft_packet_size: float = 0.0
    
    @property
    def round_trip_time(self) -> float:
        return self.draft_to_base_transfer_time + self.base_to_draft_transfer_time
    
    @property
    def round_trip_packet_size(self) -> float:
        return self.draft_to_base_packet_size + self.base_to_draft_packet_size


class UpdateDataParser:
    """Parser for update data received from Base Model"""
    
    @staticmethod
    def parse_timestamp(data: tuple, recv_timestamp: float) -> Tuple[float, tuple]:
        """Extract timestamp from data tuple end, return (transfer_time, cleaned_data)"""
        if not isinstance(data, (tuple, list)) or len(data) < 2:
            return 0.0, data
        
        last_elem = data[-1]
        if isinstance(last_elem, (int, float)) and abs(last_elem - recv_timestamp) < 10:
            transfer_time = recv_timestamp - last_elem
            return transfer_time, data[:-1]
        return 0.0, data
    
    @staticmethod
    def parse_communication_metrics(data: tuple, recv_timestamp: float, 
                                     base_to_draft_packet_size: float) -> Tuple[CommunicationMetrics, tuple]:
        """Parse communication metrics from update data, return (metrics, cleaned_data)"""
        metrics = CommunicationMetrics(base_to_draft_packet_size=base_to_draft_packet_size)
        
        if not isinstance(data, (tuple, list)) or len(data) < 5:
            return metrics, data
        
        # Check for timestamp at the end
        if len(data) > 5 and isinstance(data[-1], (int, float)):
            send_timestamp = data[-1]
            metrics.base_to_draft_transfer_time = recv_timestamp - send_timestamp
            
            # Check for draft→base metrics at -3 and -2 positions
            if len(data) > 7 and isinstance(data[-3], (int, float)) and isinstance(data[-2], (int, float)):
                metrics.draft_to_base_transfer_time = data[-3]
                metrics.draft_to_base_packet_size = data[-2]
                data = data[:-3]  # Remove trailing metrics
            else:
                data = data[:-1]  # Remove only timestamp
        
        # Check for base_loop_time at position [4]
        if len(data) >= 5 and isinstance(data[4], (int, float)):
            metrics.base_inference_time = data[4]
        if len(data) >= 6 and isinstance(data[5], (int, float)):
            metrics.base_loop_time = data[5]
        
        return metrics, data
    
    @staticmethod
    def parse_update_payload(data: tuple) -> Optional[Dict[str, Any]]:
        """Parse core update payload (hidden_state, InputIds, accept_length, idx)"""
        if not isinstance(data, (tuple, list)) or len(data) < 4:
            return None
        
        return {
            'accpt_hidden_state_new': data[0],
            'input_ids': data[1],
            'accept_length': data[2],
            'iteration': data[3]
        }


class TagBasedScheduler:
    """Tag-based scheduler: assigns a dedicated communication thread per client and uses a shared inference engine."""
    
    def __init__(self, draft_model: EaModel):
        self.draft_model = draft_model
        self.processing_threads: Dict[str, threading.Thread] = {}
        self.active_clients: Dict[str, bool] = {}
        self.lock = threading.RLock()
        
        # initialize the global inference engine
        initialize_inference_engine(draft_model)
        
    def start_client_thread(self, client_tag: str, private_communicator):
        """Start a dedicated processing thread for the client."""
        with self.lock:
            if client_tag in self.processing_threads:
                print(f"⚠️ Thread for {client_tag} already exists")
                return
            
            # create processing thread
            thread = threading.Thread(
                target=self._process_client_requests,
                args=(client_tag, private_communicator),
                daemon=True,
                name=f"ProcessThread-{client_tag}"
            )
            
            self.processing_threads[client_tag] = thread
            self.active_clients[client_tag] = True
            thread.start()
            
            print(f"🚀 Started processing thread for {client_tag}")
    
    def _process_client_requests(self, client_tag: str, communicator):
        """Process a single client's communication requests — supports multi-turn sessions."""
        print(f"📡 Communication thread started for {client_tag}")
        global_data_manager.update_client_status(client_tag, "processing")
        
        # outer loop: supports multi-turn conversation sessions
        while self.active_clients.get(client_tag, False):
            try:
                print(f"🔄 Starting new session for {client_tag}")
                session_completed = self._handle_single_session(client_tag, communicator)
                
                if not session_completed:
                    # if session did not complete normally (e.g., client requested close), break loop
                    break
                    
                # session completed, ready to accept the next conversation
                print(f"✅ Session completed for {client_tag}, ready for next")
                
            except Exception as e:
                print(f"💥 Session error for {client_tag}: {e}")
                import traceback
                traceback.print_exc()
                # if a session error occurs, continue listening for new sessions unless disconnected
                continue
                
        print(f"🔌 Client {client_tag} connection ending")
        self._cleanup_client(client_tag, communicator)
        
    def _handle_single_session(self, client_tag: str, communicator) -> bool:
        """
        Handle a single conversation session.
        
        Returns:
            True if session completed normally (ready for next session)
            False if client disconnected or error occurred
        """
        # Initialize transfer time statistics
        total_base_to_draft_transfer_time = 0.0
        
        try:
            # ============== Phase 1: Initialization ==============
            if not self._handle_initialization_phase(client_tag, communicator):
                return False
            
            # ============== Phase 2: Main Update Loop ==============
            return self._handle_update_loop(client_tag, communicator, total_base_to_draft_transfer_time)
                    
        except Exception as e:
            print(f"💥 Fatal error in session for {client_tag}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _handle_initialization_phase(self, client_tag: str, communicator) -> bool:
        """Handle the initialization phase of a session. Returns True if successful."""
        print(f"⏳ Waiting for initial input from {client_tag}")
        
        # Receive initial data
        received_data = communicator.recv_vars()
        recv_timestamp = time.time()
        
        # Parse timestamp if present
        transfer_time, received_data = UpdateDataParser.parse_timestamp(received_data, recv_timestamp)
        if transfer_time > 0:
            print(f"📥 [Init] Base→Draft: {transfer_time*1000:.2f}ms")
        
        # Check for disconnect signal
        if self._is_disconnect_signal(received_data):
            print(f"🔚 Client {client_tag} requested disconnect")
            self.active_clients[client_tag] = False
            return False
        
        # Forward input_ids to base model
        input_ids = received_data[0] if isinstance(received_data, tuple) else received_data
        print(f"📤 Forwarding input_ids from {client_tag} to base model")
        
        # Wait for init data from base model
        init_data = communicator.recv_vars()
        print(f"📥 Received init data from base model for {client_tag}")
        
        # Submit initialization task
        task_data = {
            'type': 'initialize',
            'hidden_state': init_data[0],
            'input_ids': init_data[1],
            'iteration': init_data[2]
        }
        global_data_manager.add_inference_task(client_tag, task_data)
        
        # Wait for inference result
        result = global_data_manager.get_inference_result(client_tag, timeout=30)
        if result is None:
            print(f"⏰ Timeout waiting for init result for {client_tag}")
            return False
        
        # Process result
        return self._send_init_result(client_tag, communicator, result)
    
    def _send_init_result(self, client_tag: str, communicator, result: dict) -> bool:
        """Send initialization result to base model. Returns True if successful."""
        result_type = result.get('type', 'unknown')
        
        if result_type == 'initialize_result':
            communicator.send_vars(
                result['draft_tokens'],
                result['retrieve_indices'],
                result['tree_mask'],
                result['tree_position_ids'],
                result['total_tokens'],
                include_timestamp=True
            )
            print(f"📤 Sent init result to base model for {client_tag}")
            return True
        elif result_type == 'cache_exists':
            print(f"🔄 Reusing existing KV cache for {client_tag}")
            return True
        elif result_type == 'error':
            print(f"❌ Initialize error for {client_tag}: {result.get('error', 'Unknown error')}")
            return False
        else:
            print(f"⚠️ Unexpected result type for {client_tag}: {result_type}")
            return False
    
    def _handle_update_loop(self, client_tag: str, communicator, 
                           total_base_to_draft_transfer_time: float) -> bool:
        """Handle the main update loop. Returns True if session ended normally."""
        new_token = 0
        
        while self.active_clients.get(client_tag, False):
            try:
                # Receive update data from base model
                update_data, base_to_draft_packet_size, recv_rtt = communicator.recv_vars(return_packet_size=True)
                recv_timestamp = time.time()
                
                # Check for disconnect or special signals
                signal_result = self._handle_special_signals(client_tag, update_data)
                if signal_result == 'disconnect':
                    return False
                elif signal_result == 'continue':
                    continue
                
                # Validate data format
                if not self._validate_update_data(client_tag, update_data):
                    continue
                
                # Handle end signal
                if update_data[0] == 'end':
                    return self._handle_end_signal(client_tag, update_data, total_base_to_draft_transfer_time)
                
                # Parse communication metrics using helper class
                metrics, cleaned_data = UpdateDataParser.parse_communication_metrics(
                    update_data, recv_timestamp, base_to_draft_packet_size
                )
                
                # Accumulate transfer time
                if metrics.base_to_draft_transfer_time > 0:
                    total_base_to_draft_transfer_time += metrics.base_to_draft_transfer_time
                    if metrics.draft_to_base_transfer_time > 0:
                        print(f"🔄 Round-trip: Draft→Base {metrics.draft_to_base_transfer_time*1000:.2f}ms "
                              f"({metrics.draft_to_base_packet_size:.2f}KB) | "
                              f"Base→Draft {metrics.base_to_draft_transfer_time*1000:.2f}ms")
                
                if metrics.base_loop_time > 0:
                    print(f"⏱️ Base loop: {metrics.base_loop_time*1000:.2f}ms")
                
                # Parse core payload
                payload = UpdateDataParser.parse_update_payload(cleaned_data)
                if payload is None:
                    print(f"⚠️ Invalid update_data from {client_tag}")
                    continue
                
                # Build and submit update task
                update_task = self._build_update_task(payload, new_token, metrics)
                global_data_manager.add_inference_task(client_tag, update_task)
                
                # Wait for and send result
                new_token = self._wait_and_send_update_result(client_tag, communicator)
                if new_token is None:
                    return False
                    
            except Exception as e:
                print(f"❌ Error in communication loop for {client_tag}: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        return False  # Exited loop without end signal
    
    def _is_disconnect_signal(self, data) -> bool:
        """Check if data is a disconnect signal."""
        return isinstance(data, str) and data in ("close server", "quit")
    
    def _handle_special_signals(self, client_tag: str, update_data) -> str:
        """Handle special string signals. Returns 'disconnect', 'continue', or 'process'."""
        if isinstance(update_data, str):
            if update_data in ("quit", "close server"):
                print(f"🔚 Client {client_tag} requested disconnect in main loop")
                self.active_clients[client_tag] = False
                return 'disconnect'
            elif update_data == 'NACK':
                print(f"📤 Resending previous result for {client_tag}")
                return 'continue'
        return 'process'
    
    def _validate_update_data(self, client_tag: str, update_data) -> bool:
        """Validate update data format. Returns True if valid."""
        if not isinstance(update_data, (tuple, list)):
            print(f"⚠️ Unexpected data type from {client_tag}: {type(update_data)}")
            return False
        if len(update_data) < 1:
            print(f"⚠️ Empty data from {client_tag}")
            return False
        return True
    
    def _handle_end_signal(self, client_tag: str, update_data: tuple,
                          total_base_to_draft_transfer_time: float) -> bool:
        """Handle end signal from base model. Returns True to indicate session complete."""
        print(f"🔚 Received end signal for {client_tag}")
        
        # Extract transfer time statistics
        total_draft_to_base_time = 0.0
        if len(update_data) > 3 and isinstance(update_data[3], (int, float)):
            total_draft_to_base_time = update_data[3]
        
        # Print transfer statistics summary
        print(f"\n{'='*60}")
        print(f"📊 Transfer Time Statistics for {client_tag}")
        print(f"{'='*60}")
        print(f"📤 Base→Draft total: {total_base_to_draft_transfer_time*1000:.2f}ms")
        print(f"📥 Draft→Base total: {total_draft_to_base_time*1000:.2f}ms")
        print(f"🔄 Round-trip total: {(total_base_to_draft_transfer_time + total_draft_to_base_time)*1000:.2f}ms")
        print(f"{'='*60}\n")
        
        # Submit end task
        end_task = {
            'type': 'end',
            'message': 'Session ended',
            'generated_tokens': update_data[1],
            'avg_speed_per_prompt': update_data[2],
            'base_model_iteration_speed': update_data[5] if len(update_data) > 5 else 0.0,
            'prompt_iteration_count': update_data[6] if len(update_data) > 6 else 0
        }
        global_data_manager.add_inference_task(client_tag, end_task)
        
        # Wait for end task result
        end_result = global_data_manager.get_inference_result(client_tag, timeout=10)
        if end_result:
            print(f"✅ End task processed for {client_tag}")
        
        print(f"🔄 Session ended for {client_tag}, ready for next")
        return True
    
    def _build_update_task(self, payload: dict, new_token: int, 
                          metrics: CommunicationMetrics) -> dict:
        """Build update task dictionary from payload and metrics."""
        return {
            'type': 'update',
            'accpt_hidden_state_new': payload['accpt_hidden_state_new'],
            'input_ids': payload['input_ids'],
            'accept_length': payload['accept_length'],
            'new_token': new_token,
            "base_inference_time": metrics.base_inference_time,
            'base_loop_time': metrics.base_loop_time,
            'iteration': payload['iteration'],
            # Communication metrics - complete round-trip data
            'draft_to_base_transfer_time': metrics.draft_to_base_transfer_time,
            'draft_to_base_packet_size': metrics.draft_to_base_packet_size,
            'base_to_draft_transfer_time': metrics.base_to_draft_transfer_time,
            'base_to_draft_packet_size': metrics.base_to_draft_packet_size,
            'round_trip_time': metrics.round_trip_time,
            'round_trip_packet_size': metrics.round_trip_packet_size
        }
    
    def _wait_and_send_update_result(self, client_tag: str, communicator) -> Optional[int]:
        """Wait for update result and send to base model. Returns new_token or None on error."""
        update_result = global_data_manager.get_inference_result(client_tag, timeout=30)
        if update_result is None:
            print(f"⏰ Timeout waiting for update result for {client_tag}")
            return None
        
        result_type = update_result.get('type', 'unknown')
        
        if result_type == 'update_result':
            communicator.send_vars(
                update_result['draft_tokens'],
                update_result['retrieve_indices'],
                update_result['tree_mask'],
                update_result['tree_position_ids'],
                update_result['new_token'],
                include_timestamp=True
            )
            return update_result['new_token']
        elif result_type == 'error':
            print(f"❌ Update error for {client_tag}: {update_result.get('error', 'Unknown error')}")
            return None
        else:
            print(f"⚠️ Unexpected update result type for {client_tag}: {result_type}")
            return None
    

    
    def stop_client_thread(self, client_tag: str):
        """Stop a client's processing thread."""
        with self.lock:
            if client_tag in self.active_clients:
                self.active_clients[client_tag] = False
                
            # mark client as inactive
            print(f"🛑 Stopping client thread {client_tag}")
    
    def _cleanup_client(self, client_tag: str, communicator):
        """Clean up client resources."""
        print(f"🧹 Cleaning up client {client_tag}")
        
        with self.lock:
            self.active_clients.pop(client_tag, None)
            self.processing_threads.pop(client_tag, None)
        
        # clear KV cache
        if hasattr(self.draft_model, 'ea_layer'):
            self.draft_model.ea_layer.reset_kv(client_tag)
            print("CLIENT TAG:", client_tag)
            print(f"🗑️ Cleared KV cache for {client_tag}")
        
        # clear tasks and results from the global data manager
        global_data_manager.clear_client_tasks(client_tag)
        
        try:
            communicator.close()
        except:
            pass
            
        global_data_manager.remove_client(client_tag)
    
    def get_active_clients(self) -> Dict[str, bool]:
        """Return a copy of the active clients mapping."""
        return self.active_clients.copy()
    
    def shutdown_all(self):
        """Shutdown all client threads and the inference engine."""
        print("🛑 Shutting down all client threads...")
        
        # stop all clients
        client_list = list(self.active_clients.keys())
        for client_tag in client_list:
            self.stop_client_thread(client_tag)
        
        # wait for threads to finish
        for thread in list(self.processing_threads.values()):
            if thread.is_alive():
                thread.join(timeout=5)
                
        # stop global inference engine
        inference_engine = get_inference_engine()
        if inference_engine:
            inference_engine.stop()
            print("🛑 Inference engine stopped")