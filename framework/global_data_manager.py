import threading
import queue
import time
from typing import Dict, Any, Optional
import uuid
from collections import OrderedDict

class GlobalDataManager:
    """Global data manager - Manages data and state for multiple Base models"""
    
    def __init__(self):
        self.clients_data: Dict[str, Any] = {}  # {tag: client_data}
        self.clients_queues: Dict[str, queue.Queue] = {}  # {tag: message_queue}
        self.clients_status: Dict[str, str] = {}  # {tag: status}
        self.lock = threading.RLock()  # Thread-safe lock
        self.port_allocator = 25000  # Private port allocation starting point
        self.allocated_ports: Dict[str, int] = {}  # {tag: private_port}

        # Inference task management
        self.inference_tasks: OrderedDict[str, Dict[str, Any]] = OrderedDict()  # {tag: task_data}
        self.result_queues: Dict[str, queue.Queue] = {}  # {tag: result_queue}
        self.inference_lock = threading.RLock()
        
    def register_client(self, client_tag: str = None) -> tuple[str, int]:
        """Register a new client and return the tag and allocated private port"""
        with self.lock:
            if client_tag is None:
                client_tag = f"client_{uuid.uuid4().hex[:8]}"

            # Check if the client is already registered, if so, reuse it
            if client_tag in self.clients_data:
                print(f"🔄 Reusing existing client: {client_tag} -> Port {self.allocated_ports[client_tag]}")
                # Clear old queues and status, prepare for new session
                self.clients_queues[client_tag] = queue.Queue()
                self.clients_status[client_tag] = "registered"
                # Clear inference-related tasks
                with self.inference_lock:
                    self.inference_tasks.pop(client_tag, None)
                    if client_tag in self.result_queues:
                        # Clear old result queue
                        while not self.result_queues[client_tag].empty():
                            try:
                                self.result_queues[client_tag].get_nowait()
                            except queue.Empty:
                                break
                return client_tag, self.allocated_ports[client_tag]

            # Allocate private port (find available port)
            private_port = self._find_available_port()

            # Initialize client data
            self.clients_data[client_tag] = {}
            self.clients_queues[client_tag] = queue.Queue()
            self.clients_status[client_tag] = "registered"
            self.allocated_ports[client_tag] = private_port
            
            print(f"✅ Client registered: {client_tag} -> Port {private_port}")
            return client_tag, private_port
    
    def _find_available_port(self) -> int:
        """Find an available port"""
        import socket

        # Check existing allocated port range
        used_ports = set(self.allocated_ports.values())

        # Find available port starting from the allocator
        for port in range(self.port_allocator, self.port_allocator + 100):
            if port not in used_ports:
                # Try to bind the port to confirm availability
                try:
                    test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    test_socket.bind(('localhost', port))
                    test_socket.close()
                    self.port_allocator = max(self.port_allocator, port + 1)
                    return port
                except OSError:
                    continue
                    
        raise Exception("No available ports in range")
    
    def update_client_data(self, client_tag: str, data: Any):
        """Update client data"""
        with self.lock:
            if client_tag in self.clients_data:
                self.clients_data[client_tag] = data
                # Put into queue for processing thread
                self.clients_queues[client_tag].put(data)
                
    def get_client_data(self, client_tag: str) -> Optional[Any]:
        """Get client data"""
        with self.lock:
            return self.clients_data.get(client_tag)
    
    def get_client_queue(self, client_tag: str) -> Optional[queue.Queue]:
        """Get client message queue"""
        return self.clients_queues.get(client_tag)
    
    def update_client_status(self, client_tag: str, status: str):
        """Update client status"""
        with self.lock:
            if client_tag in self.clients_status:
                self.clients_status[client_tag] = status
                
    def get_client_status(self, client_tag: str) -> str:
        """Get client status"""
        return self.clients_status.get(client_tag, "unknown")
    
    def remove_client(self, client_tag: str):
        """Remove client"""
        with self.lock:
            self.clients_data.pop(client_tag, None)
            self.clients_queues.pop(client_tag, None)
            self.clients_status.pop(client_tag, None)
            port = self.allocated_ports.pop(client_tag, None)
            print(f"❌ Client removed: {client_tag} (Port {port})")
    
    def get_all_clients(self) -> Dict[str, str]:
        """Get all client statuses"""
        with self.lock:
            return self.clients_status.copy()
    
    def get_client_port(self, client_tag: str) -> Optional[int]:
        """Get client private port"""
        return self.allocated_ports.get(client_tag)
    
    def add_inference_task(self, client_tag: str, task_data: Dict[str, Any]):
        """Add inference task to global queue"""
        import time
        with self.inference_lock:
            task_data['task_submit_time'] = time.time()
            self.inference_tasks[client_tag] = task_data
            # Ensure result queue exists
            if client_tag not in self.result_queues:
                self.result_queues[client_tag] = queue.Queue()
            print(f"📝 Added inference task for {client_tag}, queue size: {len(self.inference_tasks)}")
    
    def get_next_inference_task(self) -> Optional[tuple[str, Dict[str, Any]]]:
        """Get the next inference task (FIFO)"""
        with self.inference_lock:
            if self.inference_tasks:
                client_tag, task_data = self.inference_tasks.popitem(last=False)  # FIFO
                print(f"🎯 Processing task for {client_tag}, remaining: {len(self.inference_tasks)}")
                return client_tag, task_data
            return None
    
    def has_pending_tasks(self) -> bool:
        """Check if there are pending inference tasks"""
        with self.inference_lock:
            return len(self.inference_tasks) > 0
    
    def get_pending_tasks_count(self) -> int:
        """Get the number of pending inference tasks"""
        with self.inference_lock:
            return len(self.inference_tasks)

    
    def put_inference_result(self, client_tag: str, result: Any):
        """Put inference result into the corresponding client's result queue"""
        with self.inference_lock:
            if client_tag in self.result_queues:
                self.result_queues[client_tag].put(result)
                # print(f"✅ Put result for {client_tag}")
            else:
                print(f"⚠️ No result queue for {client_tag}")
    
    def get_inference_result(self, client_tag: str, timeout: float = None) -> Optional[Any]:
        """Get inference result"""
        result_queue = self.result_queues.get(client_tag)
        if result_queue:
            try:
                return result_queue.get(timeout=timeout)
            except queue.Empty:
                return None
        return None
    
    def clear_client_tasks(self, client_tag: str):
        """Clear all tasks and results for a client"""
        with self.inference_lock:
            self.inference_tasks.pop(client_tag, None)
            self.result_queues.pop(client_tag, None)

global_data_manager = GlobalDataManager()