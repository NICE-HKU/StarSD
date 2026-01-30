from eagle.model.draft_inference import EaModel
from eagle.model.communicator import FastDistributedCommunicator
import torch
import threading
import socket
import json
from eagle.model.config_loader import config
from framework.global_data_manager import global_data_manager
from framework.tag_based_scheduler import TagBasedScheduler
import time

class DraftServer:
    """Draft model server that handles connections from Base models and manages inference scheduling."""
    
    def __init__(self, public_port: int = None):
        self.public_port = public_port or config.communication_port
        self.draft_model = None
        self.scheduler = None
        self.server_socket = None
        self.running = False
        self.accept_thread = None
        
    def initialize_model(self):
        """Initialize Draft model"""
        print("🚀 Initializing Draft model...")
        
        self.draft_model = EaModel.from_pretrained(
            ea_model_path=config.draft_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=False,
            use_eagle3=False,
            client=None,
            depth=7,
            top_k=10
        )
        
        print("Device distribution:")
        print(f"1. base_model.lm_head: {self.draft_model.base_model.lm_head.weight.device}")
        print(f"2. ea_layer: {self.draft_model.ea_layer.fc.weight.device}")
        print(f"3. input_embedding_layer: {self.draft_model.base_model.model.embed_tokens.weight.device}")
        
        self.draft_model.eval()

        # Initialize scheduler
        self.scheduler = TagBasedScheduler(self.draft_model)
        
        print("✅ Draft model initialized")
    
    def start_server(self):
        """Start the Draft server to accept connections from Base models"""
        print(f"🌐 Starting Draft server on port {self.public_port}")

        # Create server socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('0.0.0.0', self.public_port))
        self.server_socket.listen(10)  # Support 10 concurrent connections

        self.running = True

        # Start thread to accept connections
        self.accept_thread = threading.Thread(target=self._accept_connections, daemon=True)
        self.accept_thread.start()
        
        print(f"✅ Draft server listening on port {self.public_port}")
        print("Waiting for Base model connections...")
    
    def _accept_connections(self):
        """accept incoming connections from Base models"""
        while self.running:
            try:
                client_socket, client_address = self.server_socket.accept()
                print(f"🔗 New connection from {client_address}")

                # Handle handshake in a new thread
                handshake_thread = threading.Thread(
                    target=self._handle_handshake,
                    args=(client_socket, client_address),
                    daemon=True
                )
                handshake_thread.start()
                
            except Exception as e:
                if self.running:
                    print(f"❌ Error accepting connection: {e}")
    
    def _handle_handshake(self, client_socket, client_address):
        """Handle handshake with Base model"""
        try:
            # Register new client
            client_tag, private_port = global_data_manager.register_client()
            
            # Send allocation info to Base model
            response = {
                "status": "success",
                "client_tag": client_tag,
                "private_port": private_port,
                "message": f"Allocated private port {private_port} for {client_tag}"
            }
            
            response_data = json.dumps(response).encode('utf-8')
            client_socket.send(len(response_data).to_bytes(4, byteorder='big'))
            client_socket.send(response_data)
            
            print(f"📤 Sent allocation info to {client_address}: {client_tag} -> {private_port}")

            # Close the handshake socket
            client_socket.close()

            # Start private communication server
            self._start_private_server(client_tag, private_port)
            
        except Exception as e:
            print(f"❌ Handshake error with {client_address}: {e}")
            try:
                client_socket.close()
            except:
                pass
    
    def _start_private_server(self, client_tag: str, private_port: int):
        """Start private server for a specific client"""
        try:
            # Create private communicator
            private_communicator = FastDistributedCommunicator(
                host='0.0.0.0',
                port=private_port,
                is_server=True,
                max_retries=10
            )
            
            print(f"🔒 Private server started for {client_tag} on port {private_port}")

            # Start client processing thread
            self.scheduler.start_client_thread(client_tag, private_communicator)
            
        except Exception as e:
            print(f"❌ Failed to start private server for {client_tag}: {e}")
            global_data_manager.remove_client(client_tag)
    
    def stop_server(self):
        """stop the Draft server and all client connections"""
        print("🛑 Stopping Draft server...")
        
        self.running = False

        # Close all clients
        if self.scheduler:
            self.scheduler.shutdown_all()
        
        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        print("✅ Draft server stopped")
    
    def get_server_status(self):
        """Get server status"""
        if not self.running:
            return {"status": "stopped"}
        
        return {
            "status": "running",
            "public_port": self.public_port,
            "active_clients": global_data_manager.get_all_clients(),
            "total_clients": len(global_data_manager.get_all_clients())
        }

def main():
    """Main function"""
    server = DraftServer()
    
    try:
        # Initialize model
        server.initialize_model()

        # Start server
        server.start_server()

        # Keep running
        print("\n" + "="*50)
        print("Draft Server is running!")
        print("Press Ctrl+C to stop the server")
        print("="*50 + "\n")
        
        while True:
            time.sleep(20)
            # print state
            status = server.get_server_status()
            if status["status"] == "running" and status["total_clients"] > 0:
                print(f"📊 Active clients: {status['total_clients']}")
            
    except KeyboardInterrupt:
        print("\n🔴 Received shutdown signal")
    finally:
        server.stop_server()

if __name__ == "__main__":
    main()