from eagle.model.base_model_inference import EaModel
import torch
from fastchat.model import get_conversation_template
import json
import socket
import time
import logging
import traceback
from datetime import datetime
from pathlib import Path
from eagle.model.config_loader import config

class BaseModelClient:
    """Base model client - connects to the Draft server"""
    
    def __init__(self):
        self.model = None
        self.client_tag = None
        self.private_port = None
        self.server_communicator = None
        self.logger = None
        self._setup_logger()
    
    def _setup_logger(self):
        """Configure the logger"""
        log_dir = Path("base_client_logs")
        log_dir.mkdir(exist_ok=True)
        
        # Use timestamp for log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"base_client_{timestamp}.log"
        
        # Configure logger
        self.logger = logging.getLogger(f"BaseClient_{timestamp}")
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler - detailed logs
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        # Console handler - simplified output
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"📝 Log file created: {log_file}")
        self.logger.info("="*60)
        
    def connect_to_draft_server(self, server_host='127.0.0.1', server_port=None):
        """Connect to the Draft server and obtain allocation information"""
        server_port = server_port or config.communication_port
        
        self.logger.info(f"🔗 Connecting to Draft server at {server_host}:{server_port}")
        
        try:
            # Connect to the public port
            handshake_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            handshake_socket.connect((server_host, server_port))
            self.logger.debug(f"Handshake socket connected to {server_host}:{server_port}")
            
            # Receive allocation information
            length_bytes = handshake_socket.recv(4)
            if len(length_bytes) < 4:
                error_msg = "Failed to receive response length"
                self.logger.error(error_msg)
                raise Exception(error_msg)
                
            response_length = int.from_bytes(length_bytes, byteorder='big')
            response_data = handshake_socket.recv(response_length)
            self.logger.debug(f"Received response data: {len(response_data)} bytes")
            
            handshake_socket.close()
            
            # Parse the response
            response = json.loads(response_data.decode('utf-8'))
            self.logger.debug(f"Server response: {response}")
            
            if response["status"] != "success":
                error_msg = f"Server rejected connection: {response}"
                self.logger.error(error_msg)
                raise Exception(error_msg)
            
            self.client_tag = response["client_tag"]
            self.private_port = response["private_port"]
            
            self.logger.info(f"✅ Connected to Draft server")
            self.logger.info(f"   Client Tag: {self.client_tag}")
            self.logger.info(f"   Private Port: {self.private_port}")
            
            # Connect to the private port
            self._connect_private_channel(server_host)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Draft server: {e}")
            self.logger.debug(f"Traceback:\n{traceback.format_exc()}")
            raise
    
    def _connect_private_channel(self, server_host):
        """Connect to the private communication channel"""
        from eagle.model.communicator import FastDistributedCommunicator
        
        self.logger.info(f"🔒 Connecting to private channel on port {self.private_port}")
        
        try:
            # Wait for the server to start the private port
            time.sleep(2)
            
            self.server_communicator = FastDistributedCommunicator(
                host=server_host,
                port=self.private_port,
                is_server=False,
                max_retries=20
            )
            
            self.logger.info(f"✅ Connected to private channel")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to private channel: {e}")
            self.logger.debug(f"Traceback:\n{traceback.format_exc()}")
            raise
    
    def initialize_model(self):
        """Initialize the Base model"""
        self.logger.info("🚀 Initializing Base model...")
        
        try:
            self.model = EaModel.from_pretrained(
                base_model_path=config.base_model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=False,
                use_eagle3=False,
                server=self.server_communicator,
                device=config.device_base
            ).to(config.device_base)

            self.logger.info("Device distribution:")
            self.logger.info(f"1. base_model.lm_head: {self.model.base_model.lm_head.weight.device}")
            self.logger.info(f"2. input_embedding layer: {self.model.base_model.model.embed_tokens.weight.device}")
            
            self.model.eval()
            self.logger.info(f"✅ Base model initialized (Tag: {self.client_tag})")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize model: {e}")
            self.logger.debug(f"Traceback:\n{traceback.format_exc()}")
            raise
    
    def start_interactive_session(self):
        """Start interactive session"""
        ues_llama_2_chat = False
        use_vicuna = True
        
        self.logger.info(f"\n🎯 Base model ready for interaction (Tag: {self.client_tag})")
        self.logger.info("Enter 'quit' to exit, 'close server' to shutdown")
        self.logger.info("="*60)
        
        # Statistics
        average_accept_length = []
        average_generate_speed = []
        total_token_generated = []
        total_iterations = []  # Record number of iterations per generation
        
        while True:
            try:
                your_message = input("\nYour message: ")
                self.logger.debug(f"User input: {your_message}")
                
                if your_message.lower() == "quit":
                    self.logger.info("User requested quit")
                    break
                elif your_message.lower() == "close server":
                    # 发送关闭信号给服务器
                    self.logger.info("User requested server shutdown")
                    try:
                        self.server_communicator.send_vars("close server")
                        self.logger.debug("Shutdown signal sent to server")
                    except Exception as e:
                        self.logger.warning(f"Failed to send shutdown signal: {e}")
                    break

                # Prepare conversation template
                if ues_llama_2_chat:
                    conv = get_conversation_template("llama-2-chat")  
                    sys_p = "You are a helpful, respectful and honest assistant."
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
                input_ids = self.model.tokenizer([prompt]).input_ids
                input_ids = torch.as_tensor(input_ids).to(config.device_base)
                self.logger.debug(f"Input tokenized: {input_ids.shape}")
                
                # Generate reply
                self.logger.info(f"🤔 Processing... (Client: {self.client_tag})")
                output_ids, inference_time, original_input_ids = self.model.eagenerate(
                    input_ids=input_ids,
                    temperature=0,
                    max_new_tokens=2048
                )
                
                if inference_time == 0:
                    self.logger.warning("❌ Server closed connection (inference_time = 0)")
                    break
                    
                # Decode only the newly generated tokens
                generated_ids = output_ids[:, input_ids.shape[1]:]
                output = self.model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                num_new_tokens = output_ids.shape[1] - input_ids.shape[1]

                self.logger.info(f"\n🤖 Generated response:")
                self.logger.info(output)
                
                speed = num_new_tokens/inference_time
                accept_len = num_new_tokens/self.model.total_iteration
                current_iterations = self.model.total_iteration
                
                self.logger.info(f"\n📊 Stats - Time: {inference_time:.4f}s, Tokens: {num_new_tokens}, Speed: {speed:.2f} tok/s")
                self.logger.info(f"    Accept length: {accept_len:.2f}, Total iterations: {current_iterations}")
                
                # Update statistics
                average_generate_speed.append(round(speed, 4))
                average_accept_length.append(round(accept_len, 4))
                total_token_generated.append(num_new_tokens)
                total_iterations.append(current_iterations)
                
            except KeyboardInterrupt:
                self.logger.warning("\n🔴 Interrupted by user (KeyboardInterrupt)")
                break
            except ConnectionError as e:
                self.logger.error(f"❌ Connection error during generation: {e}")
                self.logger.debug(f"Traceback:\n{traceback.format_exc()}")
                break
            except socket.error as e:
                self.logger.error(f"❌ Socket error during generation: {e}")
                self.logger.debug(f"Traceback:\n{traceback.format_exc()}")
                break
            except RuntimeError as e:
                self.logger.error(f"❌ Runtime error during generation: {e}")
                self.logger.debug(f"Traceback:\n{traceback.format_exc()}")
                break
            except Exception as e:
                self.logger.error(f"❌ Unexpected error during generation: {e}")
                self.logger.error(f"Error type: {type(e).__name__}")
                self.logger.debug(f"Traceback:\n{traceback.format_exc()}")
                # 继续运行，不要直接break
                self.logger.info("Attempting to continue...")
        
        # Print final statistics
        self.logger.info("="*60)
        if average_generate_speed:
            self.logger.info(f"\n📈 Final Statistics:")
            self.logger.info(f"   Average speed: {sum(average_generate_speed)/len(average_generate_speed):.2f} tok/s")
            self.logger.info(f"   Average accept length: {sum(average_accept_length)/len(average_accept_length):.2f}")
            self.logger.info(f"   Total tokens generated: {sum(total_token_generated)}")
            self.logger.info(f"   Total iterations: {sum(total_iterations)}")
            self.logger.info(f"   Average iterations per generation: {sum(total_iterations)/len(total_iterations):.2f}")
        else:
            self.logger.warning("No tokens were generated during this session")
    
    def disconnect(self):
        """Disconnect the client"""
        self.logger.info(f"🔌 Disconnecting client {self.client_tag}")
        
        if self.server_communicator:
            try:
                self.server_communicator.close()
                self.logger.info("Server communicator closed successfully")
            except Exception as e:
                self.logger.warning(f"Error closing server communicator: {e}")
        
        self.logger.info("="*60)
        self.logger.info("Client session ended")

def main():
    """Main function"""
    client = BaseModelClient()
    
    try:
        client.logger.info("🚀 Starting Base Model Client")
        client.logger.info(f"Configuration: {config.base_model_path}")
        
        client.connect_to_draft_server()
        
        client.initialize_model()
        
        client.start_interactive_session()
        
    except KeyboardInterrupt:
        client.logger.warning("\n🔴 Client interrupted by user (KeyboardInterrupt)")
    except ConnectionRefusedError as e:
        client.logger.error(f"❌ Connection refused: {e}")
        client.logger.error("Make sure the Draft server is running")
        client.logger.debug(f"Traceback:\n{traceback.format_exc()}")
    except Exception as e:
        client.logger.error(f"❌ Client error: {e}")
        client.logger.error(f"Error type: {type(e).__name__}")
        client.logger.error(f"Full traceback:\n{traceback.format_exc()}")
    finally:
        client.disconnect()

if __name__ == "__main__":
    main()