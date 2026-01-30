#!/usr/bin/env python3
"""
Automated Base Client - supports random sampling or sequential traversal

Features:
1. Connect to the Draft server
2. Support two question selection modes from question.jsonl:
     - random: random sampling (duplicates allowed)
     - sequential: sequential traversal (no duplicates)
3. Run inference automatically and display results
4. Collect performance statistics
5. Support selecting GPU device via the --cuda command-line argument

Usage examples:
    # Random sample 500 questions on cuda:0
    python auto_base_client.py --cuda 0 --questions 500

    # Sequentially process first 100 questions on cuda:1
    python auto_base_client.py --cuda 1 --questions 100 --mode sequential

    # Sequentially process entire dataset on cuda:2
    python auto_base_client.py --cuda 2 --mode sequential --all

Example run:
python auto_base_client.py --cuda 0 --questions 10
"""

import sys
import os
# Add StarSD directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eagle.model.base_model_inference import EaModel
import torch
from fastchat.model import get_conversation_template
import json
import socket
import time
import random
import argparse
import os
import traceback
from eagle.model.config_loader import config
from framework.simple_auto_logger import SimpleAutoLogger


class AutoBaseClient:
    """Automated Base model client"""
    
    def __init__(self, gpu_device=0, client_id=0):
        self.model = None
        self.client_tag = None
        self.private_port = None
        self.server_communicator = None
        self.results = []
        # Configure GPU device - accept integer or full device string
        if isinstance(gpu_device, int):
            self.device = f"cuda:{gpu_device}"
        elif isinstance(gpu_device, str) and gpu_device.startswith("cuda:"):
            self.device = gpu_device
        else:
            self.device = f"cuda:{gpu_device}"
        
        # Initialize logger
        self.error_logger = SimpleAutoLogger(client_id)
        self.error_logger.log_info(f"📍 GPU device: {self.device}")
        self.error_logger.log_info(f"📍 Process ID: {os.getpid()}")
        print(f"📍 Will use GPU device: {self.device}")
        
    def connect_to_draft_server(self, server_host='127.0.0.1', server_port=None):
        """Connect to the Draft server and obtain allocation information"""
        server_port = server_port or config.communication_port
        
        print(f"🔗 Connecting to Draft server at {server_host}:{server_port}")
        print(f"🔍 Client info - PID: {os.getpid()}, GPU device: {self.device}")
        
        try:
            # Connect to the public port
            handshake_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            handshake_socket.settimeout(30)
            handshake_socket.connect((server_host, server_port))
            
            # Receive allocation information
            length_bytes = handshake_socket.recv(4)
            if len(length_bytes) < 4:
                raise Exception("Failed to receive response length")
                
            response_length = int.from_bytes(length_bytes, byteorder='big')
            response_data = handshake_socket.recv(response_length)
            
            handshake_socket.close()
            
            # Parse the response
            response = json.loads(response_data.decode('utf-8'))
            
            if response["status"] != "success":
                raise Exception(f"Server rejected connection: {response}")
            
            self.client_tag = response["client_tag"]
            self.private_port = response["private_port"]
            
            self.error_logger.log_info(f"✅ Connected - Tag: {self.client_tag}, Port: {self.private_port}")
            print(f"✅ Connected to Draft server")
            print(f"   Client Tag: {self.client_tag}")
            print(f"   Private Port: {self.private_port}")
            
            # Connect to the private port
            self._connect_private_channel(server_host)
            
        except Exception as e:
            error_msg = f"Failed to connect to Draft server: {e}"
            print(f"❌ {error_msg}")
            self.error_logger.log_critical(error_msg, e)
            raise
    
    def _connect_private_channel(self, server_host):
        """Connect to the private communication channel"""
        from eagle.model.communicator import FastDistributedCommunicator
        
        print(f"🔒 Connecting to private channel on port {self.private_port}")
        
        # Wait for server to start the private port
        time.sleep(2)
        
        self.server_communicator = FastDistributedCommunicator(
            host=server_host,
            port=self.private_port,
            is_server=False,
            max_retries=20
        )
        
        print(f"✅ Connected to private channel")
    
    def initialize_model(self):
        """Initialize the Base model"""
        print(f"🚀 Initializing Base model on device: {self.device}...")
        
        # Validate GPU availability
        if self.device.startswith('cuda:'):
            gpu_id = int(self.device.split(':')[1])
            available_gpus = torch.cuda.device_count()
            if gpu_id >= available_gpus:
                error_msg = f"GPU {gpu_id} not available. Available GPUs: {list(range(available_gpus))}"
                self.error_logger.log_critical(error_msg)
                raise RuntimeError(error_msg)
            print(f"📍 GPU validation passed: {self.device} (Available: {available_gpus} GPUs)")
        
        self.model = EaModel.from_pretrained(
            base_model_path=config.base_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=False,
            use_eagle3=False,
            server=self.server_communicator,
            device=self.device
        ).to(self.device)

        print("Device distribution:")
        print(f"1. base_model.lm_head: {self.model.base_model.lm_head.weight.device}")
        print(f"2. input_embedding layer: {self.model.base_model.model.embed_tokens.weight.device}")
        
        self.model.eval()
        print(f"✅ Base model initialized (Tag: {self.client_tag}, Device: {self.device})")
    
    def load_questions(self, questions_file='question.jsonl'):
        """Load questions from question.jsonl"""
        questions = []
        
        try:
            with open(questions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    if 'turns' in data and len(data['turns']) > 0:
                        # Use the first turn as the question
                        questions.append({
                            'id': data.get('question_id', 'unknown'),
                            'category': data.get('category', 'unknown'),
                            'question': data['turns'][0]
                        })
            
            print(f"📚 Loaded {len(questions)} questions from {questions_file}")
            return questions
            
        except FileNotFoundError:
            error_msg = f"Question file {questions_file} not found"
            print(f"❌ {error_msg}")
            self.error_logger.log_error(error_msg)
            return []
        except Exception as e:
            error_msg = f"Error loading questions: {e}"
            print(f"❌ {error_msg}")
            self.error_logger.log_error(error_msg, e)
            return []
    
    def run_auto_inference(self, num_questions, mode='random', question_ids=None, wait_min=0.0, wait_max=0.0):
        """
        Run automatic inference

        Args:
            num_questions: Number of questions to process
            mode: 'random' - random sampling; 'sequential' - sequential traversal; 'manual' - specify question IDs
            question_ids: List of question IDs (used only when mode='manual')
            wait_min: Minimum wait time between inferences (seconds)
            wait_max: Maximum wait time between inferences (seconds)
        """
        print(f"\n🎯 Starting automatic inference - Mode: {mode.upper()}")
        
        # Load all questions
        all_questions = self.load_questions()
        
        if not all_questions:
            print("❌ No questions available")
            return
        
        # select questions based on mode
        if mode == 'manual':
            # Manual mode: select questions by provided IDs
            if not question_ids:
                print("❌ Manual mode requires question_ids to be specified")
                return
            
            # Create a mapping from question_id to question
            question_map = {q['id']: q for q in all_questions}
            
            selected_questions = []
            not_found_ids = []
            
            for qid in question_ids:
                if qid in question_map:
                    selected_questions.append(question_map[qid])
                else:
                    not_found_ids.append(qid)
            
            print(f"🎯 Manual mode: Selected {len(selected_questions)} questions by ID")
            if not_found_ids:
                print(f"⚠️  Warning: {len(not_found_ids)} question IDs not found: {not_found_ids}")
            
            if not selected_questions:
                print("❌ No valid questions found for the specified IDs")
                return
                
        elif mode == 'sequential':
            # Sequential mode: process entire dataset or a specified number
            selected_questions = all_questions[:num_questions] if num_questions > 0 else all_questions
            print(f"📖 Sequential mode: Processing {len(selected_questions)} questions (Total in dataset: {len(all_questions)})")
        else:
            # Random sampling mode (duplicates allowed)
            selected_questions = random.choices(all_questions, k=num_questions)
            print(f"🎲 Random sampling mode: Selected {len(selected_questions)} questions")
        
        print("=" * 60)
        
        total_time = 0
        total_tokens = 0
        successful_inferences = 0
        
        # Process each question
        total_questions = len(selected_questions)
        for i, question_data in enumerate(selected_questions):
            try:
                print(f"\n📝 Question {i+1}/{total_questions}")
                print(f"ID: {question_data['id']}, Category: {question_data['category']}")
                print(f"Question: {question_data['question'][:100]}...")
                
                result = self._process_question(question_data['question'])
                
                if result:
                    self.results.append({
                        'question_id': question_data['id'],
                        'category': question_data['category'],
                        'question': question_data['question'],
                        **result
                    })
                    
                    total_time += result['inference_time']
                    total_tokens += result['tokens_generated']
                    successful_inferences += 1
                    
                    print(f"✅ Completed - Time: {result['inference_time']:.2f}s, "
                          f"Tokens: {result['tokens_generated']}, "
                          f"Speed: {result['speed']:.2f} tok/s")
                    
                    # Random wait (if configured)
                    if wait_max > 0:
                        wait_time = random.uniform(wait_min, wait_max)
                        print(f"⏳ Waiting for {wait_time:.2f}s before next inference...")
                        time.sleep(wait_time)
                else:
                    error_msg = f"Failed to process question {i+1} (ID: {question_data['id']})"
                    print(f"❌ {error_msg}")
                    self.error_logger.log_error(error_msg)
                
                
            except Exception as e:
                error_msg = f"Error processing question {i+1} (ID: {question_data['id']}): {e}"
                print(f"❌ {error_msg}")
                self.error_logger.log_error(error_msg, e)
                continue
        
    
    def _process_question(self, question):
        """Process a single question"""
        try:
            # Use Vicuna conversation template
            conv = get_conversation_template("vicuna")
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            # Tokenize
            input_ids = self.model.tokenizer([prompt]).input_ids
            input_ids = torch.as_tensor(input_ids).to(self.device)
            
            # Generate reply
            start_time = time.time()
            output_ids, inference_time, original_input_ids = self.model.eagenerate(
                input_ids=input_ids,
                temperature=0.5,
                max_new_tokens=1024
            )
            
            if inference_time == 0:
                error_msg = "Server closed connection unexpectedly during inference"
                print(f"❌ {error_msg}")
                self.error_logger.log_error(error_msg)
                return None
            
            # Decode output
            generated_ids = output_ids[:, input_ids.shape[1]:]
            output = self.model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"Generated output: {output}")
            num_new_tokens = output_ids.shape[1] - input_ids.shape[1]
            
            # Compute metrics
            speed = num_new_tokens / inference_time if inference_time > 0 else 0
            accept_length = num_new_tokens / self.model.total_iteration if self.model.total_iteration > 0 else 0
            
            result = {
                'inference_time': inference_time,
                'tokens_generated': num_new_tokens,
                'speed': speed,
                'accept_length': accept_length,
                'total_iterations': self.model.total_iteration,
                'output': output,
                'timestamp': start_time
            }
            
            return result
            
        except Exception as e:
            error_msg = f"Error in question processing: {e}"
            print(f"❌ {error_msg}")
            self.error_logger.log_error(error_msg, e)
            return None
    
    def _show_final_statistics(self, successful_inferences, total_questions, total_time, total_tokens):
        """Display final statistics"""
        print("\n" + "=" * 60)
        print("📊 Final Statistics")
        print("=" * 60)
        
        print(f"📝 Total questions: {total_questions}")
        print(f"✅ Successful inferences: {successful_inferences}")
        print(f"❌ Failed inferences: {total_questions - successful_inferences}")
        
        if successful_inferences > 0:
            avg_time = total_time / successful_inferences
            avg_speed = total_tokens / total_time if total_time > 0 else 0
            
            print(f"⏱️  Total time: {total_time:.2f}s")
            print(f"🔢 Total tokens generated: {total_tokens}")
            print(f"📈 Average time per question: {avg_time:.2f}s")
            print(f"⚡ Average speed: {avg_speed:.2f} tok/s")
            
            # Compute accept length statistics
            if self.results:
                accept_lengths = [r['accept_length'] for r in self.results]
                avg_accept_length = sum(accept_lengths) / len(accept_lengths)
                print(f"🎯 Average accept length: {avg_accept_length:.2f}")
        else:
            print("❌ No successful inferences")
        
        # Save results to file
        if self.results:
            timestamp = int(time.time())
            results_file = f"auto_client_results_{timestamp}.json"
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Results saved to {results_file}")
    
    def disconnect(self):
        """Disconnect the client"""
        print(f"🔌 Disconnecting client {self.client_tag}")
        
        if self.server_communicator:
            try:
                # Send shutdown signal
                self.server_communicator.send_vars("close server")
                time.sleep(1)
                self.server_communicator.close()
                print("✅ Successfully disconnected")
            except Exception as e:
                error_msg = f"Error during disconnection: {e}"
                print(f"⚠️ {error_msg}")
                self.error_logger.log_warning(error_msg)
        
        self.error_logger.close()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Automated Base Client - supports random sampling, sequential traversal, or manual question ID selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  # Random sample 500 questions on cuda:0
  python auto_base_client.py --cuda 0 --questions 500 --mode random

  # Sequentially traverse the entire dataset on cuda:1
  python auto_base_client.py --cuda 1 --mode sequential --all

  # Sequentially process first 100 questions on cuda:2
  python auto_base_client.py --cuda 2 --questions 100 --mode sequential

  # Manually specify question IDs
  python auto_base_client.py --cuda 0 --mode manual --question_ids 81 82 83 84 85
        """
    )

    parser.add_argument('--cuda', type=int, required=True,
                       help='Specify CUDA device index (e.g.: 0, 1, 2, 3)')

    parser.add_argument('--client_id', type=int, default=None,
                       help='Client ID for log file naming (default: use the cuda index)')

    parser.add_argument('--questions', type=int, default=-1,
                       help='Number of questions to process (default: 2; in sequential mode use -1 or --all for all; ignored in manual mode)')

    parser.add_argument('--mode', type=str, default='sequential', choices=['random', 'sequential', 'manual'],
                       help='Inference mode: random=random sampling (duplicates allowed), sequential=sequential traversal, manual=manual IDs (default: random)')

    parser.add_argument('--question_ids', type=int, nargs='+', default=[81],
                       help='List of question IDs for manual mode, e.g.: --question_ids 81 82 83')

    parser.add_argument('--all', action='store_true',
                       help='Process the entire dataset in sequential mode (equivalent to --questions -1)')

    parser.add_argument('--questions_file', type=str, default='question.jsonl',
                       help='Questions file path (default: question.jsonl)')

    parser.add_argument('--server_host', type=str, default='127.0.0.1',
                       help='Draft server host (default: 127.0.0.1)')

    parser.add_argument('--server_port', type=int, default=None,
                       help='Draft server port (default: use port from config)')

    parser.add_argument('--wait_min', type=float, default=0.0,
                       help='Minimum wait time after each inference (seconds), default: 0.0')

    parser.add_argument('--wait_max', type=float, default=0.0,
                       help='Maximum wait time after each inference (seconds), default: 0.0')

    args = parser.parse_args()
    
    if args.all:
        args.questions = -1
    
    # Validate arguments
    if args.mode == 'random' and args.questions <= 0:
        print("❌ In random mode, questions must be > 0")
        return

    if args.mode == 'manual' and not args.question_ids:
        print("❌ In manual mode, you must specify --question_ids")
        print("   e.g.: python auto_base_client.py --cuda 0 --mode manual --question_ids 81 82 83")
        return
    
    client_id = args.client_id if args.client_id is not None else args.cuda
    
    # Display mode information
    mode_desc = {
        'random': 'Random sampling (duplicates allowed)',
        'sequential': 'Sequential traversal (no duplicates)',
        'manual': 'Manual specified question IDs'
    }
    
    print("🎯 Auto Base Client - Question Inference")
    print("=" * 60)
    print(f"🖥️  CUDA device: cuda:{args.cuda}")
    print(f"📊 Inference mode: {args.mode} ({mode_desc[args.mode]})")
    
    if args.mode == 'manual':
        print(f"📝 Specified Question IDs: {args.question_ids}")
    else:
        print(f"📝 Number of questions: {args.questions if args.questions > 0 else 'all'}")
    
    print(f"📂 Questions file: {args.questions_file}")
    print(f"🌐 Server: {args.server_host}:{args.server_port or 'default'}")
    print()

    client = AutoBaseClient(gpu_device=args.cuda, client_id=client_id)

    try:

        client.connect_to_draft_server(args.server_host, args.server_port)
        
        client.initialize_model()

        client.run_auto_inference(args.questions, mode=args.mode, question_ids=args.question_ids,
                                 wait_min=args.wait_min, wait_max=args.wait_max)
        
    except KeyboardInterrupt:
        print("\n🔴 Interrupted by user")
        if hasattr(client, 'error_logger'):
            client.error_logger.log_warning("Client interrupted by user (Ctrl+C)")
    except Exception as e:
        error_msg = f"Client error: {e}"
        print(f"❌ {error_msg}")
        if hasattr(client, 'error_logger'):
            client.error_logger.log_critical(error_msg, e)
        import traceback
        traceback.print_exc()
    finally:
        pass


if __name__ == "__main__":
    main()
