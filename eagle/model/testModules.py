import sys
import socket
import pickle
import time
import torch

def get_tensor_size(tensor):
    if tensor is None:
        return 0
    elif isinstance(tensor, torch.Tensor):
        return tensor.numel() * tensor.element_size()  # PyTorch tensor
    elif isinstance(tensor, (int, float, bool)):
        return sys.getsizeof(tensor)  # Python scalar types
    elif isinstance(tensor, (list, tuple, dict)):
        return sum(get_tensor_size(x) for x in tensor)  # Recursively compute container sizes
    else:
        raise TypeError(f"Unsupported type: {type(tensor)}")

def get_total_transfer(inputs_dict):
    total = 0
    for name, tensor in inputs_dict.items():
        try:
            size = get_tensor_size(tensor)
            print(f"{name}: {size / 1024:.2f} KB")
            total += size
        except TypeError as e:
            print(f"Warning: {name} has unsupported type ({type(tensor)}), skipped.")
    return total

def get_tensor_size_bytes(tensor):
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return 0
    return tensor.numel() * tensor.element_size()


def send_tensor_via_ip(tensor, server_ip='0.0.0.0', port=12345):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((server_ip, port))
        s.listen(1)
        print(f"Server started, listening on {server_ip}:{port}...")

        conn, client_addr = s.accept()
        with conn:
            print(f"Client connected: {client_addr}")

            data = pickle.dumps(tensor)
            data_size = len(data)

            send_time = time.time()
            print("sending")
            conn.sendall(data_size.to_bytes(8, byteorder='big'))

            chunk_size = 1024 * 1024
            for i in range(0, data_size, chunk_size):
                conn.sendall(data[i:i + chunk_size])

            print(f"Sent {data_size / (1024**2):.2f} MB of data")
            return send_time, data_size


def get_tgt_size_in_mb(
    tgt: torch.Tensor,      
    verbose: bool = True 
) -> float:
    # 1. Compute total number of elements
    num_elements = tgt.numel()
    
    # 2. Compute bytes per element based on dtype
    if tgt.dtype == torch.float32:
        bytes_per_element = 4
    elif tgt.dtype == torch.float16 or tgt.dtype == torch.bfloat16:
        bytes_per_element = 2
    else:
        raise ValueError(f"Unsupported dtype: {tgt.dtype}")
    
    # 3. Compute total memory (MB)
    size_in_mb = (num_elements * bytes_per_element) / (1024 ** 2)
    
    if verbose:
        print(
            f"tgt_shape = {tuple(tgt.shape)}\n"
            f"dtype = {tgt.dtype} ({bytes_per_element} bytes/element)\n"
            f"size_in_mb = {size_in_mb:.2f} MB"
        )
    
    return size_in_mb


def send_tensor_via_ip(tensor, server_ip='0.0.0.0', port=12345):
    """Send a tensor via IP and return the send timestamp and data size."""
    # Create socket (TCP)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((server_ip, port))
        s.listen(1)
        print(f"Server started, listening on {server_ip}:{port}...")

        # Wait for client connection
        conn, client_addr = s.accept()
        with conn:
            print(f"Client connected: {client_addr}")

            # 1. Serialize the tensor
            data = pickle.dumps(tensor)
            data_size = len(data)

            # 2. Record timestamp before sending
            send_time = time.time()
            print("sending")
            # 3. Send data length (8-byte header)
            conn.sendall(data_size.to_bytes(8, byteorder='big'))

            # 4. Send actual data in chunks (1MB per chunk)
            chunk_size = 1024 * 1024
            for i in range(0, data_size, chunk_size):
                conn.sendall(data[i:i + chunk_size])

            print(f"Sent {data_size / (1024**2):.2f} MB of data")
            return send_time, data_size
