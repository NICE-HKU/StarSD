import socket
import struct
import torch
import io
from typing import Any, Tuple, List, Union
import time


class FastDistributedCommunicator:
    def __init__(self, 
                 host: str = 'localhost', 
                 port: int = 12345, 
                 is_server: bool = False,
                 buffer_size: int = 4096,
                 max_retries: int = 30, 
                 retry_delay: float = 2.0): 
        self.host = host
        self.port = port
        self.is_server = is_server
        self.buffer_size = buffer_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.sock = None
        self.conn = None
        self.total_data_send = 0
        self.total_data_receive = 0
        self._setup_connection()

    def _setup_connection(self):
        """reconnect with retries"""
        for attempt in range(self.max_retries):
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # allow reuse address
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                # add buffer size to improve throughput
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024*1024)  # 1MB
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024*1024)  # 1MB
                
                if self.is_server:
                    self.sock.bind((self.host, self.port))
                    self.sock.listen(1)
                    print(f"Server listening on {self.host}:{self.port}")
                    self.conn, _ = self.sock.accept()
                else:
                    self.sock.connect((self.host, self.port))
                    self.conn = self.sock
                break  
            except (ConnectionRefusedError, OSError) as e:
                if self.is_server and "Address already in use" in str(e):
                    print(f"Port {self.port} already in use, attempt {attempt + 1}/{self.max_retries}")
                elif not self.is_server:
                    print(f"Waiting for server to start, attempt {attempt + 1}/{self.max_retries}")
                
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Failed to establish connection after {self.max_retries} attempts") from e
                time.sleep(self.retry_delay)

    def get_tcp_rtt(self) -> float:
        """
        Get the TCP connection RTT (Round Trip Time) - kernel-level measurement, unaffected by clock synchronization
        
        Returns:
            RTT time (milliseconds), returns 0.0 on failure
        """
        try:
            if not self.conn:
                return 0.0
            
            # Linux TCP_INFO structure
            TCP_INFO = 11
            TCP_INFO_SIZE = 232
            
            tcp_info = self.conn.getsockopt(socket.IPPROTO_TCP, TCP_INFO, TCP_INFO_SIZE)
            # RTT at offset 64, 32-bit unsigned int, in microseconds
            rtt_us = struct.unpack_from('I', tcp_info, 64)[0]
            
            return rtt_us / 1000.0  # convert to milliseconds
        except Exception as e:
            # Non-Linux system or other error, return 0
            return 0.0
    
    def _prepare_data_for_serialization(self, vars_to_send: Tuple[Any, ...]) -> Tuple[Any, ...]:
        """
        Prepare data for serialization
        Key optimization: move GPU tensors to CPU to avoid CUDA serialization overhead
        """
        prepared = []
        for var in vars_to_send:
            if isinstance(var, torch.Tensor):
                # Move to CPU (if on GPU)
                if var.is_cuda:
                    var = var.cpu()
            prepared.append(var)
        return tuple(prepared)

    def send_vars(self, *vars_to_send: Any, include_timestamp: bool = False) -> float:
        """
        Send variables - use torch.save instead of pickle
        
        Args:
            *vars_to_send: variables to send
            include_timestamp: whether to add a timestamp at the end of the tuple
        
        Returns:
            Size of the sent packet (KB)
        """
        try:
            # Prepare data (move GPU tensors to CPU)
            prepared_data = self._prepare_data_for_serialization(vars_to_send)
            
            # If timestamp is needed, add it at the end of the tuple
            if include_timestamp:
                send_timestamp = time.time()
                data_to_send = prepared_data + (send_timestamp,)
            else:
                data_to_send = prepared_data
            
            # Serialize using torch.save
            buffer = io.BytesIO()
            torch.save(data_to_send, buffer)
            serialized_data = buffer.getvalue()
            
            self.conn.sendall(struct.pack('!Q', len(serialized_data)))
            self.conn.sendall(serialized_data)
            packet_size = (len(serialized_data) + 8) / 1024  # KB
            
            # Get TCP RTT (kernel-level measurement)
            rtt_ms = self.get_tcp_rtt()
            
            print("{} KB data sent | RTT: {:.2f}ms".format(round(packet_size, 3), rtt_ms))
            self.total_data_send = self.total_data_send + packet_size
            return packet_size
        except (ConnectionError, BrokenPipeError, OSError) as e:
            print(f"❌ Send error: {e}")
            # Check if it's a Bad file descriptor error
            if "Bad file descriptor" in str(e) or isinstance(e, OSError):
                print("🔄 Attempting to reconnect due to bad file descriptor...")
                try:
                    self._reconnect()
                    # Retry sending
                    prepared_data = self._prepare_data_for_serialization(vars_to_send)
                    
                    if include_timestamp:
                        data_to_send = prepared_data + (time.time(),)
                    else:
                        data_to_send = prepared_data
                    
                    buffer = io.BytesIO()
                    torch.save(data_to_send, buffer)
                    serialized_data = buffer.getvalue()
                    self.conn.sendall(struct.pack('!Q', len(serialized_data)))
                    self.conn.sendall(serialized_data)
                    print("✅ Retry send successful")
                    packet_size = (len(serialized_data) + 8) / 1024
                    self.total_data_send = self.total_data_send + packet_size
                    return packet_size
                except Exception as retry_e:
                    print(f"❌ Retry failed: {retry_e}")
                    raise RuntimeError("Connection lost and retry failed") from retry_e
            else:
                raise RuntimeError("Connection lost during send") from e


    def recv_vars(self, return_packet_size: bool = False) -> Union[Tuple[Any, ...], Tuple[Tuple[Any, ...], float]]:
        """
        Receive data - use torch.load instead of pickle
        
        Args:
            return_packet_size: whether to return packet_size
        
        Returns:
            If return_packet_size=False: received data tuple
            If return_packet_size=True: (data tuple, packet_size_KB)
        """
        for attempt in range(self.max_retries):
            try:
                data_len_bytes = self._recv_all(8)
                data_len = struct.unpack('!Q', data_len_bytes)[0]
                serialized_data = self._recv_all(data_len)
                packet_size = (len(serialized_data) + 8) / 1024  # KB
                self.total_data_receive = self.total_data_receive + packet_size
                print("{} KB data received".format(round(packet_size, 3)))
                
                # Deserialize using torch.load
                buffer = io.BytesIO(serialized_data)
                data = torch.load(buffer, weights_only=False)
                
                # Get TCP RTT
                rtt_ms = self.get_tcp_rtt()
                
                if return_packet_size:
                    # Return: (data, packet_size, rtt_ms)
                    return data, packet_size, rtt_ms
                return data  # Return data on success
            except (ConnectionError, struct.error, OSError) as e:
                print(f"❌ Receive error (attempt {attempt + 1}): {e}")
                
                # If it's a Bad file descriptor error, attempt to reconnect
                if "Bad file descriptor" in str(e) or isinstance(e, OSError):
                    print("🔄 Bad file descriptor detected, attempting reconnect...")
                    try:
                        self._reconnect()
                        continue  # Retry receiving
                    except Exception as reconnect_e:
                        print(f"❌ Reconnect failed: {reconnect_e}")
                
                if attempt == self.max_retries - 1:
                    raise RuntimeError("Max retries exceeded") from e
                time.sleep(self.retry_delay)
        
        raise RuntimeError("Unexpected state")  # This should theoretically never be reached


    def _reconnect(self):
        """Completely rebuild the connection (not just the socket)"""
        print("🔄 Starting reconnection process...")
        
        # Safely close old connection
        try:
            if self.conn:
                self.conn.close()
        except:
            pass
        try:
            if self.sock:
                self.sock.close()
        except:
            pass
        
        # Wait a moment before reconnecting
        time.sleep(self.retry_delay)
        
        # Re-establish connection with retry mechanism
        for attempt in range(self.max_retries):
            try:
                # Completely reinitialize connection
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                # Increase buffer size to improve throughput
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024*1024)  # 1MB
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024*1024)  # 1MB
                
                if self.is_server:
                    self.sock.bind((self.host, self.port))
                    self.sock.listen(1)
                    print(f"🔄 Server relistening {self.host}:{self.port}")
                    self.conn, _ = self.sock.accept()  # Block until new client connects
                    print("✅ Connect to new client")
                    print("Waiting for message...")
                else:
                    print(f"🔄 Reconnecting to {self.host}:{self.port} (attempt {attempt + 1})")
                    self.sock.connect((self.host, self.port))
                    self.conn = self.sock
                    print("✅ Reconnection successful")
                
                return  # Exit on success
                
            except Exception as e:
                print(f"❌ Reconnect attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Failed to reconnect after {self.max_retries} attempts") from e
                time.sleep(self.retry_delay * (attempt + 1))  # Incremental delay

    def _recv_all(self, size: int) -> bytes:
        """Receive data with timeout, handling Bad file descriptor errors"""
        data = bytearray()
        timeout = 500.0  # 500 seconds timeout
        start_time = time.time()
        
        while len(data) < size:
            try:
                # Check if connection is valid
                if not self.conn:
                    raise ConnectionError("No active connection")
                
                self.conn.settimeout(1.0)  # Each recv() waits at most 1 second
                chunk = self.conn.recv(min(size - len(data), self.buffer_size))
                if not chunk:  # Connection closed by peer
                    raise ConnectionError("Connection closed by peer")
                data.extend(chunk)
            except socket.timeout:
                if time.time() - start_time > timeout:
                    raise ConnectionError("Receive timeout")
                continue
            except OSError as e:
                # Special handling for Bad file descriptor errors
                if "Bad file descriptor" in str(e):
                    raise ConnectionError(f"Bad file descriptor: {e}")
                else:
                    raise ConnectionError(f"Socket error: {e}")
            except Exception as e:
                raise ConnectionError(f"Unexpected error during receive: {e}")
            finally:
                try:
                    self.conn.settimeout(None)
                except:
                    pass
        
        return bytes(data)


    def close(self) -> None:
        """close the connection"""
        try:
            if self.conn:
                self.conn.close()
        except:
            pass
        try:
            if self.sock:
                self.sock.close()
        except:
            pass


if __name__ == "__main__":

    client = FastDistributedCommunicator(port=12345)
    

    tensor_data = torch.randn(3, 256)
    client.send_vars(tensor_data, 42, "speculative")

    result_flag, new_hidden = client.recv_vars()
    print(f"Received flag: {result_flag}, hidden shape: {new_hidden.shape}")
    
    client.close()