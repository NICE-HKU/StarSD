<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="images/logo.png">
    <img alt="StarSD" src="images/logo.png" width="55%">
  </picture>
</p>


<h3 align="center">
One-for-Many Speculative Decoding
</h3>


# About
StarSD is a distributed speculative decoding system, achieving efficient large language model inference acceleration through a Draft Server and Base Client architecture.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="images/buffer_pipeline4.png">
    <img alt="StarSD" src="images/buffer_pipeline4.png" width="80%">
  </picture>
</p>

## System Architecture
- **Draft Server**: Runs a lightweight Draft model (EAGLE-Vicuna-7B) to generate candidate token sequences
- **Base Client**: Runs the complete Base model (Vicuna-7B) to verify Draft model outputs
- **Distributed Design**: Supports multiple Base Clients processing in parallel, fully utilizing multi-GPU resources

## 📦 Requirements

- Python 3.9
- PyTorch 2.7.1
- At least 2 GPUs (1 for Draft Server, 1+ for Base Clients)

### Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `torch==2.7.1` - PyTorch deep learning framework
- `transformers==4.53.3` - Hugging Face Transformers library
- `accelerate==1.8.1` - Model loading and distributed training utilities
- `fschat==0.2.36` - FastChat for conversation templates
- `numpy==2.0.2` - Numerical computing
- `scipy==1.13.1` - Scientific computing (for smoothing, etc.)
- `matplotlib==3.9.4` - Plotting and visualization
- `pyyaml==6.0.2` - YAML configuration file support

## ⚡ Getting Started

### Step 1: Download Model Weights

Download the following model weight files:

- **Draft Model**: [EAGLE-Vicuna-7B-v1.3](https://huggingface.co/yuhuili/EAGLE-Vicuna-7B-v1.3)
  
- **Base Model**: [Vicuna-7B-v1.3](https://huggingface.co/lmsys/vicuna-7b-v1.3)

Extract the downloaded model weights to a local directory.

### Step 2: Configure Model Paths

Open the `config.yaml` file and modify the following configurations:

```yaml
# Draft model configuration
draft_model_path: "/path/to/your/EAGLE-Vicuna-7B-v1.3"

# Base model configuration
base_model_path: "/path/to/your/vicuna-7b-v1.3"
```

Replace the paths with the actual locations where you extracted the model weights.

### Step 3: Prepare for Distributed Deployment

Run the weight preparation script to set up for distributed deployment:

```bash
python preparing_weight.py
```

This script processes the model weights and extracts necessary components for distributed inference.

### Step 4: Start the Draft Server

Launch the Draft Server in a terminal:

```bash
python draft_server.py
```

Wait a few seconds until you see the following message:

```
Draft Server is running!
```

The Draft Server is now successfully started and waiting for Base Client connections.

### Step 5: Start Base Clients

In a **new terminal window**, run the startup script:

```bash
./start_clients.sh
```

**Configuring GPU Devices**:

Before running, you can edit the `start_clients.sh` file to modify the `CUDA_DEVICES` parameter to specify which GPUs to use:

```bash
# Use a single GPU (cuda:3)
CUDA_DEVICES=(3)

# Use multiple GPUs (cuda:0, cuda:1, cuda:2)
CUDA_DEVICES=(0 1 2)
```

**Notes**:
- Each number in the `CUDA_DEVICES` array represents a GPU device ID
- The array length determines the number of Base Clients to launch
- Each Base Client loads a complete Base model on the specified GPU

### Step 6: Stopping Services

Stop all Base Clients:

```bash
./start_clients.sh stop
```

Stop the Draft Server: Press `Ctrl+C` in the Draft Server terminal



## 🔍 Configuration File

The main configuration file `config.yaml` contains the following key parameters:

- `draft_model_path`: Path to Draft model weights
- `base_model_path`: Path to Base model weights
- `server_ip`: IP address of Draft Server (default: localhost)
- `server_port`: Port number of Draft Server

## Acknowledgments
- [EAGLE](https://github.com/SafeAILab/EAGLE)
- [MT-Bench](https://huggingface.co/spaces/lmsys/mt-bench)
- [Vicuna](https://arxiv.org/pdf/2306.05685)