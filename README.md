# Production GenAI Inference Architecture

A reference implementation for serving Large Language Models (70B+ parameters) at scale using **Ray Serve**, **vLLM**, and **PyTorch**.

## 🚀 Architecture Overview
This repository demonstrates a high-throughput inference pattern designed to minimize **Time-To-First-Token (TTFT)** and maximize GPU saturation for models like **Llama-3-70B** and **Qwen-72B-Chat**.

### Key Components
* **Orchestration:** Ray Serve (for autoscaling and fault tolerance).
* **Inference Engine:** vLLM (leveraging PagedAttention and Continuous Batching).
* **Quantization:** AWQ (Activation-aware Weight Quantization) for efficient VRAM usage.
* **Distribution:** Tensor Parallelism across multi-GPU nodes.

## 🛠️ Performance Optimization Strategies
1.  **Continuous Batching:** Replaces static batching to allow request injection at iteration level, improving throughput by ~4x.
2.  **KV Cache Management:** Utilizing PagedAttention to reduce memory fragmentation.
3.  **Tensor Parallelism:** Sharding model weights across 4x/8x A100 GPUs using NCCL.
4.  **Preemption Handling:** Ray Actor replication to handle AWS Spot Instance termination gracefully.

## 📦 Tech Stack
* **Python 3.10+**
* **Ray 2.9+**
* **vLLM 0.4.0+**
* **PyTorch 2.2+**
