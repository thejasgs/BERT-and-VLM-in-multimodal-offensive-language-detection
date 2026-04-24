import tensorflow as tf
import torch

# GPU Warm-up Function
def gpu_warmup(device='cuda', iterations=50):
    """
    Warm up GPU by performing matrix multiplications
    """
    print("Warming up GPU with PyTorch and Tensorflow")
    if device == 'cuda' and torch.cuda.is_available():
        # PyTorch warmup
        for i in range(iterations):
            a = torch.randn(1000, 1000, device=device)
            b = torch.randn(1000, 1000, device=device)
            c = torch.matmul(a, b)
            del a, b, c
        torch.cuda.synchronize()
        print("PyTorch GPU warmup complete")
    
    # TensorFlow warmup
    if gpus:
        for i in range(iterations):
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)
            del a, b, c
        print("TensorFlow GPU warmup complete")
    
    print("GPU warmup PyTorch and Tensorflow finished!\n")