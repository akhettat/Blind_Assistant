import torch

def check_gpu_availability():
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs available: {gpu_count}")
        for i in range(gpu_count):
            print(f"\nGPU {i} details:")
            print(f"Name: {torch.cuda.get_device_name(i)}")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 3:.2f} GB")
            print(f"Memory Cached: {torch.cuda.memory_reserved(i) / 1024 ** 3:.2f} GB")
            print(f"Capability: {torch.cuda.get_device_capability(i)}")
    else:
        print("No GPU available on this system.")

if __name__ == "__main__":
    check_gpu_availability()
