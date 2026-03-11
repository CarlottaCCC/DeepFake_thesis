import torch
import sys
print(sys.executable)

print("Env:", torch.__file__)
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
