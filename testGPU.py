import torch

try:
    deviceName = torch.cuda.get_device_name(torch.cuda.current_device())
    print(f'GPU Name: {deviceName}')
except:
    pass

# Select device
if torch.cuda.is_available():
    device = torch.device('cuda') # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = torch.device('mps') # Apple GPU (Metal Performance Shaders)
else:
    device = torch.device('cpu')
print(f'CUDA Available: {torch.cuda.is_available()}\n'
      f'CUDA Version: {torch.version.cuda}\n'
      f'Training Device: {device}')
