import torch
print(torch.cuda.is_available())  # 检查 CUDA 是否可用
print(torch.cuda.current_device())  # 当前使用的 GPU 设备编号
print(torch.cuda.get_device_name(0))  # 获取第一个 GPU 的名称
