import torch
import torchvision.models as models
from profiler import profile, ProfilerActivity
from torch.autograd import kineto_available, _supported_activities, DeviceType
from torch.autograd.profiler import record_function


model = models.resnet18().cuda()
inputs = torch.randn(5, 3, 224, 224).cuda()
print(torch.autograd._supported_activities())
with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)

prof.export_chrome_trace("trace.json")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
