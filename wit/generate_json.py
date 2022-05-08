import torch
import torchvision.models as models
from profiler import profile, ProfilerActivity
from torch.autograd import kineto_available, _supported_activities, DeviceType
from torch.autograd.profiler import record_function

def get_params(model, recurse=False):
    """Returns dictionary of paramters

    Arguments:
        model {torch.nn.Module} -- Network to extract the parameters from

    Keyword Arguments:
        recurse {bool} -- Whether to recurse through children modules

    Returns:
        Dict(str:numpy.ndarray) -- Dictionary of named parameters their
                                   associated parameter arrays
    """
    params = {k: v.detach().gpu().numpy().copy()
              for k, v in model.named_parameters(recurse=recurse)}
    return params



model = models.inception_v3().cuda()
for k, v in model.state_dict().items():
    print(k)
    print(type(v))
inputs = torch.randn(5, 3, 299, 299).cuda()
print(torch.autograd._supported_activities())
with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)

prof.export_chrome_trace("trace_resnet18.json")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
