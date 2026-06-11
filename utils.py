import os 
import torch 
import yaml
from yaml.loader import SafeLoader
import torch.distributed as dist 

def yaml_loader(yaml_file):
    with open(yaml_file,'r') as f:
        config_data = yaml.load(f,Loader=SafeLoader)
    
    return config_data

def progress(current, total, **kwargs):
    progress_percent = (current * 50 / total)
    progress_percent_int = int(progress_percent)
    data_ = ""
    for meter, data in kwargs.items():
        data_ += f"{meter}: {round(data,2)}|"
    print(f" |{chr(9608)* progress_percent_int}{' '*(50-progress_percent_int)}|{current}/{total}|{data_}",end='\r')
    if (current == total):
        print()

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"

def save_model(model, epochs, path):
    # path format: dir/model.pth
    cur_path = os.path.join(path.split('/')[0], '.'.join(path.split('/')[-1].split(".")[:-1]) + f".ec{epochs}.pth") 
    final_model = model.module if dist.is_initialized() else model 
    torch.save(final_model.state_dict(), cur_path)
    print(f"Model saved at: {cur_path}")