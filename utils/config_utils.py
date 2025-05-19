import importlib
import argparse
from omegaconf import OmegaConf

def import_class(class_path: str):
    """
    class_path: str
    """
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls

def get_instance_from_config(config, *args, **kwargs):
    cls = import_class(config["class"])
    if "params" not in config:
        return cls(*args, **kwargs)
    return cls(*args, **kwargs, **config["params"])


def parse_args():
    parser = argparse.ArgumentParser(description='Run model with configuration')
    parser.add_argument('-c', '--config', type=str, default="configs/base_config.yaml", help='Path to the configuration file')
    parser.add_argument('-p', '--param', nargs='*', help='Override parameters, e.g., -p model.params.layers=20')
    return parser.parse_args()
    
# def merge_config(config, param_list):
#     for param in param_list:
#         if '=' in param:
#             key_path, value = param.split('=')
#             keys = key_path.split('.')
#             temp = config
#             for key in keys[:-1]:
#                 temp = temp.setdefault(key, {})
#             temp[keys[-1]] = eval(value)
#     return config

def merge_base_config(config):
    if 'base_config' in config:
        base_config_path = config['base_config']
        base_config = OmegaConf.load(base_config_path)
        merged_base_config = merge_base_config(base_config)
        config = OmegaConf.merge(merged_base_config, config)
        del config['base_config']
    return config

def get_config():
    args = parse_args()
    config = OmegaConf.load(args.config)
    # if args.param:
    #     config = merge_config(config, args.param)
    cli_conf = OmegaConf.from_cli()
    config = OmegaConf.merge(config, cli_conf)
    config = merge_base_config(config)
    GlobalState.update(config["global_state"])
    return config

def save_config(config, path):
    OmegaConf.save(config, path)
    
def load_config(path):
    return OmegaConf.load(path)

GlobalState = {}

config = get_config()