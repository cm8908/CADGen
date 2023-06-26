import yaml, os, torch, wandb

class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, path, type='yaml'):
        self.path = path
        fp = open(path, 'r')
        if type == 'yaml':
            dic = yaml.load(fp, Loader=yaml.FullLoader)
        self.update(dic)
        fp.close()
        self.setup_device()
        self.setup_logger()

    def setup_logger(self):
        wandb.login(key='5c6c600b5c9e88924902fda42a2d12e143552e0d')  # ![Caution]!
        wandb.init(project=self.project,
                   name=self.exp_name,
                   notes=self.exp_note,
                   tags=self.exp_tags)
        wandb.config = self.copy()
        
    
    def setup_device(self):
        if 'device_id' not in self.keys():
            raise KeyError('No device id specified')
        elif self.device_id == 'cpu':
            self.device = torch.device('cpu')
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.device_id)
            self.device = torch.device('cuda')
        
    
    def __repr__(self):
        ret_str = ''
        ret_str += '#'*50 + '\n'
        ret_str += f'[{self.path}]\n'
        for k, v in self.items():
            ret_str += f'{k:20}: {v}' + '\n'
        ret_str += '#'*50 + '\n'
        return ret_str


if __name__ == '__main__':
    cfg = Config('transformer_lm/configs/default_config.yaml')
    print(cfg)
    