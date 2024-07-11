import yaml, os, torch, wandb, hydra
from dotenv import load_dotenv

class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, cfg):
        load_dotenv()
        self.update(cfg)
        self.setup_device()
        self.setup_logger()

    def setup_logger(self):
        print('Your WANDB_API_KEY:', os.getenv('WANDB_API_KEY'))
        wandb.login(key=os.getenv('WANDB_API_KEY'))  # ![Caution]!
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
    cfg = load_config()
    print(cfg)
    