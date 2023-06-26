import os, torch

class AverageMeter(object):
    def __init__(self):
        self.reset ()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    
def save_checkpoint(cfg, epoch, model, optimizer, **kwargs):
    ckpt_dir = os.path.join(cfg.ckpt_dir, cfg.project, cfg.exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), f'model_state_dict_{epoch}.pt')
    torch.save(optimizer.state_dict(), f'optimizer_state_dict_{epoch}.pt')
    if kwargs is not None:
        for k, obj in kwargs.items():
            torch.save(obj, k)