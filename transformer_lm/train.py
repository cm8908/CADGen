import sys; sys.path.append('.')
import torch
import wandb
from torch import nn
from tqdm import tqdm
from model import LanguageModel
from utils.config_util import Config
from utils.data_util import get_dataloader
from utils.utils import AverageMeter, save_checkpoint
from utils.macro import PAD_IDX
import hydra

@hydra.main(version_base=None, config_path='configs', config_name='default_config')
def main(cfg):
    # Load config
    cfg = Config(cfg)
    print(cfg)

    
    train_loader = get_dataloader(cfg.data_dir, 'train', cfg.batch_size)
    val_loader = get_dataloader(cfg.data_dir, 'validation', cfg.batch_size)
    model = LanguageModel(cfg).to(cfg.device)
    criterion, optimizer = nn.NLLLoss(), torch.optim.Adam(model.parameters(), lr=1e-4)
    epoch_pbar = tqdm(range(cfg.num_epochs), desc=f'Epoch')
    for e in epoch_pbar:
        epoch_loss = run_epoch(cfg, train_loader, val_loader, criterion, optimizer, model)
        epoch_pbar.set_postfix({'epoch_loss': epoch_loss})
        save_checkpoint(cfg, e, model, optimizer)
        val_metrics: dict = run_validation(cfg, val_loader, model, criterion)
        wandb.log(val_metrics)


def run_epoch(cfg, train_loader, val_loader, criterion, optimizer, model: LanguageModel):
    # Code for training transformer based language model for an epoch
    epoch_loss = AverageMeter()
    model.train()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, x in pbar:
        x = x.to(cfg.device)
        mean_loss, _ = run_batch(cfg, x, model, criterion, optimizer)
        epoch_loss.update(mean_loss)
        pbar.set_postfix({'batch_loss': mean_loss})
        
        if (i+1) % cfg.log_interval == 0:
            wandb.log({'batch_mean_loss': mean_loss})
        if (i+1) % cfg.val_interval == 0:
            val_metrics: dict = run_validation(cfg, val_loader, model, criterion)
            wandb.log(val_metrics)
    
    return epoch_loss.avg

def run_validation(cfg, val_loader, model, criterion):
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    val_pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc='Validation')
    model.eval()
    with torch.no_grad():
        for i, x in val_pbar:
            x = x.to(cfg.device)
            mean_loss, mean_acc = run_batch(cfg, x, model, criterion, optimizer=None, is_train=False)
            val_loss.update(mean_loss)
            val_acc.update(mean_acc)
            val_pbar.set_postfix({'val_batch_loss': mean_loss, 'val_batch_acc': mean_acc})
    model.train()
    return {
        'val_loss': val_loss.avg,
        'val_acc': val_acc.avg
    }

def run_batch(cfg, batch, model: LanguageModel, criterion, optimizer, is_train=True):
    mean_loss = AverageMeter()
    mean_acc = AverageMeter()
    max_seq_len = batch.size(1)
    start_idx = 0
    for seq_len in range(1, max_seq_len):
        if cfg.bptt > 0 and seq_len > cfg.bptt:  # DEBUG: Addition results in mean_loss=nan
            start_idx += 1
        src = batch[:, start_idx:seq_len]
        trg = batch[:, start_idx+1:seq_len+1]
        
        if (batch[:, seq_len] == PAD_IDX).all():  # FIXME:
            break

        logits = model.forward(src)
        loss = criterion(logits.transpose(1,2), trg)
        mean_loss.update(loss.item())
        mean_acc.update( (logits.argmax(dim=-1) == trg).float().mean().item() )

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
    return mean_loss.avg, mean_acc.avg

if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        wandb.alert(
            title='Exception raised',
            text=str(err),
            level=wandb.AlertLevel.ERROR
        )