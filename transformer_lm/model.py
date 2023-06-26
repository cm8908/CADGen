import torch
import torch.nn as nn
import sys; sys.path.append('.')
from utils.macro import PAD_IDX
from utils.config_util import Config
"""
Transformer based language model architecture

Autoregressively decode sequence starting with start token

Inputs:
    Cad sequence data [SOL, ..., EOS]

Outputs:
    Cad sequence data (reconstructed)
    
"""

class Transformer(nn.Module):
    def __init__(self, dim_emb, dim_ff, num_layers, num_heads, num_context, dropout, attn_dropout, vocab_size, **kwargs):
        super().__init__()
        
        self.vocab_size = vocab_size  # number of vocabularies (4102: 0~4101 cad tokens)
        self.num_context = num_context  # number of maximum sequence length
        self.num_layers = num_layers

        # self.register_buffer('attn_mask', torch.ones(num_context, num_context).tril()[None,None,:])  # (1, 1, N_CTX, N_CTX)
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(dim_emb)
        self.input_embedding = nn.Embedding(vocab_size + num_context + 1, dim_emb, padding_idx=PAD_IDX) 
        self.classifier = nn.Linear(dim_emb, vocab_size)
        self.drop = nn.Dropout(dropout)
        self.self_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(dim_emb, num_heads, attn_dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.feed_forward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_emb, dim_ff),
                self.activation,
                nn.Linear(dim_ff, dim_emb)
            )
            for _ in range(num_layers)
        ])
        self.norm_layers1 = nn.ModuleList([self.norm for _ in range(num_layers)])
        self.norm_layers2 = nn.ModuleList([self.norm for _ in range(num_layers)])
    
    def setup_masks(self, seq):
        """
        Args:
            seq (Tensor): (N, L)
        """
        self.key_padding_mask = ~seq.ne(PAD_IDX).bool().to(seq.device)  # (N, L)
        self.attention_mask = ~torch.ones(seq.size(1), seq.size(1), dtype=bool).tril().to(seq.device)  # (L, L)
        pass

    def attention_block(self, h, l):
        attn_out, attn_weight = self.self_attention_layers[l](
            h, h, h,
            key_padding_mask=self.key_padding_mask,
            attn_mask=self.attention_mask #torch.ones(h.size(1), h.size(1)).tril()
        )
        norm_out = self.norm_layers1[l](h + attn_out)
        ffn_out = self.feed_forward_layers[l](norm_out)
        block_out = self.norm_layers2[l](ffn_out + norm_out)
        return block_out
    
    def stack_pos_enc(self, seq):
        """_summary_

        Args:
            seq (N, L): input sequence
        Returns:
            sequence with pos_enc (N, L, 2)
        """
        bsz, seq_len = seq.shape
        pos_enc = torch.arange(self.vocab_size+1, self.vocab_size+1 + seq_len).repeat(bsz, 1,).unsqueeze(-1).to(seq.device)  # (N, L, 1)
        return torch.cat([seq.unsqueeze(-1), pos_enc], dim=-1)  # (N, L, 2)
        
    
    def forward(self, seq):
        """_summary_

        Args:
            seq (Tensor): (N, L) except for start token
        Returns:
            h (Tensor): (N, L, E)
        """
        # self.setup_masks(seq.squeeze(-1))
        self.setup_masks(seq)

        seq = self.stack_pos_enc(seq)  # (N, L, 2)
        emb_seq = self.drop(self.input_embedding(seq))  # (N, L, 2, E)
        h = emb_seq.sum(dim=2)  # (N, L, E) Position embedding added
        for l in range(self.num_layers):
            h = self.attention_block(h, l)
        return h

class LMHead(nn.Module):
    def __init__(self, model, trunc_and_reshape=False, **kwargs):
        super().__init__()
        emb_shape = model.input_embedding.weight.shape  # (vocab_size + num_context, dim_emb)
        self.dim_emb = emb_shape[1]; assert emb_shape[1] == kwargs['dim_emb']
        self.decoder = nn.Linear(emb_shape[1], emb_shape[0], bias=False)
        self.decoder.weight = model.input_embedding.weight
        self.trunc_and_reshape = trunc_and_reshape

    def forward(self, h):
        h_trunc = h[:, :-1].contiguous().view(-1, self.dim_emb) \
            if self.trunc_and_reshape else h 
        lm_logits = self.decoder(h_trunc)
        return lm_logits
        

class LanguageModel(nn.Module):
    def __init__(self, args: Config):
        super().__init__()
        self.transformer = Transformer(**args)
        self.lm_head = LMHead(self.transformer, **args)
        self.return_probs = args.return_probs
        if self.return_probs:
            pos_emb_mask = torch.zeros(1, 1, args.vocab_size + args.num_context + 1)
            pos_emb_mask[:, :, -args.num_context] = -1e12
            self.register_buffer('pos_emb_mask', pos_emb_mask)
    
    def forward(self, x):
        """_summary_

        Args:
            x (Tensor): (N, L)
        Returns:
            lm_logits (Tensor): (N, L, vocab_size+num_context)
        """
        h = self.transformer(x)  # (N, L, E)
        lm_logits = self.lm_head(h)  # (N, L, vocab_size+num_context)
        if self.return_probs:
            lm_logits = torch.softmax(lm_logits + self.pos_emb_mask, dim=-1)
        return lm_logits
    
if __name__ == '__main__':
    from types import SimpleNamespace
    from pprint import pprint
    args = SimpleNamespace()
    args.dim_emb = 64
    args.dim_ff = 128
    args.num_layers = 4
    args.num_heads = 8
    args.dropout = 0.2
    args.attn_dropout = 0.2
    args.vocab_size = 4012
    args.num_context = 512
    
    transformer = Transformer(**vars(args))
    x = torch.LongTensor([   4,    0,  229,  390,    0,  229,  485,    0,  134,  485,    0,
        134,  390,    5, 1478, 1606, 1990, 2086, 2438, 2598, 3014, 3302,
       3462, 3590, 3847,    3])[None,:,None]  # (1, 26, 1)
    trns_out = transformer.forward(x)
    print(x.shape, trns_out.shape)
    
    args.trunc_and_reshape = True
    args.return_probs = True
    lm = LanguageModel(args)
    lm_out = lm.forward(x)
    print(lm_out.shape)
    print(lm_out.argmax(-1))