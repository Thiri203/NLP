# app/model.py
import math
import re
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Text + encoding utilities
# =========================

def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def encode_pair_wordlevel(
    premise: str,
    hypothesis: str,
    word2id: Dict[str, int],
    max_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      input_ids: (1, max_len) LongTensor
      attention_mask: (1, max_len) LongTensor
    """
    PAD_ID = word2id["[PAD]"]
    UNK_ID = word2id["[UNK]"]
    CLS_ID = word2id["[CLS]"]
    SEP_ID = word2id["[SEP]"]

    p = normalize_text(premise)
    h = normalize_text(hypothesis)

    p_ids = [word2id.get(w, UNK_ID) for w in p.split()]
    h_ids = [word2id.get(w, UNK_ID) for w in h.split()]

    ids = [CLS_ID] + p_ids + [SEP_ID] + h_ids + [SEP_ID]

    # truncate
    if len(ids) > max_len:
        ids = ids[:max_len]
        ids[-1] = SEP_ID

    attn = [1] * len(ids)

    # pad
    pad_len = max_len - len(ids)
    if pad_len > 0:
        ids += [PAD_ID] * pad_len
        attn += [0] * pad_len

    input_ids = torch.LongTensor(ids).unsqueeze(0)
    attention_mask = torch.LongTensor(attn).unsqueeze(0)
    return input_ids, attention_mask


# =========================
# SBERT pooling + split
# =========================

def mean_pool(token_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    token_embeds: (bs, seq_len, hidden)
    attention_mask: (bs, seq_len)
    """
    in_mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
    pooled = torch.sum(token_embeds * in_mask, dim=1) / torch.clamp(in_mask.sum(dim=1), min=1e-9)
    return pooled


def split_pair_masks(input_ids: torch.Tensor, attention_mask: torch.Tensor, sep_id: int):
    """
    We used a single packed input: [CLS] premise [SEP] hypothesis [SEP] [PAD...]
    Create masks for A and B segments without relying on token_type_ids.

    Returns:
      attn_a, attn_b : (bs, L) LongTensor masks (0/1) multiplied by attention_mask later
    """
    bs, L = input_ids.shape

    # first [SEP] position per row
    sep_positions = (input_ids == sep_id).int()
    # If multiple SEP tokens, argmax gives first occurrence because it's 1/0 and argmax returns first max index
    first_sep = sep_positions.argmax(dim=1)  # (bs,)

    idxs = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(bs, L)

    # A: positions <= first_sep (include CLS and SEP)
    mask_a = (idxs <= first_sep.unsqueeze(1)).long()
    # B: positions >= first_sep (includes SEP and beyond). It's okay; attention_mask will zero pads.
    mask_b = (idxs >= first_sep.unsqueeze(1)).long()

    attn_a = attention_mask * mask_a
    attn_b = attention_mask * mask_b
    return attn_a, attn_b


def batch_uv_embeddings(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, sep_id: int):
    """
    Compute u and v sentence embeddings for packed pair input.
    """
    segment_ids = torch.zeros_like(input_ids)  # not used semantically; we split via masks
    hidden = model.encode(input_ids, segment_ids)  # (bs, L, d_model)

    attn_a, attn_b = split_pair_masks(input_ids, attention_mask, sep_id)
    u = mean_pool(hidden, attn_a)
    v = mean_pool(hidden, attn_b)
    return u, v


# =========================
# Model: BERT from scratch
# =========================

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, n_segments):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.seg_embed = nn.Embedding(n_segments, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        bs, seq_len = x.size()
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bs, seq_len)
        out = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(out)


def get_attn_pad_mask(seq_q, seq_k, pad_id=0):
    bs, len_q = seq_q.size()
    bs, len_k = seq_k.size()
    pad_attn_mask = seq_k.eq(pad_id).unsqueeze(1)  # (bs,1,len_k)
    return pad_attn_mask.expand(bs, len_q, len_k)  # (bs,len_q,len_k)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(Q.size(-1))
        scores.masked_fill_(attn_mask, -1e9)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        self.W_Q = nn.Linear(d_model, n_heads * d_k)
        self.W_K = nn.Linear(d_model, n_heads * d_k)
        self.W_V = nn.Linear(d_model, n_heads * d_v)
        self.fc = nn.Linear(n_heads * d_v, d_model)

        self.attention = ScaledDotProductAttention()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        residual = Q
        bs = Q.size(0)

        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        context, attn = self.attention(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        output = self.fc(context)

        return self.norm(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.fc2(F.gelu(self.fc1(x)))
        return self.norm(x + residual)


class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, d_k, d_v):
        super().__init__()
        self.enc_self_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class BERT(nn.Module):
    """
    Same architecture as your notebook. Includes:
      - encode(): returns last hidden states
      - forward(): returns MLM/NSP logits (kept for completeness)
    """
    def __init__(self, n_layers, n_heads, d_model, d_ff, d_k, d_v, n_segments, vocab_size, max_len, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = Embedding(vocab_size, d_model, max_len, n_segments)
        self.layers = nn.ModuleList([
            EncoderLayer(n_heads, d_model, d_ff, d_k, d_v) for _ in range(n_layers)
        ])

        self.fc_nsp = nn.Linear(d_model, 2)

        self.fc_mlm1 = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(d_model)
        self.fc_mlm2 = nn.Linear(d_model, vocab_size)

    def encode(self, input_ids, segment_ids):
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids, pad_id=self.pad_id)
        for layer in self.layers:
            output, _ = layer(output, enc_self_attn_mask)
        return output

    def forward(self, input_ids, segment_ids, masked_pos):
        output = self.encode(input_ids, segment_ids)

        # NSP head (CLS)
        cls_output = output[:, 0]
        logits_nsp = self.fc_nsp(cls_output)

        # MLM head (gather masked positions)
        bs, max_m = masked_pos.size()
        masked_pos = masked_pos.unsqueeze(-1).expand(bs, max_m, output.size(-1))
        h_masked = torch.gather(output, 1, masked_pos)

        h_masked = self.fc_mlm1(h_masked)
        h_masked = self.act(h_masked)
        h_masked = self.norm(h_masked)
        logits_lm = self.fc_mlm2(h_masked)

        return logits_lm, logits_nsp


# =========================
# SoftmaxLoss classifier head
# =========================

class SoftmaxClassifier(nn.Module):
    def __init__(self, d_model, num_labels=3):
        super().__init__()
        self.linear = nn.Linear(d_model * 3, num_labels)

    def forward(self, u, v):
        uv_abs = torch.abs(u - v)
        x = torch.cat([u, v, uv_abs], dim=-1)
        return self.linear(x)


# =========================
# Bundle: load + predict
# =========================

@dataclass
class NLIModelBundle:
    bert: BERT
    head: SoftmaxClassifier
    word2id: Dict[str, int]
    cfg: Dict
    device: torch.device

    def predict(self, premise: str, hypothesis: str) -> str:
        label_map = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}
        sep_id = self.word2id["[SEP]"]

        input_ids, attn = encode_pair_wordlevel(premise, hypothesis, self.word2id, self.cfg["max_len"])
        input_ids = input_ids.to(self.device)
        attn = attn.to(self.device)

        self.bert.eval()
        self.head.eval()

        with torch.no_grad():
            u, v = batch_uv_embeddings(self.bert, input_ids, attn, sep_id=sep_id)
            logits = self.head(u, v)
            pred = torch.argmax(logits, dim=1).item()

        return label_map[pred]


def load_nli_bundle(checkpoint_path: str, device: torch.device = torch.device("cpu")) -> NLIModelBundle:
    ckpt = torch.load(checkpoint_path, map_location=device)
    word2id = ckpt["word2id"]
    cfg = ckpt["cfg"]

    pad_id = word2id["[PAD]"]

    bert = BERT(
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        d_model=cfg["d_model"],
        d_ff=cfg["d_ff"],
        d_k=cfg["d_k"],
        d_v=cfg["d_v"],
        n_segments=cfg["n_segments"],
        vocab_size=cfg["vocab_size"],
        max_len=cfg["max_len"],
        pad_id=pad_id,
    ).to(device)

    head = SoftmaxClassifier(cfg["d_model"], num_labels=3).to(device)

    bert.load_state_dict(ckpt["bert_state_dict"], strict=True)
    head.load_state_dict(ckpt["classifier_state_dict"], strict=True)

    return NLIModelBundle(bert=bert, head=head, word2id=word2id, cfg=cfg, device=device)
