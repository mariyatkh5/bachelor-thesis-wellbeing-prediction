# model.py
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List
import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import (
    BinaryAUROC, BinaryAveragePrecision, BinaryF1Score, BinaryPrecision,
    BinaryRecall, BinaryConfusionMatrix, BinaryMatthewsCorrCoef
)

# ---------------- utils ----------------
def _act(name: str) -> nn.Module:
    """Return an activation module by (case-insensitive) name, default: ReLU."""
    n = (name or "relu").lower().replace("-", "_")
    return {
        "relu": nn.ReLU(inplace=True),
        "tanh": nn.Tanh(),
        "gelu": nn.GELU(),
        "elu": nn.ELU(inplace=True),
        "sigmoid": nn.Sigmoid(),
        "fast_sigmoid": nn.Hardsigmoid(),
        "hardsigmoid": nn.Hardsigmoid(),
    }.get(n, nn.ReLU(inplace=True))

def _init_conv(weight: torch.Tensor, act_name: str):
    """Initialization for ReLU-family, otherwise Xavier."""
    if (act_name or "relu").lower().replace("-", "_") in {"relu", "elu", "gelu"}:
        nn.init.kaiming_normal_(weight, nonlinearity="relu")
    else:
        nn.init.xavier_uniform_(weight)

def _norm_pool_name(p) -> str:
    """Normalize pooling choice to 'max' or 'avg'."""
    p = str(p).lower()
    if p in {"avg", "average", "mean"}:
        return "avg"
    if p in {"max"}:
        return "max"
    return "max"

# ---------------- metric mixin ----------------
class _MetricBufferMixin:
    """Collect probs/targets across validation and compute a rich metric set."""
    def _metrics_init(self):
        self.val_probs: List[torch.Tensor] = []
        self.val_targets: List[torch.Tensor] = []
        self.m_auroc = BinaryAUROC()
        self.m_auprc = BinaryAveragePrecision()
        self.m_f1 = BinaryF1Score(threshold=0.5)
        self.m_prec = BinaryPrecision(threshold=0.5)
        self.m_rec = BinaryRecall(threshold=0.5)
        self.m_bcm = BinaryConfusionMatrix(threshold=0.5)
        self.m_mcc = BinaryMatthewsCorrCoef(threshold=0.5)
        self._thr_sweep = torch.linspace(0.05, 0.95, steps=19)
        self.best_thr_ = 0.5
        self.best_f1_ = float("nan")

    def _metrics_reset(self):
        self.val_probs.clear()
        self.val_targets.clear()
        self.m_auroc.reset(); self.m_auprc.reset(); self.m_f1.reset()
        self.m_prec.reset(); self.m_rec.reset(); self.m_bcm.reset(); self.m_mcc.reset()
        self.best_thr_ = 0.5; self.best_f1_ = float("nan")

    def _metrics_update(self, logits: torch.Tensor, y: torch.Tensor):
        """Store sigmoid probs and targets (flattened)."""
        logits = torch.nan_to_num(logits.detach().view(-1), nan=0.0, posinf=50.0, neginf=-50.0)
        probs = torch.sigmoid(logits)
        dev = self.device if hasattr(self, "device") else probs.device
        self.val_probs.append(probs.to(dev))
        self.val_targets.append(y.detach().view(-1).to(dev))

    def _metrics_compute_and_log(self):
        """Compute AUROC/AUPRC/F1/etc., sweep best F1 threshold, and log."""
        if not self.val_probs:
            return
        dev = self.device if hasattr(self, "device") else self.val_probs[0].device
        probs = torch.cat(self.val_probs).to(dev)
        targets = torch.cat(self.val_targets).to(dev).int()
        probs = torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0).clamp_(0, 1)

        auroc = self.m_auroc(probs, targets); auprc = self.m_auprc(probs, targets)
        f1 = self.m_f1(probs, targets); prec = self.m_prec(probs, targets)
        rec = self.m_rec(probs, targets); mcc = self.m_mcc(probs, targets)
        cm = self.m_bcm(probs, targets)
        tn, fp, fn, tp = cm.view(-1).tolist() if cm.numel() == 4 else (math.nan,)*4
        tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        tnr = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        bacc = float("nan") if any(math.isnan(x) for x in [tpr, tnr]) else 0.5*(tpr + tnr)
        pos_rate = targets.float().mean().item() if targets.numel() > 0 else float("nan")

        best_f1, best_thr = -1.0, 0.5
        with torch.no_grad():
            for thr in self._thr_sweep.to(dev):
                preds = (probs >= thr).int()
                tp_ = (preds.eq(1) & targets.eq(1)).sum().item()
                fp_ = (preds.eq(1) & targets.eq(0)).sum().item()
                fn_ = (preds.eq(0) & targets.eq(1)).sum().item()
                denom = 2*tp_ + fp_ + fn_
                f1_thr = (2*tp_ / denom) if denom > 0 else 0.0
                if f1_thr > best_f1:
                    best_f1, best_thr = f1_thr, float(thr.item())
        self.best_thr_ = float(best_thr); self.best_f1_ = float(best_f1)

        self.log("val/auroc", auroc, prog_bar=True); self.log("val/auprc", auprc, prog_bar=True)
        self.log("val/f1", f1, prog_bar=True); self.log("val/precision", prec)
        self.log("val/recall", rec); self.log("val/mcc", mcc); self.log("val/bal_acc", bacc)
        self.log("val/pos_rate", pos_rate); self.log("val/best_f1", best_f1, prog_bar=True)
        self.log("val/best_thr", best_thr, prog_bar=True)
        self.log("val/cm_tn", tn); self.log("val/cm_fp", fp); self.log("val/cm_fn", fn); self.log("val/cm_tp", tp)

# ---------------- CNN blocks ----------------
class ConvBlock1d(nn.Module):
    """Conv1d + BN + Act + Dropout + optional Pooling, with same-length padding."""
    def __init__(self, in_ch, out_ch, kernel_size, act_name, dropout, pooling, pool_size):
        super().__init__()
        assert kernel_size % 2 == 1, "Use odd kernel sizes for 'same' padding."
        pad = kernel_size // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = _act(act_name)
        self.do = nn.Dropout(dropout)
        pooling = _norm_pool_name(pooling)
        if pool_size and pool_size > 1:
            self.pool = nn.MaxPool1d(pool_size) if pooling == "max" else nn.AvgPool1d(pool_size)
            self._pool_k = int(pool_size)
        else:
            self.pool = nn.Identity(); self._pool_k = 1
        _init_conv(self.conv.weight, act_name)

    def forward(self, x):
        x = self.conv(x); x = self.bn(x); x = self.act(x); x = self.do(x)
        if x.size(-1) >= self._pool_k:
            x = self.pool(x)
        return x

# ================= CNN Single =================
class CNN_Single_Lit(_MetricBufferMixin, pl.LightningModule):
    """Input: (B,1,T) or (B,T,1)  -> Logit"""
    def __init__(self, cfg: Dict[str, Any], input_shape: Tuple[int, int], pos_weight: Optional[float] = None):
        super().__init__()
        self.save_hyperparameters({"cfg": cfg, "input_shape": tuple(input_shape)})
        self._metrics_init()

        nf = int(cfg.get("num_filters", 64))
        nl = int(cfg.get("num_layers", 5))
        ks = int(cfg.get("kernel_size", 9))
        pool = _norm_pool_name(cfg.get("pooling", "max"))
        ps = int(cfg.get("pooling_size", 2))
        dr = float(cfg.get("dropout_rate", 0.2))
        act = str(cfg.get("activation", "relu"))

        layers = []; in_ch = 1
        for _ in range(nl):
            layers.append(ConvBlock1d(in_ch, nf, ks, act, dr, pool, ps)); in_ch = nf
        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1) if pool == "avg" else nn.AdaptiveMaxPool1d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dr), nn.Linear(nf, 1))
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

        # store pos_weight as a buffer for BCEWithLogitsLoss
        pw = torch.tensor([float(pos_weight) if pos_weight is not None else 1.0], dtype=torch.float32)
        if "pos_w" in self._buffers: self._buffers["pos_w"].copy_(pw)
        else: self.register_buffer("pos_w", pw)

        self.lr = float(cfg.get("learning_rate", 3e-4))
        self.weight_decay = float(cfg.get("weight_decay", 1e-4))

    def forward(self, x):
        """Accept (B,T,1) or (B,1,T); strictly require 1 channel after fixing layout."""
        x = x.float()
        if x.ndim != 3:
            raise ValueError(f"Expected 3D tensor (B,C,T) or (B,T,C), got {tuple(x.shape)}")
        # bring to (B,C,T) if needed
        if x.size(2) in (1, 2) and x.size(1) > x.size(2):
            x = x.transpose(1, 2)
        # strict single-channel check
        if x.size(1) != 1:
            raise ValueError(f"CNN_Single_Lit expects 1 channel, got {x.size(1)} (shape={tuple(x.shape)})")
        f = self.pool(self.backbone(x))
        return self.head(f)

    def _bce(self, logits, y):
        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
        return nn.BCEWithLogitsLoss(pos_weight=self.pos_w.to(logits.device))(logits, y.float())

    def on_validation_epoch_start(self): self._metrics_reset()

    def training_step(self, batch, _):
        x, y = batch; y = y.float().view(-1)
        logits = self(x).squeeze(-1); loss = self._bce(logits, y)
        preds = (torch.sigmoid(logits) >= 0.5).long(); acc = (preds == y.long()).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch; y = y.float().view(-1)
        logits = self(x).squeeze(-1); loss = self._bce(logits, y)
        preds = (torch.sigmoid(logits) >= 0.5).long(); acc = (preds == y.long()).float().mean()
        self._metrics_update(logits, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        return {"val_loss": loss, "val_acc": acc}

    def on_validation_epoch_end(self): self._metrics_compute_and_log()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}

# ================= CNN Multi-Head (EDA+ECG) =================
class CNN_MH_Lit(_MetricBufferMixin, pl.LightningModule):
    """Input: (B,2,T) or (B,T,2) -> Logit. Channel order: EDA=0, ECG=1."""
    def __init__(self, cfg: Dict[str, Any], input_shape: Tuple[int, int], pos_weight: Optional[float] = None):
        super().__init__()
        self.save_hyperparameters({"cfg": cfg, "input_shape": tuple(input_shape)})
        self._metrics_init()
        assert int(input_shape[-1]) == 2, f"Expect 2 channels, got {input_shape}"

        nf = int(cfg.get("num_filters", 64))
        nl = int(cfg.get("num_layers", 5))
        ks = int(cfg.get("kernel_size", 9))
        pool = _norm_pool_name(cfg.get("pooling", "max"))
        ps = int(cfg.get("pooling_size", 2))
        dr = float(cfg.get("dropout_rate", 0.2))
        act = str(cfg.get("activation", "relu"))

        def stack():
            s = []; in_ch = 1
            for _ in range(nl):
                s.append(ConvBlock1d(in_ch, nf, ks, act, dr, pool, ps)); in_ch = nf
            return nn.Sequential(*s)

        self.bb_eda = stack()
        self.bb_ecg = stack()
        self.pool = nn.AdaptiveAvgPool1d(1) if pool == "avg" else nn.AdaptiveMaxPool1d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dr), nn.Linear(2*nf, 1))
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

        pw = torch.tensor([float(pos_weight) if pos_weight is not None else 1.0], dtype=torch.float32)
        if "pos_w" in self._buffers: self._buffers["pos_w"].copy_(pw)
        else: self.register_buffer("pos_w", pw)

        self.lr = float(cfg.get("learning_rate", 3e-4))
        self.weight_decay = float(cfg.get("weight_decay", 1e-4))

    def forward(self, x):
        x = x.float()
        if x.ndim == 3 and x.size(2) == 2:
            pass
        elif x.ndim == 3 and x.size(1) == 2:
            x = x.transpose(1, 2)
        else:
            raise ValueError(
                f"Expected (B,T,2) or (B,2,T); got {tuple(x.shape)}. Use channel order EDA=0, ECG=1."
            )
        eda = x[..., 0].unsqueeze(1)  # (B,1,T)
        ecg = x[..., 1].unsqueeze(1)  # (B,1,T)
        fe = self.pool(self.bb_eda(eda))
        fc = self.pool(self.bb_ecg(ecg))
        return self.head(torch.cat([fe, fc], dim=1))

    def _bce(self, logits, y):
        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
        return nn.BCEWithLogitsLoss(pos_weight=self.pos_w.to(logits.device))(logits, y.float())

    def on_validation_epoch_start(self): self._metrics_reset()

    def training_step(self, batch, _):
        x, y = batch; y = y.float().view(-1)
        logits = self(x).squeeze(-1); loss = self._bce(logits, y)
        preds = (torch.sigmoid(logits) >= 0.5).long(); acc = (preds == y.long()).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch; y = y.float().view(-1)
        logits = self(x).squeeze(-1); loss = self._bce(logits, y)
        preds = (torch.sigmoid(logits) >= 0.5).long(); acc = (preds == y.long()).float().mean()
        self._metrics_update(logits, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        return {"val_loss": loss, "val_acc": acc}

    def on_validation_epoch_end(self): self._metrics_compute_and_log()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}

# ---------------- LSTM blocks ----------------
class AttnPool1D(nn.Module):
    """Single-head additive attention over time: returns weighted mean over T."""
    def __init__(self, d: int):
        super().__init__()
        self.proj = nn.Linear(d, d, bias=True)
        self.v = nn.Linear(d, 1, bias=False)
    def forward(self, x):  # x: (B,T,F)
        a = self.v(torch.tanh(self.proj(x))).squeeze(-1)  # (B,T)
        a = torch.softmax(a, dim=1)
        return torch.bmm(a.unsqueeze(1), x).squeeze(1)  # (B,F)

def _time_pool(x, mode: str):
    """Pool an LSTM output sequence (B,T,F) by 'last' or 'mean'."""
    if mode == "last": return x[:, -1, :]
    if mode == "mean": return x.mean(dim=1)
    raise ValueError(f"Unknown pool: {mode}")

# ================= LSTM Single =================
class LSTM_Single_Lit(_MetricBufferMixin, pl.LightningModule):
    """Input: (B,T,1) or (B,1,T) -> Logit"""
    def __init__(self, cfg: Dict[str, Any], input_shape: Tuple[int, int], pos_weight: Optional[float] = None):
        super().__init__()
        self.save_hyperparameters({"cfg": cfg, "input_shape": tuple(input_shape)})
        self._metrics_init()
        feat_in = int(input_shape[-1])
        hidden = int(cfg.get("hidden_size", 128))
        layers = int(cfg.get("num_layers", 1))
        bidir = bool(cfg.get("bidirectional", True))
        dr = float(cfg.get("dropout_rate", 0.2))
        pool = str(cfg.get("pooling", "last")).lower()

        self.lstm = nn.LSTM(
            input_size=feat_in, hidden_size=hidden, num_layers=layers,
            batch_first=True, dropout=(dr if layers > 1 else 0.0), bidirectional=bidir
        )
        out_dim = hidden * (2 if bidir else 1)
        self.attn = AttnPool1D(out_dim) if pool == "attn" else None
        self.pool = None if pool == "attn" else pool
        self.head = nn.Sequential(nn.Dropout(dr), nn.Linear(out_dim, 1))
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

        pw = torch.tensor([float(pos_weight) if pos_weight is not None else 1.0], dtype=torch.float32)
        if "pos_w" in self._buffers: self._buffers["pos_w"].copy_(pw)
        else: self.register_buffer("pos_w", pw)

        self.lr = float(cfg.get("learning_rate", 3e-4))
        self.weight_decay = float(cfg.get("weight_decay", 1e-4))

    def forward(self, x):
        """Accept (B,T,1) or (B,1,T); LSTM expects (B,T,F)."""
        x = x.float()
        if x.ndim == 3 and x.size(1) in (1, 2) and x.size(2) > x.size(1):
            x = x.transpose(1, 2)  # (B,1,T) -> (B,T,1)
        y, _ = self.lstm(x)
        f = self.attn(y) if self.attn is not None else _time_pool(y, self.pool)
        return self.head(f)

    def _bce(self, logits, y):
        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
        return nn.BCEWithLogitsLoss(pos_weight=self.pos_w.to(logits.device))(logits, y.float())

    def on_validation_epoch_start(self): self._metrics_reset()

    def training_step(self, batch, _):
        x, y = batch; y = y.float().view(-1)
        logits = self(x).squeeze(-1); loss = self._bce(logits, y)
        preds = (torch.sigmoid(logits) >= 0.5).long(); acc = (preds == y.long()).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch; y = y.float().view(-1)
        logits = self(x).squeeze(-1); loss = self._bce(logits, y)
        preds = (torch.sigmoid(logits) >= 0.5).long(); acc = (preds == y.long()).float().mean()
        self._metrics_update(logits, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        return {"val_loss": loss, "val_acc": acc}

    def on_validation_epoch_end(self): self._metrics_compute_and_log()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}

# ================= LSTM Multi-Head (EDA+ECG) =================
class LSTM_MH_Lit(_MetricBufferMixin, pl.LightningModule):
    """Input: (B,T,2) or (B,2,T) -> Logit. Channel order: EDA=0, ECG=1."""
    def __init__(self, cfg: Dict[str, Any], input_shape: Tuple[int, int], pos_weight: Optional[float] = None):
        super().__init__()
        self.save_hyperparameters({"cfg": cfg, "input_shape": tuple(input_shape)})
        self._metrics_init()
        assert int(input_shape[-1]) == 2, f"Expect 2 channels, got {input_shape}"
        hidden = int(cfg.get("hidden_size", 128))
        layers = int(cfg.get("num_layers", 1))
        bidir = bool(cfg.get("bidirectional", True))
        dr = float(cfg.get("dropout_rate", 0.2))
        pool = str(cfg.get("pooling", "last")).lower()

        def make_lstm():
            return nn.LSTM(
                input_size=1, hidden_size=hidden, num_layers=layers,
                batch_first=True, dropout=(dr if layers > 1 else 0.0), bidirectional=bidir
            )

        self.lstm_eda = make_lstm()
        self.lstm_ecg = make_lstm()
        out_dim = hidden * (2 if bidir else 1)
        self.attn_eda = AttnPool1D(out_dim) if pool == "attn" else None
        self.attn_ecg = AttnPool1D(out_dim) if pool == "attn" else None
        self.pool = None if pool == "attn" else pool
        self.head = nn.Sequential(nn.Dropout(dr), nn.Linear(2*out_dim, 1))
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

        pw = torch.tensor([float(pos_weight) if pos_weight is not None else 1.0], dtype=torch.float32)
        if "pos_w" in self._buffers: self._buffers["pos_w"].copy_(pw)
        else: self.register_buffer("pos_w", pw)

        self.lr = float(cfg.get("learning_rate", 3e-4))
        self.weight_decay = float(cfg.get("weight_decay", 1e-4))

    def forward(self, x):
        x = x.float()
        if x.ndim == 3 and x.size(2) == 2:
            pass
        elif x.ndim == 3 and x.size(1) == 2:
            x = x.transpose(1, 2)
        else:
            raise ValueError(
                f"Expected (B,T,2) or (B,2,T); got {tuple(x.shape)}. Use channel order EDA=0, ECG=1."
            )
        eda = x[..., 0:1]  # (B,T,1)
        ecg = x[..., 1:1+1]  # (B,T,1)
        ye, _ = self.lstm_eda(eda)
        yc, _ = self.lstm_ecg(ecg)
        fe = self.attn_eda(ye) if self.attn_eda is not None else _time_pool(ye, self.pool)
        fc = self.attn_ecg(yc) if self.attn_ecg is not None else _time_pool(yc, self.pool)
        return self.head(torch.cat([fe, fc], dim=-1))

    def _bce(self, logits, y):
        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
        return nn.BCEWithLogitsLoss(pos_weight=self.pos_w.to(logits.device))(logits, y.float())

    def on_validation_epoch_start(self): self._metrics_reset()

    def training_step(self, batch, _):
        x, y = batch; y = y.float().view(-1)
        logits = self(x).squeeze(-1); loss = self._bce(logits, y)
        preds = (torch.sigmoid(logits) >= 0.5).long(); acc = (preds == y.long()).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch; y = y.float().view(-1)
        logits = self(x).squeeze(-1); loss = self._bce(logits, y)
        preds = (torch.sigmoid(logits) >= 0.5).long(); acc = (preds == y.long()).float().mean()
        self._metrics_update(logits, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        return {"val_loss": loss, "val_acc": acc}

    def on_validation_epoch_end(self): self._metrics_compute_and_log()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}

# --------- builders ----------
def build_cnn_single(cfg: Dict[str, Any], input_shape, pos_weight: Optional[float] = None) -> pl.LightningModule:
    return CNN_Single_Lit(cfg, input_shape, pos_weight=pos_weight)

def build_cnn_mh(cfg: Dict[str, Any], input_shape, pos_weight: Optional[float] = None) -> pl.LightningModule:
    return CNN_MH_Lit(cfg, input_shape, pos_weight=pos_weight)

def build_lstm_single(cfg: Dict[str, Any], input_shape, pos_weight: Optional[float] = None) -> pl.LightningModule:
    return LSTM_Single_Lit(cfg, input_shape, pos_weight=pos_weight)

def build_lstm_mh(cfg: Dict[str, Any], input_shape, pos_weight: Optional[float] = None) -> pl.LightningModule:
    return LSTM_MH_Lit(cfg, input_shape, pos_weight=pos_weight)
