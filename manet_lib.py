"""Shared code for the COMP560 final project: MANet pet segmentation.

Models are the same MANet blocks used in the wildfire project
(dev/manet/centered64_lib.py): GroupNorm DoubleConvs, a position-attention
block (PAB) at the bottleneck, and multi-scale fusion attention (MFAB) on
every skip connection. The only change is that `n_classes` is now honored.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Oxford-IIIT Pet trimap values
TRIMAP_PET, TRIMAP_BG, TRIMAP_BORDER = 1, 2, 3


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def _make_norm(num_channels, max_groups=8):
    groups = min(max_groups, num_channels)
    while groups > 1 and (num_channels % groups != 0):
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = _make_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = _make_norm(out_channels)

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)), inplace=True)
        x = F.relu(self.norm2(self.conv2(x)), inplace=True)
        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.doubleconv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.doubleconv(x)
        return self.pool(skip), skip


class PAB(nn.Module):
    """Position attention block: self-attention over every bottleneck pixel."""

    def __init__(self, in_channels, reduction=8):
        super().__init__()
        inter = in_channels // reduction
        self.f = nn.Conv2d(in_channels, inter, kernel_size=1)
        self.g = nn.Conv2d(in_channels, inter, kernel_size=1)
        self.h = nn.Conv2d(in_channels, inter, kernel_size=1)
        self.W_z = nn.Conv2d(inter, in_channels, kernel_size=1)
        nn.init.constant_(self.W_z.weight, 0)  # starts as identity
        nn.init.constant_(self.W_z.bias, 0)

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w
        f_x = self.f(x).view(b, -1, n).permute(0, 2, 1)
        g_x = self.g(x).view(b, -1, n)
        h_x = self.h(x).view(b, -1, n)
        beta = F.softmax(torch.bmm(f_x, g_x), -1)
        o = torch.bmm(h_x, beta.permute(0, 2, 1)).view(b, -1, h, w)
        return self.W_z(o) + x


def _interp_matrix(out_size: int, in_size: int, device, dtype) -> torch.Tensor:
    """[out_size, in_size] matrix M with M @ v == 1D linear interpolation of v
    (align_corners=False), built by interpolating the identity."""
    eye = torch.eye(in_size, device=device, dtype=torch.float64)[None]  # [1, in, in]
    m = F.interpolate(eye, size=out_size, mode="linear", align_corners=False)[0]  # [in, out]
    return m.t().contiguous().to(dtype)


def bilinear_upsample(x: torch.Tensor, size) -> torch.Tensor:
    """Same output as F.interpolate(x, size, mode="bilinear", align_corners=False),
    written as two matmuls (bilinear is separable). Upsampling a 1x1..6x6 map to
    256x256 with F.interpolate has a very slow backward pass on CUDA (thousands of
    output pixels atomically accumulating into a few input cells); matmuls avoid it."""
    h, w = size
    _, _, s_h, s_w = x.shape
    a_h = _interp_matrix(h, s_h, x.device, x.dtype)  # [H, s_h]
    a_w = _interp_matrix(w, s_w, x.device, x.dtype)  # [W, s_w]
    return a_h @ x @ a_w.t()


class MFAB(nn.Module):
    """Multi-scale fusion attention block applied to skip connections."""

    def __init__(self, in_channels, reduction=8, pool_scales=(1, 2, 4, 6)):
        super().__init__()
        inter = max(1, in_channels // reduction)
        self.pool_scales = pool_scales
        self.fuse = nn.Conv2d(in_channels * (len(pool_scales) + 1), in_channels, kernel_size=1)
        self.fc1 = nn.Conv2d(in_channels, inter, 1)
        self.fc2 = nn.Conv2d(inter, in_channels, 1)

    def forward(self, x):
        _, _, h, w = x.size()
        feats = [x]
        for s in self.pool_scales:
            pooled = F.adaptive_avg_pool2d(x, output_size=(s, s))
            feats.append(bilinear_upsample(pooled, (h, w)))
        fused = self.fuse(torch.cat(feats, dim=1))
        attn = torch.sigmoid(self.fc2(F.relu(self.fc1(F.adaptive_avg_pool2d(fused, 1)))))
        return fused * attn


class downfive(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.double_conv(x)


class upone(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels * 2, out_channels * 2)
        self.up = nn.ConvTranspose2d(out_channels * 2, out_channels, kernel_size=2, stride=2)
        self.mfab = MFAB(in_channels)

    def forward(self, x, skip):
        cat = torch.cat([self.mfab(skip), x], 1)
        return self.up(self.conv(cat))


class FinalBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mfab = MFAB(in_channels)
        self.fuse = DoubleConv(in_channels * 2, in_channels)
        self.outc = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x_dec, x1_skip):
        x = torch.cat([self.mfab(x1_skip), x_dec], dim=1)
        return self.outc(self.fuse(x))


class MANet(nn.Module):
    """Full MANet: 4 downsamples, 64 -> 1024 channels. Input H, W divisible by 16."""

    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.down1 = Down(n_channels, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = downfive(512, 1024)
        self.pab = PAB(1024)
        self.up1 = upone(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.up5 = FinalBlock(64, n_classes)

    def forward(self, x):
        x1, s1 = self.down1(x)
        x2, s2 = self.down2(x1)
        x3, s3 = self.down3(x2)
        x4, s4 = self.down4(x3)
        x5 = self.pab(self.down5(x4))
        x = self.up1(x5)
        x = self.up2(x, s4)
        x = self.up3(x, s3)
        x = self.up4(x, s2)
        return self.up5(x, s1)


class MANet_small(nn.Module):
    """3 downsamples, 64 -> 512 channels. Input H, W divisible by 8."""

    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.down1 = Down(n_channels, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = downfive(256, 512)
        self.pab = PAB(512)
        self.up1 = upone(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = FinalBlock(64, n_classes)

    def forward(self, x):
        x1, s1 = self.down1(x)
        x2, s2 = self.down2(x1)
        x3, s3 = self.down3(x2)
        x4 = self.pab(self.down4(x3))
        x = self.up1(x4)
        x = self.up2(x, s3)
        x = self.up3(x, s2)
        return self.up4(x, s1)


class MANet_small_small(nn.Module):
    """3 downsamples, 32 -> 256 channels (the wildfire model). Input H, W divisible by 8."""

    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.down1 = Down(n_channels, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = downfive(128, 256)
        self.pab = PAB(256)
        self.up1 = upone(256, 128)
        self.up2 = Up(128, 64)
        self.up3 = Up(64, 32)
        self.up4 = FinalBlock(32, n_classes)

    def forward(self, x):
        x1, s1 = self.down1(x)
        x2, s2 = self.down2(x1)
        x3, s3 = self.down3(x2)
        x4 = self.pab(self.down4(x3))
        x = self.up1(x4)
        x = self.up2(x, s3)
        x = self.up3(x, s2)
        return self.up4(x, s1)


class BasicCNN(nn.Module):
    """Baseline: the most basic encoder-decoder CNN.

    One 3x3 conv + ReLU per stage, max-pool down 3 times, transposed convs back
    up. No skip connections, no normalization, no attention.
    """

    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 2, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(16, n_classes, 1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


MODELS = {
    "MANet": MANet,
    "MANet_small": MANet_small,
    "MANet_small_small": MANet_small_small,
    "BasicCNN": BasicCNN,
}


def build_model(name: str, n_channels=3, n_classes=1) -> nn.Module:
    return MODELS[name](n_channels=n_channels, n_classes=n_classes)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def read_pet_list(root: Path, split: str):
    """Parse annotations/{trainval,test}.txt into a list of dicts.

    Columns: image name, class id (1-37), species (1=cat, 2=dog), breed id.
    """
    path = Path(root) / "annotations" / f"{split}.txt"
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, class_id, species, _ = line.split()
        breed = name.rsplit("_", 1)[0]
        rows.append({
            "name": name,
            "class_id": int(class_id),
            "species": "cat" if int(species) == 1 else "dog",
            "breed": breed,
        })
    return rows


def split_train_val(rows, val_frac=0.15, seed=42):
    """Split trainval rows into train / val, stratified by breed."""
    rng = random.Random(seed)
    by_breed = {}
    for r in rows:
        by_breed.setdefault(r["breed"], []).append(r)
    train, val = [], []
    for breed in sorted(by_breed):
        items = by_breed[breed][:]
        rng.shuffle(items)
        n_val = max(1, round(len(items) * val_frac))
        val += items[:n_val]
        train += items[n_val:]
    return train, val


def make_transforms(img_size: int, train: bool):
    if train:
        return v2.Compose([
            v2.RandomResizedCrop((img_size, img_size), scale=(0.5, 1.0), ratio=(0.75, 1.33), antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandomApply([v2.RandomRotation(15)], p=0.3),
            v2.ColorJitter(0.3, 0.3, 0.3, 0.05),  # only touches the image, not the mask
            v2.ToDtype({tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64, "others": None}, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return v2.Compose([
        v2.Resize((img_size, img_size), antialias=True),
        v2.ToDtype({tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64, "others": None}, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class PetSegDataset(Dataset):
    """Oxford-IIIT Pet images with a binary pet mask.

    Foreground = pet body + its border (trimap 1 or 3); background = trimap 2.
    Returns (image [3,H,W] normalized, mask [1,H,W] float, meta dict).
    """

    def __init__(self, root, rows, img_size=256, train=False):
        self.root = Path(root)
        self.rows = rows
        self.tf = make_transforms(img_size, train)

    def __len__(self):
        return len(self.rows)

    def load_raw(self, idx):
        r = self.rows[idx]
        img = Image.open(self.root / "images" / f"{r['name']}.jpg").convert("RGB")
        trimap = np.array(Image.open(self.root / "annotations" / "trimaps" / f"{r['name']}.png"))
        return img, trimap

    def __getitem__(self, idx):
        r = self.rows[idx]
        img, trimap = self.load_raw(idx)
        mask = (trimap != TRIMAP_BG).astype(np.uint8)
        img_t = tv_tensors.Image(torch.from_numpy(np.array(img)).permute(2, 0, 1))
        mask_t = tv_tensors.Mask(torch.from_numpy(mask)[None])
        img_t, mask_t = self.tf(img_t, mask_t)
        meta = {"name": r["name"], "species": r["species"], "breed": r["breed"]}
        return img_t.as_subclass(torch.Tensor), mask_t.as_subclass(torch.Tensor).float(), meta


def denormalize(img: torch.Tensor) -> np.ndarray:
    """[3,H,W] normalized tensor -> [H,W,3] float array in 0..1 for plotting."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (img.detach().cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def overlay(image: np.ndarray, mask: np.ndarray, color=(1.0, 0.3, 0.1), alpha=0.45) -> np.ndarray:
    """Tint `mask` pixels of an [H,W,3] 0..1 image."""
    out = image.copy()
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * np.array(color)
    return out


def mask_outline(mask: np.ndarray, width: int = 2) -> np.ndarray:
    """Return the boundary pixels of a binary mask (for drawing outlines)."""
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    k = 2 * width + 1
    dil = F.max_pool2d(t, k, stride=1, padding=width)
    ero = -F.max_pool2d(-t, k, stride=1, padding=width)
    return ((dil - ero) > 0)[0, 0].numpy()


# ---------------------------------------------------------------------------
# Loss and metrics
# ---------------------------------------------------------------------------
class DiceBCELoss(nn.Module):
    """BCE-with-logits + soft Dice. Returns (total, parts) like CenteredFireLoss."""

    def __init__(self, bce_weight=1.0, dice_weight=1.0, eps=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.eps = eps

    def forward(self, logits, targets):
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(1, 2, 3))
        denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice_loss = 1.0 - ((2 * inter + self.eps) / (denom + self.eps)).mean()
        total = self.bce_weight * bce + self.dice_weight * dice_loss
        return total, {"bce": float(bce.detach()), "dice_loss": float(dice_loss.detach())}


def per_image_scores(preds: torch.Tensor, targets: torch.Tensor, eps=1e-6):
    """Dice, IoU and pixel accuracy for each image in a batch of binary masks."""
    preds = preds.float().flatten(1)
    targets = targets.float().flatten(1)
    tp = (preds * targets).sum(1)
    fp = (preds * (1 - targets)).sum(1)
    fn = ((1 - preds) * targets).sum(1)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    acc = (preds == targets).float().mean(1)
    return dice, iou, acc


def run_epoch(model, loader, criterion, optimizer=None, scheduler=None, device="cuda", amp=True):
    """One pass over `loader`. Trains when an optimizer is given."""
    train_mode = optimizer is not None
    model.train(train_mode)
    totals = {"loss": 0.0, "dice": 0.0, "iou": 0.0, "acc": 0.0}
    n_imgs = 0
    use_amp = amp and torch.device(device).type == "cuda"

    with torch.set_grad_enabled(train_mode):
        for X, Y, _ in loader:
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(X)
            logits = logits.float()
            loss, _ = criterion(logits, Y)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            dice, iou, acc = per_image_scores(logits.detach() > 0, Y)
            b = X.size(0)
            totals["loss"] += float(loss.detach()) * b
            totals["dice"] += float(dice.sum())
            totals["iou"] += float(iou.sum())
            totals["acc"] += float(acc.sum())
            n_imgs += b

    return {k: v / max(n_imgs, 1) for k, v in totals.items()}


@torch.no_grad()
def predict_probs(model, x: torch.Tensor, tta: bool = False) -> torch.Tensor:
    """Pet probabilities for a batch already on the model's device.

    tta=True is horizontal-flip test-time augmentation: also predict on the
    mirrored image, flip that prediction back, and average the two maps.
    """
    model.eval()
    use_amp = x.device.type == "cuda"
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
        probs = torch.sigmoid(model(x).float())
        if tta:
            flipped = torch.sigmoid(model(torch.flip(x, dims=[3])).float())
            probs = 0.5 * (probs + torch.flip(flipped, dims=[3]))
    return probs.float()


@torch.no_grad()
def predict_dataset(model, loader, device="cuda", threshold=0.5, keep_tensors=False, tta=False):
    """Run the model over a loader and return one record per image."""
    model.eval()
    records = []
    for X, Y, meta in loader:
        probs = predict_probs(model, X.to(device), tta=tta).cpu()
        preds = (probs > threshold).float()
        dice, iou, acc = per_image_scores(preds, Y)
        for i in range(X.size(0)):
            rec = {
                "name": meta["name"][i],
                "species": meta["species"][i],
                "breed": meta["breed"][i],
                "dice": float(dice[i]),
                "iou": float(iou[i]),
                "acc": float(acc[i]),
                "gt_frac": float(Y[i].mean()),  # 0 => empty label (a few Oxford trimaps are broken)
            }
            if keep_tensors:
                rec.update({"image": X[i], "mask": Y[i, 0], "prob": probs[i, 0], "pred": preds[i, 0]})
            records.append(rec)
    return records


@torch.no_grad()
def predict_image(model, img: Image.Image, img_size=256, device="cuda", tta=False):
    """Predict a pet-probability map for any PIL image, returned at the original resolution."""
    img = img.convert("RGB")
    x = make_transforms(img_size, train=False)(tv_tensors.Image(torch.from_numpy(np.array(img)).permute(2, 0, 1)))
    x = x.as_subclass(torch.Tensor)[None].to(device)
    prob = predict_probs(model, x, tta=tta)
    prob = F.interpolate(prob, size=(img.height, img.width), mode="bilinear", align_corners=False)
    return prob[0, 0].cpu().numpy()


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
# The MANet checkpoints (134 MB each) are too big for git, so they live on the GitHub Release.
RELEASE_URL = "https://github.com/cjell/comp560finalproject/releases/download/v1.0"


def ensure_checkpoint(path) -> Path:
    """Return `path`, downloading the file from the GitHub Release first if it isn't on disk."""
    path = Path(path)
    if path.exists():
        return path
    url = f"{RELEASE_URL}/{path.name}"
    print(f"{path} not found locally, downloading {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.hub.download_url_to_file(url, str(path))
    except Exception as e:
        raise FileNotFoundError(
            f"Could not download {path.name}. Get it from "
            f"{RELEASE_URL.rsplit('/download', 1)[0]} and put it in {path.parent}/"
        ) from e
    return path
