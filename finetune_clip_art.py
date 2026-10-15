"""Adapt CLIP (Apple DFN5B ViT-H/14) to art by re-training its last vision layers on WikiArt.

The last 4 of the 32 vision layers (80M of 632M parameters) learn to recognise the style
(58 classes), genre (29) and artist (159) of 101,750 WikiArt paintings. A distillation term
keeps the new CLIP numbers close to the original ones, so CLIP does not forget what it knew.
The rest of CLIP stays frozen. The result, clip_art_adapter.pt (160 MB), holds only the
re-trained layers; painting_analyzer.py loads it on top of the original CLIP.

Needs: an NVIDIA GPU with 6 GB, torch, open_clip_torch, pyarrow, and the WikiArt dataset
(Hugging Face "Artificio/WikiArt", 1.6 GB) in the Hugging Face cache. Takes about 80 minutes
on an RTX 4050 Laptop GPU.

    py finetune_clip_art.py                 # train, then install as clip_art_adapter.pt
    py painting_analyzer.py train <paintings folder> --labels labels.csv   # retrain on top
"""
import os, io, glob, json, math, time, copy, random, argparse, shutil
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import pyarrow.parquet as pq
from PIL import Image
import open_clip

ap = argparse.ArgumentParser()
HERE = Path(__file__).resolve().parent
ap.add_argument("--out", default=str(HERE / "clip_art_training"))
ap.add_argument("--blocks", type=int, default=4)
ap.add_argument("--steps", type=int, default=1600)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--distill", type=float, default=2.0)
ap.add_argument("--save_at", default="500,1000,1600")
ap.add_argument("--val_every", type=int, default=250)
ap.add_argument("--no-install", action="store_true", help="do not copy the result to clip_art_adapter.pt")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
log = open(os.path.join(args.out, "log.txt"), "a", buffering=1)


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.write(s + "\n")


torch.manual_seed(0); random.seed(0); np.random.seed(0)

# ---------------- data ----------------
files = sorted(glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--Artificio--WikiArt/snapshots/*/data/*.parquet")))
t = pq.read_table(files, columns=["image", "style", "genre", "artist"])
imgs = [d["bytes"] for d in t.column("image").to_pylist()]
labels, vocab = {}, {}
for col, minimum in (("style", 200), ("genre", 200), ("artist", 150)):
    vals = t.column(col).to_pylist()
    cnt = {}
    for x in vals:
        cnt[x] = cnt.get(x, 0) + 1
    keep = sorted(x for x, c in cnt.items() if c >= minimum and x not in (None, "None", ""))
    idx = {x: i for i, x in enumerate(keep)}
    vocab[col] = keep
    labels[col] = np.array([idx.get(x, -100) for x in vals], dtype=np.int64)
del t
say("images", len(imgs), {k: len(x) for k, x in vocab.items()})
json.dump(vocab, open(os.path.join(args.out, "vocab.json"), "w"))
order = np.random.permutation(len(imgs))
val_idx, train_idx = order[:1500], order[1500:]
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


def load(i, train=True):
    im = Image.open(io.BytesIO(imgs[i])).convert("RGB")
    w, h = im.size
    if train:
        for _ in range(10):
            area = w * h * random.uniform(0.5, 1.0)
            ratio = math.exp(random.uniform(math.log(3 / 4), math.log(4 / 3)))
            cw, ch = int(round(math.sqrt(area * ratio))), int(round(math.sqrt(area / ratio)))
            if cw <= w and ch <= h:
                x0, y0 = random.randint(0, w - cw), random.randint(0, h - ch)
                im = im.crop((x0, y0, x0 + cw, y0 + ch))
                break
        if random.random() < 0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
    im = im.resize((224, 224), Image.BICUBIC)
    a = (np.asarray(im, np.float32) / 255 - MEAN) / STD
    return a.transpose(2, 0, 1)


def batch_of(ids, train=True):
    x = torch.from_numpy(np.stack([load(i, train) for i in ids]))
    y = {k: torch.from_numpy(x_[ids]) for k, x_ in labels.items()}
    return x, y


# ---------------- model ----------------
model = open_clip.create_model("ViT-H-14-quickgelu", pretrained="dfn5b")
v = model.visual
del model
v.eval()
N = args.blocks
blocks = v.transformer.resblocks
frozen_blocks, train_blocks = blocks[:-N], blocks[-N:]
teacher = nn.ModuleDict({"blocks": copy.deepcopy(train_blocks), "ln_post": copy.deepcopy(v.ln_post)})
teacher_proj = v.proj.detach().clone()
for p in v.parameters():
    p.requires_grad_(False)
v.to(torch.bfloat16)
for m in list(train_blocks) + [v.ln_post]:
    m.float()
    for p in m.parameters():
        p.requires_grad_(True)
v.proj.data = v.proj.data.float()
v.proj.requires_grad_(True)
teacher.to(torch.bfloat16)
teacher_proj = teacher_proj.to(torch.bfloat16)
for p in teacher.parameters():
    p.requires_grad_(False)
v.cuda(); teacher.cuda(); teacher_proj = teacher_proj.cuda()


class Head(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n, 1024) * 0.01)

    def forward(self, e):
        return 30 * F.normalize(e, dim=-1) @ F.normalize(self.w, dim=-1).T


heads = nn.ModuleDict({k: Head(len(x)) for k, x in vocab.items()}).cuda()
loss_w = {"style": 1.0, "genre": 1.0, "artist": 0.5}


def trunk(x):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        h = v._embeds(x.to(torch.bfloat16))
        for b in frozen_blocks:
            h = b(h)
    return h


def top(h, blks, ln, proj):
    for b in blks:
        h = b(h)
    return ln(h)[:, 0] @ proj


tparams = [p for m in list(train_blocks) + [v.ln_post] for p in m.parameters()] + [v.proj]
opt = torch.optim.AdamW([{"params": tparams, "lr": args.lr, "weight_decay": 0.05},
                         {"params": heads.parameters(), "lr": 1e-3, "weight_decay": 0.0}])
base_lrs = [g["lr"] for g in opt.param_groups]
WARM = 150


def set_lr(step):
    f = step / WARM if step < WARM else 0.5 * (1 + math.cos(math.pi * (step - WARM) / max(1, args.steps - WARM)))
    for g, b in zip(opt.param_groups, base_lrs):
        g["lr"] = b * f


def evaluate():
    heads.eval()
    correct = {k: [0, 0] for k in vocab}
    cos = []
    with torch.no_grad():
        for s in range(0, len(val_idx), 32):
            ids = val_idx[s:s + 32]
            x, y = batch_of(ids, train=False)
            h = trunk(x.cuda())
            with torch.autocast("cuda", dtype=torch.bfloat16):
                e = top(h, train_blocks, v.ln_post, v.proj).float()
                et = top(h, teacher["blocks"], teacher["ln_post"], teacher_proj).float()
            cos.append(F.cosine_similarity(e, et).cpu())
            for k in vocab:
                yy = y[k].cuda()
                m = yy >= 0
                if m.any():
                    pr = heads[k](e).argmax(-1)
                    correct[k][0] += int((pr[m] == yy[m]).sum())
                    correct[k][1] += int(m.sum())
    heads.train()
    return {k: round(c / max(n, 1), 3) for k, (c, n) in correct.items()}, round(float(torch.cat(cos).mean()), 4)


def save(step):
    state = {}
    for i, b in enumerate(train_blocks):
        for n, p in b.state_dict().items():
            state[f"transformer.resblocks.{len(blocks) - N + i}.{n}"] = p.detach().half().cpu()
    for n, p in v.ln_post.state_dict().items():
        state[f"ln_post.{n}"] = p.detach().half().cpu()
    state["proj"] = v.proj.detach().half().cpu()
    torch.save({"blocks": N, "state": state}, os.path.join(args.out, f"adapter_step{step}.pt"))
    say("  saved", f"adapter_step{step}.pt")


save_at = {int(s) for s in args.save_at.split(",")} | {args.steps}
say("trainable params (M):", round(sum(p.numel() for p in tparams) / 1e6, 1),
    "steps", args.steps, "batch", args.batch, "x", args.accum, "lr", args.lr, "distill", args.distill)
ptr, perm = 0, np.random.permutation(train_idx)
t0 = time.time()
run = {"loss": 0.0, "dist": 0.0, "n": 0}
for step in range(1, args.steps + 1):
    set_lr(step)
    opt.zero_grad(set_to_none=True)
    for _ in range(args.accum):
        if ptr + args.batch > len(perm):
            perm, ptr = np.random.permutation(train_idx), 0
        ids = perm[ptr:ptr + args.batch]
        ptr += args.batch
        x, y = batch_of(ids)
        h = trunk(x.cuda())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            e = top(h, train_blocks, v.ln_post, v.proj).float()
            with torch.no_grad():
                et = top(h, teacher["blocks"], teacher["ln_post"], teacher_proj).float()
        loss = 0.0
        for k, wgt in loss_w.items():
            yy = y[k].cuda()
            if (yy >= 0).any():
                loss = loss + wgt * F.cross_entropy(heads[k](e), yy, ignore_index=-100)
        dist = (1 - F.cosine_similarity(e, et)).mean()
        ((loss + args.distill * dist) / args.accum).backward()
        run["loss"] += float(loss); run["dist"] += float(dist); run["n"] += 1
    torch.nn.utils.clip_grad_norm_(tparams, 1.0)
    opt.step()
    if step % 50 == 0 or step == args.steps:
        el = time.time() - t0
        say(f"step {step}  loss {run['loss'] / max(run['n'], 1):.3f}  drift {run['dist'] / max(run['n'], 1):.4f}  "
            f"{step * args.batch * args.accum / el:.1f} img/s  mem {torch.cuda.max_memory_allocated() / 1e9:.2f} GB  "
            f"elapsed {el / 60:.1f} min")
        run = {"loss": 0.0, "dist": 0.0, "n": 0}
    if step % args.val_every == 0 or step == args.steps:
        acc, cs = evaluate()
        say(f"  VAL step {step}: accuracy {acc}  cosine to original {cs}")
    if step in save_at:
        save(step)
say("done in", round((time.time() - t0) / 60, 1), "min")
if not args.no_install:
    shutil.copy(os.path.join(args.out, f"adapter_step{args.steps}.pt"), HERE / "clip_art_adapter.pt")
    say("installed as", HERE / "clip_art_adapter.pt", "- now run 'py painting_analyzer.py train ...' again")
