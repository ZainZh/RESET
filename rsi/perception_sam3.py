# SAM3 text-grounded block perception (new module; rsi/perception.py untouched).
# Colour-free: prompt "small cube", lift instance masks to world points with depth, heights from a fitted table plane.
# Low-score instances (0.3-0.5) are accepted only with enough points and a block-like top height; the bowl is
# located with SAM3 plus a cylinder fit and its disc (fit radius + margin) is excluded.
import os
import numpy as np
from PIL import Image
from rsi import perception_geo as G, perception as P

CKPT = os.environ.get("SAM3_CHECKPOINT", "sam3.pt")


def build(device="cuda", threshold=0.3):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    torch.inference_mode().__enter__()
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    model = build_sam3_image_model(checkpoint_path=CKPT, load_from_HF=False)
    return Sam3Processor(model, confidence_threshold=threshold)


def _prompt(proc, state, text):
    if hasattr(proc, "remote_prompt"):
        return proc.remote_prompt(state, text)
    proc.reset_all_prompts(state)
    out = proc.set_text_prompt(text, state)
    n = len(out["scores"]) if "scores" in out else 0
    if not n:
        return np.zeros((0, 1, 1), bool), np.zeros(0)
    return out["masks"].squeeze(1).cpu().numpy().astype(bool), out["scores"].float().cpu().numpy()


def locate_bowl(proc, state, p, ys, xs, plane):
    masks, scores = _prompt(proc, state, "bowl")
    if not len(scores) or scores.max() < 0.5:
        return None
    m = masks[int(scores.argmax())]
    q = p[m[ys, xs]]
    if len(q) < 200:
        return None
    hq = G.height(q, plane)
    top = q[hq > np.percentile(hq, 90)]
    xy, r = P.fit_upright_cylinder(top)
    return np.asarray(xy, float), float(r)


def find_blocks(proc, obs, prompt=("small cube", "block"), plane=None, bowl="auto", min_score=0.7, low_score=0.3, min_pts=40, drop=0.010, margin=0.01):
    plane = plane or G.fit_plane(obs)
    p, ys, xs = G.safe_points(obs)
    h = G.height(p, plane)
    state = proc.set_image(Image.fromarray(obs["rgb"]))
    if isinstance(bowl, str):
        bowl = locate_bowl(proc, state, p, ys, xs, plane)
    prompts = (prompt,) if isinstance(prompt, str) else tuple(prompt)
    pairs = []
    for pr in prompts:
        mk, sc = _prompt(proc, state, pr)
        pairs += list(zip(mk, sc))
    res = []
    for m, s in sorted(pairs, key=lambda x: -x[1]):
        if s < low_score:
            continue
        sel = m[ys, xs] & (h > 0.005)
        if sel.sum() < min_pts:
            continue
        q = p[sel]
        hq = h[sel]
        p95 = float(np.percentile(hq, 95))
        keep = hq < p95 + 0.015
        q = q[keep]
        hq = hq[keep]
        top = q[hq > p95 - drop]
        c = np.r_[top[:, :2].mean(0), np.median(top[:, 2])]
        topz = p95
        if s < min_score and not (sel.sum() >= 100 and 0.022 <= topz <= 0.040):
            continue
        if bowl is not None and np.linalg.norm(c[:2] - bowl[0]) < bowl[1] + margin:
            continue
        if topz > 0.060:
            continue
        if any(np.linalg.norm(c[:2] - r["center"][:2]) < 0.015 for r in res):
            continue
        rgb = obs["rgb"][ys[sel], xs[sel]]
        res.append({"center": c, "n": int(sel.sum()), "score": float(s), "top": topz, "points": q, "mean_rgb": rgb.mean(0)})
    return sorted(res, key=lambda b: -b["score"]), bowl


_PROC = None


def get_proc():
    global _PROC
    if _PROC is None:
        _PROC = RemoteProc() if os.environ.get("RSI_SAM3_REMOTE") else build()
    return _PROC


class RemoteProc:
    """Client of rsi/sam3_service.py: same calls find_blocks needs, no local model."""

    def __init__(self):
        from multiprocessing.connection import Client
        self.conn = Client(os.environ.get("RSI_SAM3_ADDR", __import__('tempfile').gettempdir() + "/sam3.sock"), "AF_UNIX", authkey=b"rsi-sam3")

    def set_image(self, image):
        return {"rgb": np.asarray(image)}

    def reset_all_prompts(self, state):
        pass

    def remote_prompt(self, state, text):
        self.conn.send((state["rgb"], [text]))
        masks, scores = self.conn.recv()[0]
        return masks, scores


def locate_object(proc, obs, prompt, plane=None, min_score=0.5, min_pts=100, max_height=0.20):
    """Highest-score SAM3 instance for a text prompt, lifted to world points (heights from the fitted plane).
    Returns {points, top, center, table, score, n} like perception.locate, or None."""
    plane = plane or G.fit_plane(obs)
    p, ys, xs = G.safe_points(obs)
    h = G.height(p, plane)
    state = proc.set_image(Image.fromarray(obs["rgb"]))
    masks, scores = _prompt(proc, state, prompt)
    if not len(scores) or scores.max() < min_score:
        return None
    m = masks[int(scores.argmax())]
    sel = m[ys, xs] & (h > 0.005) & (h < max_height)
    if sel.sum() < min_pts:
        return None
    q = p[sel]
    return {"points": q, "top": float(np.percentile(q[:, 2], 95)), "center": np.r_[np.median(q[:, :2], 0), np.median(q[:, 2])],
            "table": float(plane[0][2]), "score": float(scores.max()), "n": int(sel.sum())}
