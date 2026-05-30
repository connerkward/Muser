"""NSFW scoring benchmark: Falconsai (real ViT classifier) vs zero-shot (SigLIP
concept similarity) vs a weighted blend — find the best weight, if any.

Ground truth: an INDEPENDENT strong classifier (AdamCodd ViT) as a labeling oracle,
run on a stratified sample of the user's OWN library (so the weight is tuned to the
real distribution, not an external porn dataset). This measures "agreement with a
strong independent classifier," not human labels — stated plainly.
"""

import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, ".")
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from muser.embedders import _device, _load_rgb  # noqa: E402

SCORES = os.path.expanduser("~/.muser/scores.json")


def nsfw_prob(res, pos=("nsfw", "porn")):
    return float(next((r["score"] for r in res if r["label"].lower() in pos), 0.0))


def main():
    s = json.load(open(SCORES))
    sc = s["scores"]
    canon = s["canonical"]
    ranked = sorted(canon, key=lambda p: -sc[p]["nsfw"])
    random.seed(0)
    sample = list(dict.fromkeys(ranked[:700] + random.sample(canon, 700)))  # likely-pos + background
    print(f"sample {len(sample)} images", flush=True)

    from transformers import pipeline

    dev = _device()
    falcon = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=dev)
    oracle = pipeline("image-classification", model="AdamCodd/vit-base-nsfw-detector", device=dev)

    imgs = [_load_rgb(p) for p in sample]
    print("scoring Falconsai…", flush=True)
    fp = np.array([nsfw_prob(r) for r in falcon(imgs, batch_size=16)])
    print("scoring oracle (AdamCodd)…", flush=True)
    op = np.array([nsfw_prob(r) for r in oracle(imgs, batch_size=16)])
    zs = np.array([sc[p]["nsfw"] for p in sample])  # zero-shot, percentile-normalized 0..1

    y = (op > 0.5).astype(int)
    n_pos = int(y.sum())
    print(f"\noracle positives: {n_pos}/{len(y)}", flush=True)
    if n_pos < 5 or n_pos > len(y) - 5:
        print("too few/many positives for a meaningful AUC — widen the sample.", flush=True)
        return

    def report(name, score):
        print(f"  {name:22} AUC {roc_auc_score(y, score):.3f}   AP {average_precision_score(y, score):.3f}")

    print("\n--- each signal vs oracle ---")
    report("falconsai", fp)
    report("zero-shot", zs)

    print("\n--- blend  w*falconsai + (1-w)*zeroshot ---")
    best = (-1, None)
    for w in np.linspace(0, 1, 11):
        comb = w * fp + (1 - w) * zs
        auc = roc_auc_score(y, comb)
        ap = average_precision_score(y, comb)
        print(f"  w={w:.1f}  AUC {auc:.3f}  AP {ap:.3f}")
        if auc > best[0]:
            best = (auc, w)
    print(f"\nbest blend: w={best[1]:.1f} (AUC {best[0]:.3f})  [w=1.0 ⇒ Falconsai alone is best]")
    # agreement sanity between the two real classifiers
    agree = ((fp > 0.5) == (op > 0.5)).mean()
    print(f"Falconsai vs oracle agreement: {agree:.1%}")


if __name__ == "__main__":
    main()
