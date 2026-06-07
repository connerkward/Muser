# Muser retrieval benchmark — Flickr30k (2026-06-07)

**Dataset:** Flickr30k test split (text→image retrieval), via HF `nlphuji/flickr30k`
parquet branch. **Corpus:** 300 images. **Queries:** 300 (1 caption/image; each
caption's correct answer is its source image). **Retrieval depth k=10.**
**Engine:** real path — embed corpus → LanceDB (cosine over L2-normalized vectors) →
query → ranx metrics. Hardware: Apple Silicon (MPS).

Metrics are ranx's. `index(s)` = full corpus embed+insert time. `query(s)` = all 300
queries (embed text + LanceDB search). Per-query warm latency ≈ query(s)/300 ≈ 3-4 ms.

| model            | tier     | hits@1 | recall@5 | recall@10 |  mrr  | ndcg@10 |  map  | index(s) | query(s) |
|------------------|----------|--------|----------|-----------|-------|---------|-------|----------|----------|
| siglip2-so400m   | frontier | 0.950  | 0.993    | 1.000     | 0.970 | 0.977   | 0.970 | 85.53    | 3.32     |
| **siglip2-b** ◀  | frontier | 0.947  | 0.990    | 0.997     | 0.964 | 0.972   | 0.964 | 23.11    | 1.10     |
| clip-l14         | baseline | 0.850  | 0.983    | 0.997     | 0.907 | 0.929   | 0.907 | 21.78    | 1.00     |
| clip-b32         | baseline | 0.777  | 0.953    | 0.983     | 0.857 | 0.888   | 0.857 | 16.52    | 1.02     |

◀ = shipped default. siglip2-so400m ran separately (n=300, same split); it edges
siglip2-b by +0.003 hits@1 / +0.005 ndcg@10 but indexes ~3.7× slower (85.5s vs 23.1s,
i.e. ~285 ms/img vs ~77 ms/img) — the default's quality/speed trade is the right call.

## Notes
- jina-v4 (the documented CUDA-server default, 0.967 Flickr in repo notes) was SKIPPED:
  7.5GB download + per registry it leaks MPS memory / hard-crashes on real images.
- jina-v4-mlx is experimental (broken cross-modal retrieval) — not benchmarked.
- ~3 ms/query warm latency here vs the ~40 ms quoted in CLAUDE.md: the harness path
  has no dedup/c2pa/path-projection service overhead; it's the bare index.search.
