# Self-retrieval benchmark — your library (300 queries)
Sampled 300 deduped images (cluster-stratified) from the live 18028-image canonical set.
Query = gpt-4o-mini natural-language description per image; hit = source image OR a near-dupe in top-10.

hits@1   0.517
recall@5 0.807
recall@10 0.880
mrr      0.640
not-found-in-top10: 36/300

tokens in=865200 out=2813  ·  COST $0.1315  (budget $0.25)  ·  54s  ·  aborted=False
