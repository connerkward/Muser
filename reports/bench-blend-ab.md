# Blend A/B — recall vs aesthetic weight (normalized), 300 queries, top-50 retrieved

| config | hits@1 | recall@5 | recall@10 | mrr |
|---|---|---|---|---|
| vec100 | 0.553 | 0.800 | 0.863 | 0.667 |
| vec70_aes30 | 0.530 | 0.797 | 0.823 | 0.641 |
| vec50_aes50 | 0.477 | 0.723 | 0.773 | 0.594 |
| vec30_aes70 | 0.277 | 0.460 | 0.550 | 0.375 |

cost $0.1309 · 52s
