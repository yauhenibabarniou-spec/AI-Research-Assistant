| Name | chunk_size | chunk_overlap | embedding_model | k | hit@k | recall@k | MRR | nDCG | error |
|------|------------|---------------|-----------------|---|-------|----------|-----|------|-------|
| base | 800 | 120 | sentence-transformers/all-MiniLM-L6-v2 | 3 | 1.0000 | 0.8125 | 0.9444 | 0.8093 |  |
| small_chunks | 400 | 80 | sentence-transformers/all-MiniLM-L6-v2 | 3 | 0.9583 | 0.5208 | 0.8958 | 0.7279 |  |
| large_chunks | 1200 | 200 | sentence-transformers/all-MiniLM-L6-v2 | 3 | 0.9583 | 0.8750 | 0.8542 | 0.8103 |  |
| k_5 | 800 | 120 | sentence-transformers/all-MiniLM-L6-v2 | 5 | 1.0000 | 0.9792 | 0.9444 | 0.8951 |  |
| k_7 | 800 | 120 | sentence-transformers/all-MiniLM-L6-v2 | 7 | 1.0000 | 0.9792 | 0.9444 | 0.8951 |  |
| strict_threshold | 800 | 120 | sentence-transformers/all-MiniLM-L6-v2 | 3 | 0.2083 | 0.1042 | 0.2083 | 0.1277 |  |
