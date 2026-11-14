# Same-checkpoint refinement ablation

Both variants use identical 67999-step features and geometry; only labels change. No optimizer updates.

| Variant | Min views | Clusters | Nonzero CLIP | Classified | mIoU | Acc@.25 | Acc@.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_refine | 1 | 3348 | 65.77% | 65.77% | 16.68% | 28.17% | 14.08% |
| no_refine | 5 | 3348 | 65.77% | 47.94% | 18.34% | 28.17% | 18.31% |
| refine | 1 | 5263 | 40.02% | 40.02% | 15.62% | 25.35% | 11.27% |
| refine | 5 | 5263 | 40.02% | 25.85% | 17.66% | 32.39% | 16.90% |
