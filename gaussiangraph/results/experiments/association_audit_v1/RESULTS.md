# Frozen CF association audit

Only IoU association, pseudo filtering and view gates vary; no training or GT selection.

| Experiment | Min views | Coverage | mIoU | Acc@.25 | Acc@.5 |
|---|---:|---:|---:|---:|---:|
| cf_iou_filtered | 1 | 19.78% | 9.65% | 15.49% | 2.82% |
| cf_iou_filtered | 5 | 6.70% | 9.14% | 15.49% | 1.41% |
| cf_iou_unfiltered | 1 | 30.95% | 15.45% | 23.94% | 5.63% |
| cf_iou_unfiltered | 5 | 18.59% | 15.57% | 28.17% | 5.63% |
| sam2_cf_iou_filtered | 1 | 29.86% | 8.68% | 15.49% | 7.04% |
| sam2_cf_iou_filtered | 5 | 12.30% | 7.70% | 11.27% | 4.23% |
| sam2_cf_iou_unfiltered | 1 | 40.02% | 15.62% | 25.35% | 11.27% |
| sam2_cf_iou_unfiltered | 5 | 25.85% | 17.66% | 32.39% | 16.90% |
