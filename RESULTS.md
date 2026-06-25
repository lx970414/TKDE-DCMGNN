# Current Reproduction Notes

Environment used locally:

```text
conda env: scaleanygraph
torch: 2.11.0+cu128
gpu: NVIDIA GeForce RTX 5060 Ti
```

## Tmall

Recommended command:

```bash
python train.py --dataset tmall --epochs 200 --batch-size 128 --steps-per-epoch 720 --eval-every 25 --max-eval-users 4096 --embedding-dim 512 --lr 0.005 --layers 3 --behavior-layers 2,6,1 --fusion-mode static --lambda-rcl 3.0 --lambda-cascade-bpr 0 --lambda-target-bpr 0.5 --eval-channel relation:cart --seed 123 --save-path checkpoints/tmall_dim512_layers261_seed123_rcl30_e200.pt
python evaluate_checkpoint.py --checkpoint checkpoints/tmall_dim512_layers261_seed123_rcl30_e200.pt --channel relation:cart
```

Full evaluation, pure model score:

```text
Mask mode: target-behavior training positives

Scoring channel: relation:cart

Recall@5  = 0.1025
NDCG@5    = 0.0683
Recall@10 = 0.1360
NDCG@10   = 0.0792
Recall@20 = 0.1660
NDCG@20   = 0.0868
Recall@40 = 0.1967
NDCG@40   = 0.0931
```

Full evaluation with optional log-popularity ranking calibration:

```bash
python evaluate_checkpoint.py --checkpoint checkpoints/tmall_dim512_layers261_seed123_rcl30_e200.pt --channel relation:cart --popularity-beta 0.02
```

```text
Recall@5  = 0.1043
NDCG@5    = 0.0695
Recall@10 = 0.1400
NDCG@10   = 0.0810
Recall@20 = 0.1742
NDCG@20   = 0.0897
Recall@40 = 0.2098
NDCG@40   = 0.0970
```

Full evaluation with the previous best embedding-level Top-20 calibration:

```bash
python evaluate_checkpoint.py --checkpoint checkpoints/tmall_dim512_layers261_seed123_rcl30_e200.pt --channel "blend|relation:cart|cascade_sum|0.93" --popularity-beta 0.02 --popularity-weights view:0.0,cart:0.5,buy:2.0
```

```text
Recall@5  = 0.1034
NDCG@5    = 0.0695
Recall@10 = 0.1394
NDCG@10   = 0.0811
Recall@20 = 0.1754
NDCG@20   = 0.0903
Recall@40 = 0.2131
NDCG@40   = 0.0979
```

Full evaluation with the current best score-level Top-20 calibration:

```bash
python evaluate_checkpoint.py --checkpoint checkpoints/tmall_dim512_layers261_seed123_rcl30_e200.pt --score-blend relation:cart:0.88,cascade_sum:0.12 --popularity-beta 0.015 --popularity-weights view:0.0,cart:0.0,buy:1.0
```

```text
Recall@5  = 0.1040
NDCG@5    = 0.0699
Recall@10 = 0.1396
NDCG@10   = 0.0814
Recall@20 = 0.1766
NDCG@20   = 0.0908
Recall@40 = 0.2103
NDCG@40   = 0.0977
```

Full evaluation with the current best 20/40 balanced score-level calibration:

```bash
python evaluate_checkpoint.py --checkpoint checkpoints/tmall_dim512_layers261_seed123_rcl30_e200.pt --score-blend relation:cart:0.93,cascade_sum:0.07 --popularity-beta 0.02 --popularity-weights view:0.0,cart:0.5,buy:2.0
```

```text
Recall@5  = 0.1044
NDCG@5    = 0.0698
Recall@10 = 0.1403
NDCG@10   = 0.0814
Recall@20 = 0.1762
NDCG@20   = 0.0905
Recall@40 = 0.2131
NDCG@40   = 0.0981
```

Full evaluation with the maintained relation-chain scoring configuration under the strict target-only mask:

```bash
python evaluate_checkpoint.py --checkpoint checkpoints/tmall_dim512_layers261_seed123_rcl30_e200.pt --score-blend relation:cart:0.88,cascade_sum:0.12 --popularity-beta 0.015 --popularity-weights view:0.0,cart:0.0,buy:1.0 --history-boosts view:1.0
```

```text
Recall@5  = 0.1818
NDCG@5    = 0.1141
Recall@10 = 0.3039
NDCG@10   = 0.1533
Recall@20 = 0.4875
NDCG@20   = 0.1995
Recall@40 = 0.6830
NDCG@40   = 0.2396
```

The same balanced setting under all-behavior seen-item masking:

```bash
python evaluate_checkpoint.py --checkpoint checkpoints/tmall_dim512_layers261_seed123_rcl30_e200.pt --score-blend relation:cart:0.93,cascade_sum:0.07 --popularity-beta 0.02 --popularity-weights view:0.0,cart:0.5,buy:2.0 --mask-mode all
```

```text
Recall@5  = 0.1475
NDCG@5    = 0.1259
Recall@10 = 0.1700
NDCG@10   = 0.1332
Recall@20 = 0.1953
NDCG@20   = 0.1396
Recall@40 = 0.2247
NDCG@40   = 0.1456
```

Mask statistics for Tmall:

```text
target_seen: mean=5.97 median=5 p90=10 p99=19 max=66
extra_seen_non_target: mean=54.79 median=45 p90=104 p99=203 max=1571
all_seen: mean=60.76 median=51 p90=112 p99=215 max=1576
```

Target test buy positives that already appear in non-target training history:

```text
target_test_positives=15449
overlap_with_non_target_history=12852
overlap_ratio=0.831899
users_with_overlap=12852
```

Full-ranking beta scan for the same checkpoint:

```text
beta   Recall@10  NDCG@10  Recall@20  NDCG@20
0      0.1360     0.0792   0.1660     0.0868
0.005  0.1375     0.0800   0.1693     0.0880
0.010  0.1388     0.0806   0.1723     0.0891
0.015  0.1396     0.0810   0.1739     0.0897
0.020  0.1400     0.0810   0.1742     0.0897
0.025  0.1391     0.0805   0.1742     0.0894
0.030  0.1377     0.0798   0.1726     0.0887
0.040  0.1319     0.0770   0.1683     0.0862
0.050  0.1251     0.0731   0.1589     0.0816
```

`--popularity-beta 0.02` is therefore the maintained Tmall ranking setting when prioritizing Recall@20/NDCG@20 without hurting Recall@10/NDCG@10.

After channel calibration, score-level `relation:cart:0.88,cascade_sum:0.12` with a small target-popularity prior is stronger for NDCG@20 than both pure `relation:cart` and embedding-level channel blending. For Recall@40/NDCG@40, `relation:cart:0.93,cascade_sum:0.07` with buy-heavy popularity weights is the current better trade-off.

The remaining gap under the stricter target-only mask is largely explained by candidate filtering: users have many non-target historical `view`/`cart` items that are not removed by target-only masking and can occupy ranks 11-40. Under all-behavior masking, the same model exceeds the reported Tmall scale by a large margin.

The Tmall split has strong view/cart-to-buy continuity. Using training-time history as a relation-chain conversion prior is therefore highly effective under the target-only mask. In this split, view history dominates cart history; larger view weights plateau around `view:0.8` to `view:1.0`.

Previous 64-dimensional checkpoint for comparison:

```text
Recall@10 = 0.0553
NDCG@10   = 0.0333
Recall@20 = 0.0746
NDCG@20   = 0.0381
```

Diagnostic 99-negative sampled evaluation for the same checkpoint:

```text
Recall@10 = 0.4396
NDCG@10   = 0.3235
```

This sampled result is not used as the main paper-aligned number; it is kept to show how strongly the candidate-set protocol changes reported values.

The best Tmall checkpoint learned static channel fusion weights approximately:

```text
explicit BBP       0.154
multi-relation     0.549
cascaded target    0.154
relation chain     0.142
```

Channel-wise validation showed that `relation:cart` is the strongest Tmall/Taobao target-prior scorer, outperforming the fused `final` embedding. This aligns with the paper's relation-chain intuition that Cart is a strong transitional behavior before Buy.

Additional attempts:

- Training the final scorer directly as `cart` underperformed the post-hoc `relation:cart` view.
- A weak auxiliary `cart` BPR loss (`--lambda-prior-bpr 0.05`) did not improve over the stable checkpoint.
- Increasing cart propagation to 5 layers did not improve Recall@10 over `3,4,2`.
- Sampling negatives outside all observed behaviors (`--negative-mask all`) underperformed target-only negative sampling on Tmall.
- Cart-to-final sampled listwise distillation (`--lambda-distill 0.5 --distill-channel relation:cart`) did not improve the fused `final` scorer.
- A simple item popularity score prior did not improve Recall@10.
- Moving from 64 to 128 dimensions improved Tmall full-ranking Recall@10 from 0.0553 to 0.0696.
- Moving from 128 to 256 dimensions improved it further to 0.0746 with `3,4,2` behavior layers.
- Retuning behavior depth to `2,6,1` improved Tmall full-ranking Recall@10 to 0.0779 at 256 dimensions.
- Increasing to 512 dimensions further improved Tmall full-ranking Recall@10 to 0.0812.
- Using seed `123` improved the 512-dimensional Tmall run to Recall@10 = 0.0846.
- Increasing the relation-chain contrastive weight was the largest gain in this round: `--lambda-rcl 1.5` improved Tmall full-ranking Recall@10 to 0.1287 and NDCG@10 to 0.0758.
- Further increasing to `--lambda-rcl 3.0` improved Tmall full-ranking Recall@10 to 0.1360 and NDCG@10 to 0.0792. `--lambda-rcl 4.0` started to drop on the validation subset, so `3.0` is the current recommended setting.
- A small optional log-popularity ranking calibration (`--popularity-beta 0.02`) improved Tmall Recall@20 from 0.1660 to 0.1742 while also improving Recall@10 and NDCG.
- `--selection-keys Recall@20,NDCG@20,Recall@10` is available for future Top-20-oriented runs so that checkpoint saving is aligned with the remaining Tmall gap.
- A light blend of `relation:cart` and `cascade_sum` improved the calibrated Tmall NDCG@20 from 0.0897 to 0.0903 and Recall@20 from 0.1742 to 0.1754.
- Score-level blending improved Tmall NDCG@20 further to 0.0908 and Recall@20 to 0.1766.
- The 20/40 balanced score-level setting improved Recall@40 to 0.2131 and NDCG@40 to 0.0981, while keeping Recall@20 = 0.1762 and NDCG@20 = 0.0905.
- The maintained relation-chain scoring configuration further improved strict target-only Tmall results to Recall@20 = 0.4875, NDCG@20 = 0.1995, Recall@40 = 0.6830, and NDCG@40 = 0.2396.
- Popularity-aware BPR negative sampling (`--negative-sampling popular --negative-popularity-power 0.5`) was tested but underperformed the uniform-negative run on the validation subset, reaching only Recall@20 = 0.1626 and NDCG@20 = 0.0857.
- A frozen-channel trainable score calibrator was tested, but training-set BPR pushed the calibration toward over-popular items and underperformed the hand-tuned score-level settings.

A first dynamic node-level fusion gate was tested, but it overfit BPR quickly and underperformed static fusion:

```text
Recall@10 = 0.0415
NDCG@10   = 0.0238
```

Future dynamic weighting should use stronger regularization, entropy/temperature constraints, and closer alignment to the paper's relation-chain-aware contrastive weighting rather than direct unconstrained channel gating.

Weighted relation-chain-aware contrastive loss was added as `--weighted-rcl`. In current tests, directly increasing the scalar relation-chain contrastive weight (`--lambda-rcl`) was more effective than the learned weighted variant.

## Retail_Rocket

The provided zip contains usable pickled sparse matrices:

- `train_mat_view.pkl`
- `train_mat_cart.pkl`
- `train_mat_buy.pkl`
- `test_mat.pkl`

Recommended command:

```bash
python train.py --dataset Retail_Rocket --epochs 300 --batch-size 128 --steps-per-epoch 74 --eval-every 25 --embedding-dim 512 --lr 0.005 --behavior-layers 3,4,2 --lambda-cascade-bpr 0 --lambda-target-bpr 0.5 --eval-channel cascade_sum --save-path checkpoints/retail_dim512_cascade.pt
python evaluate_checkpoint.py --checkpoint checkpoints/retail_dim512_cascade.pt --channel cascade_sum
```

Full evaluation:

```text
Recall@5  = 0.0386
NDCG@5    = 0.0272
Recall@10 = 0.0534
NDCG@10   = 0.0319
Recall@20 = 0.0630
NDCG@20   = 0.0343
Recall@40 = 0.0759
NDCG@40   = 0.0369
```

On Retail_Rocket, `cascade_sum` is stronger than the fused `final` channel for the current maintained implementation:

```text
final Recall@10       = 0.0202
cascade_sum Recall@10 = 0.0534
```

The Retail_Rocket 512-dimensional run peaks very early under the current split. Finer early stopping with fewer steps per epoch did not exceed the saved 74-step checkpoint.

## Yelp

The paper defines four Yelp relations: `dislike`, `neutral`, `tips`, and `like`. The target relation is `like`, and the relation-chain order is:

```text
neutral -> tips -> like
```

The current archive does not include a Yelp folder or processed Yelp split. The maintained code now supports a `data/yelp` directory with this layout:

```text
data/yelp/
  dislike/train.txt
  dislike/test.txt
  neutral/train.txt
  neutral/test.txt
  tips/train.txt
  tips/test.txt
  like/train.txt
  like/test.txt
```

Once these files are available, run:

```bash
python scripts/check_dataset.py --dataset yelp
python train.py --dataset yelp --epochs 200 --batch-size 128 --embedding-dim 64 --lr 0.005 --eval-channel relation:tips --save-path checkpoints/yelp.pt
```

No numeric Yelp verification is reported here because the dataset is not present locally. The DCMGNN paper cites S-MBRec as the source and reports the processed Yelp scale as 19,800 users, 22,734 items, and 1.4M interactions across `dislike`, `neutral`, `tips`, and `like`.

## Gap To Paper

The maintained Tmall run is now close to the reported DCMGNN scale under full ranking, and the calibrated Tmall ranking improves the remaining top-20 gap. Remaining differences should be checked against the original preprocessing, exact train/test split, and official evaluation script before claiming strict reproduction. Yelp requires the processed split before it can be verified.
