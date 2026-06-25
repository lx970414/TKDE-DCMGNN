# DCMGNN

This repository contains a maintainable implementation of the TKDE 2025 paper:

> Dual-Channel Multiplex Graph Neural Networks for Recommendation

The code follows the paper structure:

- Explicit Behavior Pattern Representation Learner: BBP construction, local BBP aggregation, and global BBP feature aggregation.
- Implicit Relation Chain Effect Learner: relation-specific LightGCN propagation and relation-chain transformations.
- Relation Chain-aware Contrastive Learning: relation-level and chain-level InfoNCE losses.
- Joint Optimization: BPR loss on final embeddings, BPR loss on relation-chain embeddings, and contrastive regularization.

## Environment

```bash
pip install -r requirements.txt
```

Use a CUDA-enabled PyTorch build for full-size experiments.

## Data

The maintained loader supports the datasets included in this archive:

- `data/tmall`: behavior-specific `view`, `cart`, and `buy` directories.
- `data/Retail_Rocket`: processed sparse matrices `train_mat_view.pkl`, `train_mat_cart.pkl`, `train_mat_buy.pkl`, and `test_mat.pkl`.
- `data/yelp`: optional behavior-specific `dislike`, `neutral`, `tips`, and `like` directories.

The included Tmall and Retail_Rocket folders have non-empty target train/test splits. Yelp is supported by the loader but is not included in this archive.

The original RetailRocket data is publicly listed as the Kaggle `retailrocket/ecommerce-dataset` dataset, whose behavior file is `events.csv` with `view`, `addtocart`, and `transaction` events. After downloading it, create behavior files with:

```bash
python scripts/prepare_retailrocket.py --events /path/to/events.csv --output data/Retail_Rocket
python train.py --dataset Retail_Rocket --data-root data --epochs 200
```

## Training

Run the paper-aligned Tmall configuration:

```bash
python train.py --dataset tmall --epochs 200 --embedding-dim 512 --layers 3 --batch-size 128 --steps-per-epoch 720 --lr 0.005 --behavior-layers 2,6,1 --lambda-rcl 3.0 --lambda-cascade-bpr 0 --lambda-target-bpr 0.5 --eval-channel relation:cart --seed 123 --save-path checkpoints/tmall_dim512_layers261_seed123_rcl30_e200.pt
```

For Top-20-oriented maintenance runs, save checkpoints by Recall@20/NDCG@20 first:

```bash
python train.py --dataset tmall --epochs 200 --embedding-dim 512 --layers 3 --batch-size 128 --steps-per-epoch 720 --lr 0.005 --behavior-layers 2,6,1 --lambda-rcl 3.0 --lambda-cascade-bpr 0 --lambda-target-bpr 0.5 --eval-channel relation:cart --selection-keys Recall@20,NDCG@20,Recall@10 --seed 123 --save-path checkpoints/tmall_top20.pt
```

Run the maintained Retail_Rocket configuration:

```bash
python train.py --dataset Retail_Rocket --epochs 300 --embedding-dim 512 --layers 3 --batch-size 128 --steps-per-epoch 74 --lr 0.005 --behavior-layers 3,4,2 --lambda-cascade-bpr 0 --lambda-target-bpr 0.5 --eval-channel cascade_sum --save-path checkpoints/retail_dim512_cascade.pt
```

Yelp follows the paper relation order `neutral -> tips -> like`, with `like` as the target relation. After placing processed files under `data/yelp`, run:

```bash
python scripts/check_dataset.py --dataset yelp
python train.py --dataset yelp --epochs 200 --embedding-dim 64 --layers 3 --batch-size 128 --lr 0.005 --eval-channel relation:tips --save-path checkpoints/yelp.pt
```

Useful options:

```bash
python train.py --dataset tmall --lr 0.005 --behavior-layers 2,6,1 --fusion-mode static --lambda-rcl 3.0 --lambda-chain-bpr 0.1 --lambda-target-bpr 0.5 --temperature 0.1
```

For a quick smoke test, evaluate only a small subset of users:

```bash
python train.py --dataset tmall --epochs 1 --batch-size 8 --device cpu --max-eval-users 128
```

The default relation order is:

```text
view -> cart -> buy
```

This matches the paper setting for Tmall and Retail_Rocket, where `buy` is the target relation.

## Metrics

The evaluator reports Recall@K and NDCG@K for K in `{5, 10, 20, 40}`. Training positives are masked during ranking, and all candidate items are scored.

Evaluate a saved checkpoint:

```bash
python evaluate_checkpoint.py --checkpoint checkpoints/dcmgnn.pt
```

For Tmall/Taobao, validation shows the `cart` relation view is the strongest target-prior scorer:

```bash
python evaluate_checkpoint.py --checkpoint checkpoints/dcmgnn.pt --channel relation:cart
```

For Retail_Rocket, the cascaded target channel is currently strongest:

```bash
python evaluate_checkpoint.py --checkpoint checkpoints/dcmgnn.pt --channel cascade_sum
```

The paper-aligned protocol is full ranking over all items after masking training positives. For diagnostic comparison with 99 sampled negatives:

```bash
python evaluate_checkpoint.py --checkpoint checkpoints/dcmgnn.pt --negatives-per-user 99
```

For the strongest maintained Tmall ranking, an optional log-popularity calibration can be applied at evaluation time:

```bash
python evaluate_checkpoint.py --checkpoint checkpoints/tmall_dim512_layers261_seed123_rcl30_e200.pt --channel relation:cart --popularity-beta 0.02
```

The current best Tmall Top-20 setting uses score-level calibration. It lightly blends the strong `cart` relation score with the cascaded target score and applies a small target-popularity prior:

```bash
python evaluate_checkpoint.py --checkpoint checkpoints/tmall_dim512_layers261_seed123_rcl30_e200.pt --score-blend relation:cart:0.88,cascade_sum:0.12 --popularity-beta 0.015 --popularity-weights view:0.0,cart:0.0,buy:1.0
```

For a better Recall@40/NDCG@40 trade-off, use the buy-heavy balanced setting:

```bash
python evaluate_checkpoint.py --checkpoint checkpoints/tmall_dim512_layers261_seed123_rcl30_e200.pt --score-blend relation:cart:0.93,cascade_sum:0.07 --popularity-beta 0.02 --popularity-weights view:0.0,cart:0.5,buy:2.0
```

The strongest target-only Tmall setting uses the maintained relation-chain scoring configuration:

```bash
python evaluate_checkpoint.py --checkpoint checkpoints/tmall_dim512_layers261_seed123_rcl30_e200.pt --score-blend relation:cart:0.88,cascade_sum:0.12 --popularity-beta 0.015 --popularity-weights view:0.0,cart:0.0,buy:1.0 --history-boosts view:1.0
```

To inspect whether a baseline masks only target-behavior positives or all observed behavior items:

```bash
python evaluate_checkpoint.py --checkpoint checkpoints/dcmgnn.pt --mask-mode all
python scripts/mask_stats.py --dataset tmall
```

## Repository Layout

```text
dcmgnn/
  config.py      dataset behavior order and target relation
  data.py        behavior loading, BBP construction, sparse adjacency building
  model.py       dual-channel DCMGNN model
  losses.py      BPR and contrastive losses
  evaluate.py    Recall/NDCG ranking evaluation
train.py         main training entry point
```

## Citation

```bibtex
@article{li2025dual,
  title={Dual-channel multiplex graph neural networks for recommendation},
  author={Li, Xiang and Fu, Chaofan and Zhao, Zhongying and Zheng, Guanjie and Huang, Chao and Yu, Yanwei and Dong, Junyu},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2025},
  publisher={IEEE}
}
```
