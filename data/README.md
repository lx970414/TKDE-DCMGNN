# Data Notes

This directory contains the processed datasets used by the maintained DCMGNN code.

- `tmall/`: processed multi-behavior Tmall/Taobao splits with `view`, `cart`, and `buy` behaviors.
- `Retail_Rocket/`: processed RetailRocket splits with `view`, `cart`, and `buy` behaviors.
- `yelp/`: not included. The code supports the paper's Yelp behavior layout: `dislike`, `neutral`, `tips`, and `like`.

RetailRocket raw data can be obtained from Kaggle's `retailrocket/ecommerce-dataset`. Use:

```bash
python scripts/prepare_retailrocket.py --events /path/to/events.csv --output data/Retail_Rocket
```

For Yelp, place processed behavior files under:

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
