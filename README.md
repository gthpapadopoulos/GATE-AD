## Repository Layout

```text
GATE-AD/
├── configs/defaults/         # dataset-specific default configs
├── datasets/                 # local datasets
├── scripts/                  # convenience launch scripts
├── src/gate_ad/              # model/training/eval code
└── weights/                  # local checkpoints 
```

## Conda Environment Setup

Python `>=3.10` is required.

```bash
cd GATE-AD
conda create -n gate-ad python=3.10 -y
conda activate gate-ad
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Dataset Download And Placement

Download datasets from their official sources:
- MVTec AD: https://www.mvtec.com/company/research/datasets/mvtec-ad
- VisA: https://github.com/amazon-science/spot-diff
- MPDD: https://github.com/stepanje/MPDD

Create dataset folders:

```bash
mkdir -p datasets/mvtec datasets/visa datasets/mpdd
```

After downloading each dataset archive from the official source, extract/copy contents into those folders.

Place them under `GATE-AD/datasets/` like this:

```text
datasets/
├── mvtec/
├── visa/
│   └── split_csv/1cls.csv
└── mpdd/
```

## Required Dataset Format

MVTec and MPDD are expected in MVTec-style structure:

```text
<data_root>/<object_name>/
├── train/good/*.png
├── test/good/*.png
├── test/<defect_type>/*.png
└── ground_truth/<defect_type>/*_mask.png
```

VisA is loaded from `split_csv/1cls.csv` plus image/mask files referenced by that CSV:
- `configs/defaults/visa.yaml` expects `datasets/visa/split_csv/1cls.csv`
- Required CSV fields: `object`, `split`, `label`, `image`, `mask`
- `image` and `mask` paths can be relative to `datasets/visa` or absolute paths

Quick check:

```bash
test -f datasets/visa/split_csv/1cls.csv
```

## Official Backbone Weights (DINOv2 / DINOv3)

Download the official weights from:
- DINOv3: https://github.com/facebookresearch/dinov3
- DINOv2: https://github.com/facebookresearch/dinov2

Pass checkpoint path with `--backbone_ckpt` in `run_one`/`run_sweep`, or set it in `configs/defaults/*.yaml` as `backbone_ckpt`.

## Default Configs
- `configs/defaults/mvtec.yaml`
- `configs/defaults/visa.yaml`
- `configs/defaults/mpdd.yaml`




## Example Commands

Run defaults:

```bash
bash scripts/run_mvtec_default.sh
bash scripts/run_visa_default.sh
bash scripts/run_mpdd_default.sh
```

Run all defaults:

```bash
bash scripts/run_all_defaults.sh
```

Run a single object manually (MVTec defaults, explicit):

```bash
PYTHONPATH=src python -m gate_ad.cli.run_one \
  --dataset mvtec \
  --object_name screw \
  --data_root ./datasets/mvtec \
  --model_name dinov2_vitl14_reg \
  --backbone_ckpt ./weights/dinov2_vitl14_reg4_pretrain.pth \
  --device cuda:0 \
  --resolution 488 \
  --backbone_last_n_layers 8 \
  --backbone_layer_agg avg \
  --shots 1 \
  --seed 0 \
  --epochs 2000 \
  --lr 3e-4 \
  --mask_ratio 0.2 \
  --a 2.0 \
  --gnn_layers 3 \
  --latent_dim 256 \
  --gnn_hidden_dims 256,256,256 \
  --dropout 0.3 \
  --topk_ratio 0.025 \
  --image_score_pool topk_mean \
  --out_dir ./outputs_manual \
  --run_name mvtec_screw_example
```

Run VisA manual example (VisA defaults, explicit):

```bash
PYTHONPATH=src python -m gate_ad.cli.run_one \
  --dataset visa \
  --object_name chewinggum \
  --data_root ./datasets/visa \
  --visa_split_csv ./datasets/visa/split_csv/1cls.csv \
  --model_name dinov3_vitb16 \
  --backbone_ckpt ./weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --device cuda:0 \
  --resolution 512 \
  --backbone_last_n_layers 8 \
  --backbone_layer_agg avg \
  --shots 1 \
  --seed 0 \
  --epochs 2000 \
  --lr 3e-4 \
  --mask_ratio 0.2 \
  --a 2.0 \
  --gnn_layers 3 \
  --latent_dim 256 \
  --gnn_hidden_dims 256,256,256 \
  --dropout 0.3 \
  --topk_ratio 0.01 \
  --image_score_pool topk_mean \
  --out_dir ./outputs_manual \
  --run_name visa_chewinggum_example
```

## Notes
- `datasets/` and `weights/` are intentionally ignored by git.
