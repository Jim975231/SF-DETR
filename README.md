# SF-DETR: Spatial–Frequency Synergistic Enhancement for Robust Infrared Small Target Detection in Cluttered Scenes

**This paper has been submitted to *The Visual Computer*.**

Source code and model weights are released to facilitate reproducibility and future research.

## Overview

SF-DETR is an improved RT‑DETR framework tailored for **infrared small target detection (IRSTD)** in highly cluttered scenes.  
We propose three novel modules that exploit the synergy between spatial and frequency domains to enhance weak target features and suppress complex background interference.

| Module | File | Core Function |
|--------|------|---------------|
| **SFSCBlock** | `ultralytics/nn/IRSTD_modules/SFSCBlock.py` | Backbone feature extraction with depthwise separable convolution, frequency‑domain gating, and spatial calibration |
| **SRAIM**    | `ultralytics/nn/IRSTD_modules/SRAIM.py`    | Intra‑scale feature interaction via spectral residual attention (Frequency Anomaly Miner + Transformer) |
| **CFARepC3**  | `ultralytics/nn/IRSTD_modules/CFARepC3.py`  | Lightweight frequency detail injection and coordinate attention for FPN/PAFPN fusion |

The full network architecture is defined in `model/SF-DETR.yaml`.  
All modules are integrated into the Ultralytics YOLO/RT‑DETR framework.

## Environment

- Python 3.10  
- PyTorch 2.2.2  
- CUDA 12.1  

Install dependencies:

```bash
pip install -r requirements.txt
```

## datasets

We evaluate SF-DETR on two major benchmarks:

- [DAUB](https://www.scidb.cn/en/detail?dataSetId=720626420933459968)
- [HIT‑UAV](https://www.scidb.cn/en/detail?dataSetId=90340ed7f15d4b83a32094f1cfa9b39e&version=V1)

1. **DAUB:** Since the dataset does not provide an official split, we divide it into training, validation, and test sets with a ratio of 8:1:1.
2. **HIT-UAV:** We strictly follow the official training/validation/test split.

**Note:** All local absolute Windows paths have been removed. Please ensure your `data.yaml` is configured with the correct relative paths for your local environment.


## Download Checkpoints
We provide pre-trained weights for **Seed 0** to ensure exact reproduction of the tables in the paper.
1. Go to the **[Releases](https://github.com/Jim975231/SF-DETR/releases)** section of this repository.
2. Download the weight file from the release titled **"Checkpoints for Paper Reproduction"**.
3. Create a folder named `weights` in the root directory.
4. Place the downloaded `.pt` file into the `weights/` directory.

### Training from Scratch
To train the model on your chosen dataset (ensure `data.yaml` and `SF-DETR.yaml` are correctly configured):
```bash
python train.py
Note: This script initializes the random seed to 0 as reported in the manuscript.
```
To evaluate the model and generate performance metrics (mAP, Precision, Recall, GFLOPs, and FPS breakdown):
```bash
python val.py
```

## Citation

If you use this code or the datasets in your research, please cite our paper as follows:

> “Spatial–Frequency Synergistic Enhancement for Robust Infrared Small Target Detection in Cluttered Scenes”, submitted to *The Visual Computer*, 2026. (DOI if available: [to be added])

## Acknowledgments
This work builds upon [Ultralytics RT‑DETR](https://github.com/ultralytics/ultralytics) and uses the publicly available DAUB and HIT‑UAV datasets. We thank the contributors of these resources.
