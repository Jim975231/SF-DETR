# Spatial–Frequency Synergistic Enhancement for Robust Infrared Small Target Detection in Cluttered Scenes

## **This paper has been submitted to 《*The Visual Computer》*.**  

The source code and models are released to facilitate reproducibility and further research.

##  Overview

SF-DETR is an improved RT‑DETR framework designed for **infrared small target detection (IRSTD)** in highly cluttered scenes.  
We propose three novel modules that jointly leverage **spatial and frequency domain synergies** to enhance weak target features while suppressing complex background interference.

| Module        | File Location                               | Core Function                                                |
| ------------- | ------------------------------------------- | ------------------------------------------------------------ |
| **SFSCBlock** | `ultralytics/nn/IRSTD_modules/SFSCBlock.py` | Backbone feature extraction using depthwise partial convolution + frequency‑domain gating + spatial calibration |
| **SRAIFI**    | `ultralytics/nn/IRSTD_modules/SRAIFI.py`    | Intra‑scale feature interaction via spectral‑residual attention (frequency anomaly miner + transformer) |
| **CFARepC3**  | `ultralytics/nn/IRSTD_modules/CFARepC3.py`  | Lightweight frequency detail injection + coordinate attention for FPN/PAFPN fusion |

The overall network architecture is defined in `model/SF-DETR.yaml`.  
All modules are integrated into the Ultralytics YOLO/RT‑DETR framework.

## Datasets

We evaluate on two public infrared small‑target datasets:

- [DAUB](https://www.scidb.cn/en/detail?dataSetId=720626420933459968)
- [HIT‑UAV](https://www.scidb.cn/en/detail?dataSetId=90340ed7f15d4b83a32094f1cfa9b39e&version=V1)

##  Environment

- Python 3.10

- PyTorch 2.1.0

- CUDA 12.1

  **Install dependencies:**

```
pip install -r requirements.txt
```



##  Citation

If you use this code or the datasets in your research, please cite our paper as follows:



```
[Spatial–Frequency Synergistic Enhancement for Robust Infrared Small Target Detection in Cluttered Scenes], The Visual Computer, (submitted). 
DOI of the paper (if available later): [will be added]
```



##  Acknowledgements

This work builds upon [Ultralytics RT‑DETR](https://github.com/ultralytics/ultralytics) and uses the publicly available DAUB and HIT‑UAV datasets. We thank the contributors of these resources.
