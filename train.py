import warnings, os
warnings.filterwarnings('ignore')

from ultralytics import RTDETR
import random
import numpy as np
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':
    model_yamls = [
        r'models/SF-DETR.yaml'
    ]

    datasets = [
        r'data.yaml'
    ]

    runs_per_model = 1

    for i in range(runs_per_model):
        print(f"\n==============================")
        print(f"Start experiment {i+1}/{runs_per_model}")
        print(f"==============================\n")

        for dataset_path in datasets:
            dataset_name = os.path.basename(dataset_path).replace('.yaml', '')

            for yaml_path in model_yamls:
                model_name = os.path.basename(yaml_path).replace('.yaml', '')

                print(f"\n------ Training model: {model_name} on dataset: {dataset_name} (run {i+1}) ------\n")
                set_seed(0)
                model = RTDETR(yaml_path)
                model.train(
                    data=dataset_path,
                    cache=False,
                    imgsz=640,
                    epochs=100,
                    batch=4,
                    workers=0,
                    project='runs/train',
                    name='exp'
                )

                print(f"\n------ Model {model_name} on dataset {dataset_name} run {i+1} completed ------\n")
