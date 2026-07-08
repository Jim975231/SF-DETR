import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
from prettytable import PrettyTable
from ultralytics import RTDETR
from ultralytics.utils.torch_utils import model_info

def get_weight_size(path):
    return f'{os.stat(path).st_size / 1024 / 1024:.1f}'

if __name__ == '__main__':
    model_path = r'runs\weights\best.pt'
    model = RTDETR(model_path)
    result = model.val(data='D:\Project\RTDETR-main\dataset\data.yaml',
                       split='val', imgsz=640, batch=4,
                       project='runs/val', name='exp')

    if model.task == 'detect':
        # Get basic info
        n_l, n_p, n_g, flops = model_info(model.model)
        speed = result.speed
        pre, inf, post = speed['preprocess'], speed['inference'], speed['postprocess']
        total_time = pre + inf + post

        # Model info table
        info_table = PrettyTable(title="Model Info")
        info_table.field_names = ["GFLOPs", "Parameters", "Preprocess/Img", "Inference/Img",
                                  "Postprocess/Img", "FPS(Total)", "FPS(Inference)", "Model Size(MB)"]
        info_table.add_row([f'{flops:.1f}', f'{n_p:,}',
                            f'{pre/1000:.6f}s', f'{inf/1000:.6f}s',
                            f'{post/1000:.6f}s', f'{1000/total_time:.2f}',
                            f'{1000/inf:.2f}', get_weight_size(model_path)])

        # Metrics table
        metric_table = PrettyTable(title="Model Metrics")
        metric_table.field_names = ["Class", "Precision", "Recall", "F1-Score", "mAP50", "mAP75", "mAP50-95"]
        for idx, name in enumerate(result.names.values()):
            metric_table.add_row([name,
                                  f"{result.box.p[idx]:.4f}",
                                  f"{result.box.r[idx]:.4f}",
                                  f"{result.box.f1[idx]:.4f}",
                                  f"{result.box.ap50[idx]:.4f}",
                                  f"{result.box.all_ap[idx, 5]:.4f}",
                                  f"{result.box.ap[idx]:.4f}"])
        # Average row
        metric_table.add_row(["all(average)",
                              f"{result.results_dict['metrics/precision(B)']:.4f}",
                              f"{result.results_dict['metrics/recall(B)']:.4f}",
                              f"{np.mean(result.box.f1):.4f}",
                              f"{result.results_dict['metrics/mAP50(B)']:.4f}",
                              f"{np.mean(result.box.all_ap[:, 5]):.4f}",
                              f"{result.results_dict['metrics/mAP50-95(B)']:.4f}"])

        # Print and save
        print(info_table, '\n', metric_table)
        save_path = result.save_dir / 'paper_data.txt'
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(str(info_table) + '\n' + str(metric_table))
        print(f'Results saved to {save_path}')
