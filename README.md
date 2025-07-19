# Autonomous Vehicle Object Detection (ES327 Project)

This repository presents the third-year project for the ES327 module, aimed at enhancing object detection in autonomous vehicles (AVs) using the YOLOv10 architecture. The project focuses on addressing detection errors in challenging conditions such as poor lighting and adverse weather, with an emphasis on improving **recall** across key object classes.

---

##  Overview

The main goals of this project are:

- ✅ Improve detection of critical AV objects like pedestrians, cars, trucks, etc.
- 🔁 Fine-tune YOLOv10L for real-world performance.
- 🧪 Optimize detection under harsh lighting/weather conditions.

---


## Dataset Preparation

- Dataset: [nuImages](https://www.nuscenes.org/nuimages)
- Script: `data_prep.py`
- Classes:
  - `pedestrian`, `car`, `bus`, `bicycle`, `truck`, `motorcycle`
- Steps:
  - Extract relevant classes
  - Convert bounding boxes to YOLO format
  - Split into `train` and `val`
  - Save `.yaml` config for YOLOv8/10

> **Label Mapping**:
```python
{
  "pedestrian": 0,
  "car": 1,
  "bus": 2,
  "bicycle": 3,
  "truck": 4,
  "motorcycle": 5
}
```

## Model Architecture

- **Model**: YOLOv10L  
- **Framework**: [Ultralytics YOLOv8/10](https://github.com/ultralytics/ultralytics)  
- **Backbone**: CSPNet (Cross Stage Partial Network)  
- **Neck**: PAN (Path Aggregation Network)  
- **Head**: Dual structure — one for training (multi-label) and one for inference (single-label) to optimize latency  
- **Benchmarked Against**: YOLOv10n (Nano), YOLOv10m (Medium)

---

## Training Configuration

- **Input Size**: `640 × 640`  
- **Batch Size**: `16`  
- **Optimizer**: `AdamW`  
- **Learning Rate**: `1e-3 → 1e-4` (linear decay)  

### Data Augmentations:
- Rotation: ±10°  
- Scaling & Translation  
- Mosaic  
- Random Erasing  
- HSV Adjustments (hue, saturation, value)

---

## Results Summary

| Model      | mAP@50 | mAP@50–95 | F1-Score | Inference Time (ms) | FPS |
|------------|--------|-----------|----------|----------------------|-----|
| YOLOv10n   | 0.600  | 0.424     | 0.493    | 7.6                  | 132 |
| YOLOv10m   | 0.669  | 0.478     | 0.599    | 9.4                  | 106 |
| **YOLOv10L** | **0.714**  | **0.526**     | **0.660**    | **13.0**                 | **77**  |

> **Post Loss Adjustment:**
> - 🔼 **Recall**: `0.545 → 0.556`  
> - 🔽 **Precision**: `0.850 → 0.840`

---

## ⚠️ Known Issues

-  **Truck class underperformed**, likely due to visual similarity with background buildings
  - Low representation in dataset (class imbalance)
-  **Overfitting observed** in training loss curves, although still showed strong generalization on validation set

---

## 🚀 Future Work

-  Implement **adaptive loss weighting**
- Benchmark performance on **Jetson Nano** or embedded AV hardware
- Expand dataset for better generalization
- Improve detection for **visually ambiguous objects** (e.g., trucks vs buildings)

---

## Documentation

 Full technical report available in repository:  
**2200211_ES327_TR.pdf**

---

## Citation & Acknowledgements

- **Dataset**: [nuImages](https://www.nuscenes.org/nuimages) © nuScenes  
- **Model Architecture**: YOLOv10 © Tsinghua University (Wang et al., 2024)  
- **Framework**: [Ultralytics YOLOv10](https://github.com/ultralytics/ultralytics)

---

## Contact

Created as part of the **University of Warwick**'s **ES327 Individual** module.

- **Author**: *Leon Etiobi*  
- **Email**: *Leon.Etiobi@warwick.ac.uk*
