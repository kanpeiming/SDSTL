# Dense Prediction on Event Cameras — Paper Search Keywords

> 用于在 Google Scholar / arXiv / Semantic Scholar 检索"事件相机 + 密集预测 + 模态融合"相关论文的英文关键词清单。
> 组合检索式可直接复制粘贴；单关键词可自由 AND/OR 组合。

---

## 1. Core Concepts (核心概念词)

### 1.1 Event Camera / Event Data
- `event camera`
- `event-based camera`
- `event-based vision`
- `event data`
- `neuromorphic camera`
- `neuromorphic vision`
- `dynamic vision sensor` / `DVS`
- `DAVIS`
- `event stream` / `event stream representation`
- `event-to-frame` / `event representation`

### 1.2 Dense Prediction
- `dense prediction`
- `dense prediction tasks`
- `pixel-level prediction`
- `dense prediction benchmark`

### 1.3 Multimodal / Cross-modal Fusion
- `multimodal fusion` / `multi-modal fusion`
- `cross-modal fusion`
- `cross-modal learning`
- `cross-modal alignment`
- `sensor fusion`
- `multi-sensor fusion`
- `modality fusion`

---

## 2. Dense Prediction Tasks (密集预测任务)

### 2.1 Object Detection
- `event-based object detection`
- `event camera detection`
- `spiking object detection`
- `SpikingYOLOX` / `spiking YOLO`
- `event-based pedestrian detection`
- `Gen1` / `1Mpx` (event detection datasets)

### 2.2 Semantic / Instance / Panoptic Segmentation
- `event-based semantic segmentation`
- `event camera segmentation`
- `event-based scene segmentation`
- `event-based instance segmentation`
- `event-based panoptic segmentation`
- `DVS segmentation`
- `spiking segmentation`

### 2.3 Optical Flow
- `event-based optical flow`
- `event camera optical flow`
- `optical flow from events`
- `event-based motion estimation`
- `spike-based optical flow`

### 2.4 Depth Estimation
- `event-based depth estimation`
- `monocular depth estimation event camera`
- `event-based 3D reconstruction`
- `stereo event camera depth`

### 2.5 Image / Video Reconstruction
- `event-to-image reconstruction` / `E2VID`
- `event-to-video reconstruction`
- `event-based image reconstruction`
- `video reconstruction from events`
- `event-based video synthesis`

### 2.6 Tracking
- `event-based visual tracking`
- `event camera tracking`
- `event-based object tracking`
- `spiking tracking`

### 2.7 Visual Odometry / SLAM
- `event-based visual odometry`
- `event-based SLAM`
- `neuromorphic SLAM`
- `event camera pose estimation`

---

## 3. Modality Fusion (模态融合细节)

### 3.1 RGB–Event Fusion
- `RGB-event fusion`
- `image-event fusion`
- `RGB and event camera fusion`
- `frame-event fusion`
- `RGB-event multimodal`
- `synchronous RGB-event`

### 3.2 Event–Infrared / Other Modalities
- `event-infrared fusion`
- `event-based visible and infrared fusion`
- `event-LiDAR fusion`
- `event-thermal fusion`

### 3.3 Fusion Strategies
- `early fusion` / `late fusion` / `mid-level fusion`
- `cross-modal attention`
- `cross-modal transformer`
- `feature-level fusion`
- `cross-modal distillation`

---

## 4. SNN / Neuromorphic Methods (脉冲神经网络方法)

- `spiking neural network` / `SNN`
- `neuromorphic computing`
- `spike-based` / `event-driven SNN`
- `spiking neuron` / `LIF` / `membrane potential`
- `spiking transformer`
- `spiking convolutional network`
- `temporal efficient training` / `TET`
- `surrogate gradient` / `BPTT`
- `ann-to-snn conversion` / `ann2snn`

---

## 5. Pretraining / Learning Paradigms (预训练与学习范式)

- `self-supervised learning` / `self-supervised pretraining`
- `contrastive learning`
- `masked autoencoder` / `MAE`
- `masked event modeling`
- `event camera pretraining` / `event data dense pretraining`
- `transfer learning`
- `domain adaptation` / `unsupervised domain adaptation`
- `cross-domain transfer`
- `foundation model` / `event camera foundation model`
- `pretext task`

---

## 6. Combined Search Queries (组合检索式，可直接粘贴)

> Google Scholar 支持 `"exact phrase"` + `AND/OR`（OR 需大写，用 `{}` 分组；Scholar 的高级搜索更稳）。
> arXiv 搜索建议用纯关键词空格分隔。

### 6.1 事件相机 + 密集预测（总览）
```
"event camera" "dense prediction"
```
```
"event-based" AND ("semantic segmentation" OR "object detection" OR "optical flow" OR "depth estimation")
```
```
"event camera" ("semantic segmentation" OR "optical flow" OR "image reconstruction")
```

### 6.2 事件相机 + 模态融合
```
"event camera" "multimodal fusion"
```
```
("RGB" OR "image") AND "event" AND "fusion" AND ("detection" OR "segmentation")
```
```
"event-based" "visible and infrared fusion"
```
```
"cross-modal" "event camera" "alignment"
```

### 6.3 SNN + 事件相机 + 密集预测
```
"spiking neural network" "event camera" ("detection" OR "segmentation" OR "optical flow")
```
```
"spiking" "event-based" "dense"
```
```
"neuromorphic" ("semantic segmentation" OR "object detection")
```

### 6.4 事件数据自监督预训练
```
"event camera" "self-supervised" pretraining
```
```
"event data" "dense pre-training"
```
```
"masked" "event" "self-supervised"
```

### 6.5 迁移学习 / 跨域（承接本项目）
```
"event camera" "transfer learning" ("detection" OR "segmentation")
```
```
"domain adaptation" "event-based" ("RGB" OR "frame")
```
```
"cross-domain" "event" "spiking"
```

### 6.6 综述 / Survey（先读综述摸全景）
```
"event camera" survey
```
```
"event-based vision" "survey" 2024
```
```
"neuromorphic vision" "survey"
```

---

## 7. Author / Anchored Papers (锚点文献，顺藤摸瓜)

- **Survey**: *Recent Event Camera Innovations: A Survey* (arXiv 2408.13627, ECCV 2024)
- **Self-supervised pretraining**: *Event Camera Data Dense Pre-training* (BIT, 2025)
- **SNN detection**: *SpikingYOLOX* / *Advanced SpikingYOLOX* (ACM MM 2025)
- **SSM/Event**: *State Space Models for Event Cameras* (CVPR 2024 Spotlight)
- **Multimodal**: *Event-based Visible and Infrared Fusion via Multi-task Collaboration* (CVPR 2024)
- **Paper list**: *Awesome-Spiking-Neural-Networks* (github)

> 用法：在 Google Scholar 打开上述论文页 → 点 "Cited by" → 顺引用链找最新进展；或看作者主页近期工作。

---

## 8. Recommended Search Sources (推荐检索源)

- **Google Scholar**: https://scholar.google.com （引用链最全）
- **arXiv**: https://arxiv.org/list/cs.CV/recent （预印本，最新）
- **Semantic Scholar**: https://www.semanticscholar.org （语义相关 + 引用图）
- **Papers with Code**: https://paperswithcode.com （带 SOTA 榜单，按任务找数据集）
- **CVF Open Access**: https://openaccess.thecvf.com （CVPR/ICCV/WACV 全文）

---

*关键词清单，可持续扩展。新增方向直接追加到对应 section 即可。*
