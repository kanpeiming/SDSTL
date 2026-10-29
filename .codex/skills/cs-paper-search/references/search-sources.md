# Search Sources and Venue Map

Use this reference when choosing where to search computer science papers or when the user asks for top venues, journals, or open source paper lists.

## Primary Search Sources

- arXiv: https://arxiv.org
- OpenReview: https://openreview.net
- CVF Open Access: https://openaccess.thecvf.com
- Papers with Code: https://paperswithcode.com
- Semantic Scholar: https://www.semanticscholar.org
- IEEE Xplore: https://ieeexplore.ieee.org
- ACM Digital Library: https://dl.acm.org
- Springer: https://link.springer.com
- ScienceDirect: https://www.sciencedirect.com
- Google Scholar: https://scholar.google.com

Google Scholar can be useful for citation chains, but automated access may be limited. Prefer arXiv, official proceedings, OpenReview, CVF, DOI pages, Semantic Scholar, and project pages when they provide enough information.

## Top Conferences

Computer vision and multimodal learning:

- CVPR: https://openaccess.thecvf.com/CVPR
- ICCV: https://openaccess.thecvf.com/ICCV
- ECCV: https://www.ecva.net/papers.php
- WACV: https://openaccess.thecvf.com/WACV
- ACM MM: https://dl.acm.org/conference/mm

Machine learning and AI:

- NeurIPS: https://papers.nips.cc
- ICML: https://proceedings.mlr.press
- ICLR: https://openreview.net/group?id=ICLR.cc
- AAAI: https://ojs.aaai.org/index.php/AAAI
- IJCAI: https://www.ijcai.org/proceedings

Robotics and event camera adjacent venues:

- ICRA: https://ieeexplore.ieee.org/xpl/conhome/1000639/all-proceedings
- IROS: https://ieeexplore.ieee.org/xpl/conhome/1000393/all-proceedings
- RA-L: https://www.ieee-ras.org/publications/ra-l

## Strong Journals

- IEEE TPAMI
- IEEE TIP
- IEEE TNNLS
- IEEE TCSVT
- IEEE TMM
- International Journal of Computer Vision
- Pattern Recognition
- Neural Networks
- Information Fusion
- Expert Systems with Applications
- Neurocomputing

## Useful GitHub Paper Lists

- Awesome Event-based Vision lists
- Awesome Spiking Neural Networks lists
- Awesome Neuromorphic Vision lists
- Awesome Domain Adaptation lists
- Papers with Code task pages for event camera, SNN, object classification, detection, segmentation, optical flow, and depth estimation

Treat curated lists as discovery tools. Verify each selected paper against its official page, arXiv page, DOI, or proceedings page before presenting it as evidence.

## Seed Query Patterns

Recent event camera and SNN:

```text
"event camera" +"spiking neural network" 2025
"event-based vision" +"spiking" +"classification"
"DVS" +"SNN" +"transfer learning"
```

RGB, frame, and event transfer:

```text
"RGB to event" +"transfer learning"
"image-to-event" +"spiking"
"video to events" +"event camera" +"classification"
"event camera" +"domain adaptation" +"frames"
```

Attention and alignment:

```text
"event camera" +"attention" +"classification"
"cross-modal attention" +"event camera"
"RGB-event alignment" +"event camera"
```

High frequency and pooling in SNN:

```text
"spiking neural networks" +"high frequency"
"SNN" +"MaxPool" +"AvgPool"
"frequency bias" +"spiking neural network"
```

Dense prediction expansion:

```text
"event camera" +"dense prediction"
"event-based" +"semantic segmentation"
"event camera" +"object detection" +"spiking"
"event-based optical flow" +"SNN"
```

## Suggested Deliverable Shapes

For a quick scan, return 5 to 8 papers grouped by relevance.

For an experiment bottleneck, return:

1. Most relevant papers.
2. What each suggests changing.
3. Which idea is lowest risk to test first.
4. Which result would support or refute the user's current hypothesis.

For a related work section, group papers by theme rather than by search source.
