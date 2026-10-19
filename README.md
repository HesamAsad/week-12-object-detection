# COMP90086 Workshop 12: Object Detection

An interactive workbook for the Workshop 12 tutorial. It is grounded in COMP90086 lecture 21 (Object detection I: sliding windows, R-CNN, Fast R-CNN, Faster R-CNN), lecture 22 (Object detection II: YOLO, DETR, NMS, mAP) and the notebook `worksheet12_colab_new.ipynb`.

Live site: [hesamasad.github.io/week-12-object-detection](https://hesamasad.github.io/week-12-object-detection/). You can also open [`index.html`](index.html) locally in a browser. The only external requests are Google Fonts, KaTeX and highlight.js. A Light / Dark / Auto switch sits in the sidebar.

## What's inside

- **Part I, lecture primer (A1–A11).** Each topic in lecture order, with an interactive figure:
  - sliding windows with stride, scale and shape controls, and the window count;
  - IoU of two draggable boxes;
  - R-CNN, Fast R-CNN, Faster R-CNN and a single-stage detector drawn from the same blocks, coloured by what runs once per image and what runs once per region, with a step-through (redrawn after J. Johnson, EECS 498 Lecture 15);
  - where the backbone ends and the per-region network begins, for AlexNet and ResNet-50;
  - bounding-box regression, with "set the regression targets";
  - RoI pooling on a 20 × 15 feature map (with the image → CNN → features pipeline above it), and smooth ℓ1;
  - Faster R-CNN anchors, and the RPN and single-stage heads with their output tensor shapes;
  - the YOLO grid and its output tensor;
  - DETR's bipartite matching on slide 21's example;
  - NMS step by step;
  - the dog precision–recall example, with TP/FP toggles and interpolated AP;
  - mAP at each IoU threshold.
- **Part II, notebook walkthrough.** Every code cell (lightly cleaned, collapsed), next to live figures:
  - the model zoo's speed against accuracy;
  - an explorer for the 50 COCO images and their ground truth, with box-format conversion;
  - a prediction viewer with live score and IoU thresholds and TP / FP / FN colouring;
  - the classification report, precision–recall curves, mAP and per-image errors, all recomputed in the browser with FiftyOne's matching rules.
- **Part III, exercises 1–4.** Problem statement, prerequisites, a self-checking warm-up, a guided activity and an answer. Exercise 4 has a step-by-step solution with cheat notes.
- **Reference material.** Cheat notes per exercise, changes to the notebook code, a summary table and 17 references (DOIs checked on Crossref, arXiv IDs on arXiv).

## How the detections were produced

The notebook's code was run with TensorFlow 2.21 on a CPU, with the same SSD MobileNet V1 FPN 640×640 model and the same 25 evaluated images. Every detection with score ≥ 0.05 is stored, so the score threshold can change in the browser.

The run reproduces the notebook's saved output exactly: the first detection of cell 45 (elephant, 0.8228) and the per-class report agree, except for "book" and "clock". The saved notebook run recorded no detections for those two classes. Without them our mAP is the notebook's 0.4292; with them it is 0.4498.

For the exercises, three more Model Zoo detectors were run on the same images:
- SSD ResNet50 V1 FPN;
- Faster R-CNN ResNet50 V1;
- CenterNet ResNet50 V1 FPN.

Every model was also run on grayscale copies of the images, and the notebook's model on blurred copies.

## Question-by-question sequence

| Exercise | Prerequisites (inline) | Live check |
|---|---|---|
| 1 · Faster R-CNN vs SSD vs CenterNet; backbones | three detector families; the backbone | the four models on the 25 images: table, side-by-side images, per-class AP |
| 2 · `min_score_thresh` | a threshold picks one point on the P–R curve | precision, recall, F1 and COCO mAP against the threshold |
| 3 · Modified inputs | distribution shift; RGB to grayscale | mAP on original, grayscale and blurred images; per-class change |
| 4 · mAP for one image (horses) | the matching rule | IoU-threshold slider with the P–R table and AP |

## Warm-up exercises

| Before | Warm-up | Answer |
|---|---|---|
| Exercise 1 | reading the Model Zoo table | +5.2 mAP from ResNet-50; 4.5× slower |
| Exercise 2 | P and R at two thresholds on six detections | (2/3, 1/2) and (2/3, 1) |
| Exercise 3 | grayscale of orange and green | 151 and 150 |
| Exercise 4 | the lecture's dog example | (0.5, 2/3) and (0.6, 1) |

Quick checks in the primer and walkthrough: IoU on a grid (1/3), the best possible AP (1), micro against macro precision (0.457 and 0.242).

## Changes to the notebook code

- **Cell 27:** installs the Object Detection API with `pip install .`; current pip rejects `--use-feature=2020-resolver`.
- **Cell 23:** prints `session.url` as it is, since it already ends in `?polling=true`.
- **Cell 30:** converts every image to RGB, and adds the `GRAYSCALE` switch that Exercise 3 refers to.
- **Cell 35:** passes `eval_key="eval"`, so the sample-level fields of section 3(5) are written.
- **Cell 37:** keeps the 10 most common classes, as its comment says.
- **Cell 45:** prints the first prediction's evaluation fields.
- **Cell 9:** its text names the model the code actually uses.

## Key numbers (SSD MobileNet, 25 images, score ≥ 0.3)

- COCO mAP 0.450, mAP@0.5 0.594, precision 0.42, recall 0.65.
- SSD ResNet50: 0.521. CenterNet ResNet50: 0.452. Faster R-CNN ResNet50: 0.423.
- Grayscale costs 3–14 % of mAP; blur costs 47 % (SSD MobileNet).
