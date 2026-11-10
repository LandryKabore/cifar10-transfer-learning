# CIFAR-10 Transfer Learning Classifier

CMSC-5713 Machine Learning Project — Landry Kabore

Fine-tunes a pretrained **ResNet18** (PyTorch) on **CIFAR-10**, then serves predictions through a **Gradio** web app.

## Why this project

- Goes beyond a from-scratch toy model by using **transfer learning / fine-tuning**
- Freezes early ResNet layers and trains `layer4` + a custom classifier head
- Wraps the model in a real interactive application (Gradio)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python src/train.py --epochs 5 --batch-size 64
```

Artifacts:
- `checkpoints/resnet18_cifar10_best.pt` — best model weights
- `checkpoints/training_history.json` — per-epoch metrics

## Run the demo

```bash
python src/app.py
```

Open the local Gradio URL, upload an image, and view top class probabilities.

## Project layout

```
src/model.py   # ResNet18 transfer-learning architecture
src/train.py   # CIFAR-10 fine-tuning loop
src/app.py     # Gradio inference UI
checkpoints/   # saved model + metrics
```

## Notes

CIFAR-10 images are upsampled to 224×224 so they match the ImageNet-pretrained ResNet input size.
