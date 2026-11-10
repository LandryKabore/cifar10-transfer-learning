# Machine Learning Project Report

**Course:** CMSC-5713 Artificial Intelligence  
**Student:** Landry Kabore  
**Project:** Image Classification with Transfer Learning and a Gradio Demo App

## What I built and why

I built an image classifier that fine-tunes a pretrained ResNet18 on the CIFAR-10 dataset using PyTorch, then wrapped the trained model in a Gradio web app so a user can upload an image and see predicted class probabilities.

I chose this project because it matches the approved topic and clearly goes beyond a basic from-scratch network: it uses transfer learning, nontrivial fine-tuning, and a real interactive application layer.

## Technology choices

### Framework: PyTorch
I used PyTorch because:
- torchvision provides pretrained ResNet weights and CIFAR-10 loaders out of the box
- the training loop is transparent and easy to customize for partial freezing / fine-tuning
- Gradio integrates cleanly with PyTorch tensors for inference demos

Alternatives considered:
- TensorFlow/Keras would also work for transfer learning, but PyTorch gave me more direct control over which layers stay frozen

### Architecture: ResNet18 + custom classifier head
ResNet18 is a strong ImageNet-pretrained baseline that is small enough to fine-tune on a laptop. I:
1. Loaded ImageNet-pretrained weights
2. Froze early convolutional layers
3. Fine-tuned `layer4` and replaced the final fully connected layer with Dropout + Linear(10)

This is more meaningful than training a tiny CNN from random initialization on CIFAR-10.

### Dataset: CIFAR-10
CIFAR-10 has 60,000 32×32 color images across 10 classes. It is:
- standard and reproducible for coursework
- large enough to show real training dynamics
- easy to evaluate with a held-out test set

Because ResNet expects larger inputs, images are resized to 224×224 and normalized with ImageNet mean/std.

### Application layer: Gradio
Instead of stopping at a notebook/script, I wrapped inference in Gradio so the model can be demonstrated live: upload an image → get ranked class probabilities.

## Problems and solutions

1. **Input size mismatch**  
   CIFAR-10 images are 32×32, while ResNet18 expects ~224×224.  
   **Solution:** Resize / crop transforms before training and inference.

2. **Training cost**  
   Full end-to-end fine-tuning of every layer is slower on a laptop.  
   **Solution:** Freeze early layers and only train `layer4` + the classifier head.

3. **Demo dependency on checkpoint**  
   The Gradio app needs a trained checkpoint.  
   **Solution:** Training saves `checkpoints/resnet18_cifar10_best.pt` and records metrics in `training_history.json`.

4. **Dataset download reliability**  
   The classic Toronto CIFAR-10 tarball was slow / SSL-fragile on my machine.  
   **Solution:** Load CIFAR-10 through the Hugging Face `uoft-cs/cifar10` dataset mirror and wrap it for the PyTorch DataLoader.

## What I learned

- Transfer learning can reach strong accuracy faster than training from scratch
- Freezing vs fine-tuning layers is a practical tradeoff between speed and adaptation
- Packaging a model into an interactive app forces cleaner checkpointing and preprocessing consistency between train and serve time

## How I would improve it

- Train for more epochs and unfreeze additional layers once the head stabilizes
- Add confusion-matrix analysis and per-class metrics
- Support a custom dataset (for example, a small personal photo dataset)
- Deploy the Gradio app publicly (Hugging Face Spaces) for easier sharing
- Add automated tests for preprocessing and class-label mapping

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py --epochs 5
python src/app.py
```

## Results

Trained for **5 epochs** on Apple MPS (batch size 64, Adam lr=1e-3, cosine schedule).  
Best held-out CIFAR-10 **test accuracy: 93.56%**.

| Epoch | Train loss | Train acc | Test acc |
|------:|-----------:|----------:|---------:|
| 1 | 0.4524 | 84.52% | 88.43% |
| 2 | 0.2940 | 90.06% | 88.20% |
| 3 | 0.2215 | 92.38% | 91.39% |
| 4 | 0.1467 | 95.12% | 92.63% |
| 5 | 0.0898 | 97.00% | **93.56%** |

Full metrics are also saved in `checkpoints/training_history.json`. The Gradio app loads `checkpoints/resnet18_cifar10_best.pt` and shows the same top-5 class probabilities used in the live demo.
