# Machine Learning Project Report

**Course:** CMSC-5713 Artificial Intelligence  
**Student:** Landry Kabore  
**Project:** Image Classification with Transfer Learning and a Gradio Demo App

## What I built and why

For this project I fine-tuned a pretrained ResNet18 on CIFAR-10 in PyTorch, then put the model behind a small Gradio app so you can upload a picture and see the class scores.

I picked this because I wanted something past a tiny CNN trained from scratch. Transfer learning felt like a better fit for the “go beyond a basic model” requirement, and wrapping it in Gradio made it easy to show in the demo video instead of just printing accuracy in a terminal.

## Technology choices

I went with **PyTorch** mostly because torchvision already has ResNet weights and the training loop is easy to mess with. I looked at TensorFlow/Keras too, but freezing some layers and swapping the head felt more straightforward in PyTorch for me.

For the model I used **ResNet18**. It’s pretrained on ImageNet, not huge, and runs fine on my Mac. What I did:

- load the pretrained weights
- freeze the early layers
- fine-tune `layer4` plus a new classifier head (Dropout + Linear for 10 classes)

I used **CIFAR-10** because it’s a normal benchmark (60k tiny 32×32 images, 10 classes) and has a real test set. ResNet wants bigger inputs though, so I resize everything to 224×224 and use ImageNet normalization.

I used **Gradio** for the app layer so the demo is just: open the page, click an example or upload something, hit Submit, look at the top predictions.

## Problems I ran into

The first annoying one was image size. CIFAR images are 32×32 and look blurry when you blow them up in the UI, but ResNet expects around 224×224. I just resize/crop in both training and the app so they match.

Training every layer was going to be slow on a laptop, so I froze most of the network and only trained the later block + head. That got me solid accuracy without waiting forever.

Getting the dataset was also a pain. The usual Toronto download was slow / kept having SSL issues on my machine, so I pulled CIFAR-10 from the Hugging Face `uoft-cs/cifar10` mirror instead and wrapped it for the DataLoader.

One more small thing: the Gradio app needs the saved checkpoint. Training writes `checkpoints/resnet18_cifar10_best.pt` and the metrics go in `training_history.json`.

## What I learned

Transfer learning is honestly a big jump compared to training from scratch on this kind of data. Freezing vs unfreezing layers is a real tradeoff — faster training if you freeze more, but maybe less adaptation. Also, once you put the model in an app you notice quickly if train and inference preprocessing don’t match.

## How I’d improve it

I’d probably train longer and unfreeze more layers after the head settles. A confusion matrix / per-class breakdown would help see what it’s bad at. It would also be cool to try my own photos, or host the Gradio app somewhere public so people don’t need to run it locally.

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py --epochs 5
python src/app.py
```

## Results

I trained for 5 epochs on Apple MPS (batch size 64, Adam, lr=1e-3, cosine schedule). Best test accuracy was **93.56%**.

| Epoch | Train loss | Train acc | Test acc |
|------:|-----------:|----------:|---------:|
| 1 | 0.4524 | 84.52% | 88.43% |
| 2 | 0.2940 | 90.06% | 88.20% |
| 3 | 0.2215 | 92.38% | 91.39% |
| 4 | 0.1467 | 95.12% | 92.63% |
| 5 | 0.0898 | 97.00% | **93.56%** |

Same numbers are in `checkpoints/training_history.json`. The demo loads the best checkpoint and shows top-5 probabilities.
