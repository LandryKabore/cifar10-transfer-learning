"""Gradio demo for the fine-tuned CIFAR-10 ResNet18 classifier."""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from torchvision import transforms

from model import CIFAR10_CLASSES, build_model, get_device

ROOT = Path(__file__).resolve().parents[1]
CKPT_PATH = ROOT / "checkpoints" / "resnet18_cifar10_best.pt"

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

preprocess = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)


def load_model():
    device = get_device()
    model = build_model(num_classes=len(CIFAR10_CLASSES), pretrained=False)
    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {CKPT_PATH}. Run: python src/train.py"
        )
    checkpoint = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    classes = checkpoint.get("classes", CIFAR10_CLASSES)
    test_acc = checkpoint.get("test_acc", None)
    return model, device, classes, test_acc


MODEL, DEVICE, CLASSES, TEST_ACC = load_model()


@torch.no_grad()
def predict(image: Image.Image) -> dict[str, float]:
    if image is None:
        return {}
    image = image.convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    logits = MODEL(tensor)
    probs = torch.softmax(logits, dim=1)[0]
    ranking = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
    return dict(sorted(ranking.items(), key=lambda x: x[1], reverse=True))


def main() -> None:
    subtitle = "Fine-tuned ResNet18 (PyTorch transfer learning) on CIFAR-10"
    if TEST_ACC is not None:
        subtitle += f" | held-out test accuracy: {TEST_ACC:.1f}%"

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Upload an image"),
        outputs=gr.Label(num_top_classes=5, label="Predictions"),
        title="CIFAR-10 Transfer Learning Classifier",
        description=subtitle,
        examples=None,
        allow_flagging="never",
    )
    demo.launch()


if __name__ == "__main__":
    main()
