"""Training-curve and Grad-CAM visualization utilities."""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from .config import IMAGENET_MEAN, IMAGENET_STD, CONFIG

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate(self, fused_input, class_idx):
        self.model.zero_grad()
        output = self.model(fused_input)
        output[0, class_idx].backward()
        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.sum(weights[:, None, None] * activations, axis=0)
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (fused_input.shape[3], fused_input.shape[2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def plot_training_history(history):
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.plot(history["seg_train_loss"], label="Train Loss")
    plt.title("Stage A: a-Net Segmentation Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history["cls_train_loss"], label="Train Loss")
    plt.title("Stage B: ResNet50 Classification Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history["seg_val_dsc"], label="Val DSC (%)")
    plt.plot(history["cls_val_acc"], label="Val ACC (%)")
    plt.title("Validation Metric Progression")
    plt.xlabel("Epoch"); plt.legend()
    plt.tight_layout()
    return plt.gcf()
