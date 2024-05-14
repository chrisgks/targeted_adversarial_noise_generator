import logging
from PIL import Image

import torch
import torchvision.transforms as transforms
import torchvision.models as models

import matplotlib.pyplot as plt

from data_classes import VisualisationData


class AdversarialHelper:
    # this class acts as a collection of the supporting functions to the main engine

    @staticmethod
    def load_torchvision_pre_trained_model(model_name: str) -> models:
        logging.info("Loading torchvision model...")
        return getattr(models, model_name)(pretrained=True).eval()

    @staticmethod
    def load_and_transform_image_to_tensor(
        image_path: str, size: tuple[int, int] = (224, 224)
    ) -> torch.FloatTensor:
        logging.info("Loading and trasforming image...")
        transform = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.ToTensor(),
            ]
        )
        image = Image.open(image_path)
        return transform(image).unsqueeze(0)

    @staticmethod
    def transform_tensors_to_images(
        image_tensors: list[torch.FloatTensor],
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        images: tuple = ()
        for tensor in image_tensors:
            images += (transforms.ToPILImage()(tensor.squeeze(0)),)
        return images

    @staticmethod
    def load_imagenet_classes() -> list[str]:
        with open("../notebooks/data/imagenet_classes.txt") as f:
            classes = [line.strip() for line in f.readlines()]
            return classes

    def build_visuals(visual_data: VisualisationData):

        fig, axs = plt.subplots(1, 3, figsize=(15, 8))
        axs[0].imshow(visual_data.original_image)
        axs[0].title.set_text(
            f"Original Prediction \nClass name: {visual_data.original_class_name}\nConfidence score: {visual_data.original_confidence_score:.3f}%"
        )
        axs[0].axis("off")

        axs[1].imshow(visual_data.perturbation_image)
        axs[1].title.set_text(
            f"Perturbation (epsilon={visual_data.epsilon})\nAttack method: {visual_data.attack_method}\nIterations: {visual_data.iterations}"
        )
        axs[1].axis("off")

        axs[2].imshow(visual_data.adversarial_image)
        axs[2].title.set_text(
            f"Adversarial Prediction \nClass name: {visual_data.adversarial_class_name}\nConfidence score: {visual_data.adversarial_confidence_score:.3f}%"
        )
        axs[2].axis("off")
