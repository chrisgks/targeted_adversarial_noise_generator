import logging
from PIL import Image

import torchvision.transforms as transforms
import torchvision.models as models
from torch.tensor import Tensor


class AdversarialHelper:

    @staticmethod
    def load_torchvision_pre_trained_model(model_name: str) -> models:
        logging.info("Loading torchvision model...")
        return getattr(models, model_name)(pretrained=True).eval()

    @staticmethod
    def load_and_transform_image_to_tensor(
        image_path: str, size: tuple[int, int] = (224, 224)
    ) -> Tensor:
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
        image_tensors: list[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        images: tuple = ()
        for tensor in image_tensors:
            images += (transforms.ToPILImage()(tensor.squeeze(0)),)
        return images

    @staticmethod
    def load_imagenet_classes() -> list[str]:
        with open("../notebooks/data/imagenet_classes.txt") as f:
            classes = [line.strip() for line in f.readlines()]
            return classes
