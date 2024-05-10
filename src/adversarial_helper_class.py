from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models


class AdversarialHelper:

    def load_torchvision_pre_trained_model(model_name):
        return getattr(models, model_name)(pretrained=True).eval()

    def load_and_transform_image_to_tensor(image_path, size=(224, 224)):
        transform = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.ToTensor(),
            ]
        )
        image = Image.open(image_path)
        return transform(image).unsqueeze(0)

    def transform_tensors_to_images(image_tensors):
        images = ()
        for tensor in image_tensors:
            images += (transforms.ToPILImage()(tensor.squeeze(0)),)
        return images

    def load_imagenet_classes():
        with open("../notebooks/data/imagenet_classes.txt") as f:
            classes = [line.strip() for line in f.readlines()]
            return classes
