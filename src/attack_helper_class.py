from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models


class AttackHelper:

    def load_torchvision_pre_trained_model(model_name="resnet18"):
        return getattr(models, model_name)(pretrained=True).eval()

    def load_and_transform_image_for_model_input(image_path, size=(224, 224)):
        transform = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.ToTensor(),
            ]
        )
        image = Image.open(image_path)
        return transform(image).unsqueeze(0)

    def transform_tensor_to_image(image_tensor):
        return transforms.ToPILImage()(image_tensor.squeeze(0))

    def load_imagenet_classes():
        with open("../notebooks/data/imagenet/imagenet_classes.txt") as f:
            classes = [line.strip() for line in f.readlines()]
            return classes
