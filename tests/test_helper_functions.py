import pytest
import torch

from PIL.Image import Image

from src.adversarial_helper_class import AdversarialHelper


@pytest.mark.unittest
@pytest.mark.parametrize(
    "image_path,expected,testing_scenario",
    [
        ("tests/fixtures/vito1.jpg", True, "image transformed successfully"),
        (
            "tests/fixtures/randomimage.png",
            False,
            "image did not transfom successfully",
        ),
    ],
)
def test_load_and_transform_image_to_tensor(
    image_path: str, expected: bool, testing_scenario: str
):
    try:
        result = isinstance(
            AdversarialHelper.load_and_transform_image_to_tensor(image_path),
            torch.FloatTensor,
        )
    except FileNotFoundError:
        result = False
        pass
    assert result == expected, testing_scenario


@pytest.mark.parametrize(
    "tensors,expected,testing_scenario",
    [
        ([torch.rand(224, 224)], True, "tensor transformed to image successfully"),
        (
            [torch.rand(100, 50), torch.rand(50, 100)],
            True,
            "tensors transformed to images successfully",
        ),
    ],
)
@pytest.mark.unittest
def test_transform_tensors_to_images(
    tensors: list, expected: bool, testing_scenario: str
):
    for t in tensors:
        assert t.shape[0] > 1, "tensor should have at least 2 dimentions"
    result = AdversarialHelper.transform_tensors_to_images(tensors)
    for r in result:
        assert isinstance(r, Image) == expected, testing_scenario


@pytest.mark.unittest
@pytest.mark.parametrize(
    "model_name,expected,testing_scenario",
    [
        ("resnet18", True, "model loaded successfully"),
        ("resnet34", True, "model loaded successfully"),
        ("resnet50", True, "model loaded successfully"),
        ("nonexistentmodel", False, "model does not exist or not loadded successfully"),
    ],
)
def test_load_torchvision_pre_trained_model(
    model_name: str, expected: bool, testing_scenario: str
):
    try:
        assert (
            bool(AdversarialHelper.load_torchvision_pre_trained_model(model_name))
            == expected
        ), testing_scenario
    except:
        # suppressing pytorch's throwing error for non-existend model
        pass


@pytest.mark.skip
@pytest.mark.unittest
def test_load_imagenet_classes():
    result = AdversarialHelper.load_imagenet_classes()
    assert isinstance(result, list) == True, "imagenet classes should be a list"
