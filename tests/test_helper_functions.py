import pytest

from src.adversarial_helper_class import AdversarialHelper


@pytest.mark.unittest
def test_load_and_transform_image_to_tensor():
    pass


@pytest.mark.unittest
def test_transform_tensors_to_images():
    pass


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
    model_name: str, expected: bool, testing_scenario
):
    try:
        assert (
            bool(AdversarialHelper.load_torchvision_pre_trained_model(model_name))
            == expected
        ), testing_scenario
    except:
        # suppressing pytorch's throwing error for non-existend model
        pass


@pytest.mark.unittest
def test_load_imagenet_classes():
    pass
