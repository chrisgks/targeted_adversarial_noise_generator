import pytest
import torch

from PIL.Image import Image

from src.adversarial_engine_class import AdversarialEngine

test_engine = AdversarialEngine()


@pytest.mark.integrationtest
@pytest.mark.parametrize(
    "expected,image_path,epsilon,attack_method,iterations,target_class,testing_scenario",
    [
        (
            ("669", "mosquito_net"),
            "tests/fixtures/vito1.jpg",
            0.02,
            "fgsm",
            1,
            291,  # imagenet class of the lion
            "fgsm method did not mamange to confuse the model that the input is a lion",
        ),
        (
            ("291", "lion"),
            "tests/fixtures/vito1.jpg",
            0.02,
            "bim",
            15,
            291,
            "ibm methond managed to confuse the model that the input is a lion",
        ),
        (
            ("961", "dough"),
            "tests/fixtures/vito4.jpg",
            0.02,
            "fgsm",
            1,
            291,  # imagenet class of the lion
            "fgsm method did not mamange to confuse the model that the input is a lion",
        ),
        (
            ("291", "lion"),
            "tests/fixtures/vito4.jpg",
            0.02,
            "bim",
            15,
            291,  # imagenet class of the lion
            "fgsm method did not mamange to confuse the model that the input is a lion",
        ),
        (
            ("669", "mosquito_net"),
            "tests/fixtures/junky2.jpeg",
            0.02,
            "fgsm",
            1,
            291,  # imagenet class of the lion
            "fgsm method did not mamange to confuse the model that the input is a lion",
        ),
        (
            ("291", "lion"),
            "tests/fixtures/junky2.jpeg",
            0.02,
            "bim",
            10,
            291,  # imagenet class of the lion
            "fgsm method did not mamange to confuse the model that the input is a lion",
        ),
    ],
)
def test_perform_adversarial_attack(
    expected: int,
    image_path: str,
    epsilon: float,
    attack_method: str,
    iterations: int,
    target_class: int,
    testing_scenario: str,
):
    result = test_engine.perform_adversarial_attack(
        image_path=image_path,
        epsilon=epsilon,
        target_class=target_class,
        attack_method=attack_method,
        iterations=iterations,
    )
    assert result == expected, testing_scenario
