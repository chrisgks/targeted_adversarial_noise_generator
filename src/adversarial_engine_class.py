import logging
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.models as models

import matplotlib.pyplot as plt

from adversarial_helper_class import AdversarialHelper
from data_classes import VisualisationData
import exceptions


logging.basicConfig(level=logging.INFO)


class AdversarialEngine:
    def __init__(self, model_name: str = "resnet50") -> None:
        # Regardless of the attack method, the following attributes will always be needed

        self.model: models = AdversarialHelper.load_torchvision_pre_trained_model(
            model_name=model_name
        )
        logging.info("Model loaded successfully!!")
        self.classes: dict[str, str] = AdversarialHelper.load_imagenet_classes()

        self.original_prediction: torch.FloatTensor | None = None
        self.original_confidence_score: float | None = None
        self.original_class_name: float | None = None
        self.original_class_id: str | None = None

        self.adversarial_prediction: torch.FloatTensor | None = None
        self.adversarial_confidence_score: float | None = None
        self.adversarial_class_name: float | None = None
        self.adversarial_class_id: str | None = None

        self.original_image: Image.Image | None = None
        self.perturbation_image: Image.Image | None = None
        self.adversarial_image: Image.Image | None = None
        logging.info("Adversarial Engine is up and running...")

    def _forward_pass(
        self, image: torch.FloatTensor
    ) -> tuple[torch.FloatTensor, float, float]:
        output: torch.FloatTensor = self.model(image)
        probabilities: torch.FloatTensor = F.softmax(output, dim=1)
        prediction: torch.FloatTensor = output.max(1, keepdim=True)[1]
        confidence_score: float = probabilities.max().item() * 100

        return probabilities, prediction, confidence_score

    def _apply_fgsm_method(
        self,
        image: torch.FloatTensor,
        epsilon: float,
        target_class: int,
        iterations: int = 15,
    ) -> torch.FloatTensor:

        # for iterations=1 this function inmplements the Fast Sign Gradient Method (FGSM)
        # for iterations=n, where n>1, this function implements the Basic Iterative Method (BIM)
        # by applying iterations of the Fast Sign Gradiend Method (FGSM)

        adversarial_image: torch.FloatTensor = image.clone().detach()
        adversarial_image.requires_grad = True

        for _ in range(iterations):
            output: torch.FloatTensor = self.model(adversarial_image)
            loss: torch.FloatTensor = -torch.nn.functional.cross_entropy(
                output, torch.tensor([target_class], device=output.device)
            )
            self.model.zero_grad()
            loss.backward()
            data_grad: torch.FloatTensor = adversarial_image.grad.data

            sign_data_grad: torch.FloatTensor = data_grad.sign()
            adversarial_image = (
                adversarial_image + epsilon * sign_data_grad / iterations
            )
            adversarial_image = torch.clamp(adversarial_image, 0, 1)

            adversarial_image = adversarial_image.detach()
            adversarial_image.requires_grad = True

        return adversarial_image

    def _apply_pgd_method(self, image: torch.FloatTensor, target_class: int):
        raise NotImplementedError

    def visualise_attack(
        self,
        epsilon: float,
        iterations: int,
        attack_method: str,
        save_visual_on_disc: bool = True,
        show_visualisation: bool = True,
    ) -> None:
        logging.info("Visualising attack...")

        self.original_class_id = str(self.original_prediction.item())
        self.original_class_name = self.classes[self.original_class_id]

        self.adversarial_class_id = str(self.adversarial_prediction.item())
        self.adversarial_class_name = self.classes[self.adversarial_class_id]

        visual_data = VisualisationData(
            original_image=self.original_image,
            original_class_name=self.original_class_name,
            original_class_id=self.original_class_id,
            original_confidence_score=self.original_confidence_score,
            perturbation_image=self.perturbation_image,
            epsilon=epsilon,
            attack_method=attack_method,
            iterations=iterations,
            adversarial_image=self.adversarial_image,
            adversarial_class_name=self.adversarial_class_name,
            adversarial_class_id=self.adversarial_class_id,
            adversarial_confidence_score=self.adversarial_confidence_score,
        )

        AdversarialHelper.build_visuals(visual_data)

        if save_visual_on_disc:
            attack_id = f"example_attacked_by_{attack_method}_epsilon_{str(epsilon).replace(".", "")}_iterations_{iterations}"
            output_path = f"src/adversarial_outputs/{attack_id}.jpg"
            plt.savefig(output_path)
            logging.info("Attack visualisation saved @ " + output_path)

        if show_visualisation:
            plt.show()

    def perform_adversarial_attack(
        self,
        image_path: str,
        epsilon: float,
        target_class: int,
        attack_method: str,
        iterations: int,
    ) -> None:
        logging.info("Performing adversarial attack...")
        image = AdversarialHelper.load_and_transform_image_to_tensor(
            image_path=image_path
        )

        (
            self.original_probabilities,
            self.original_prediction,
            self.original_confidence_score,
        ) = self._forward_pass(image)

        match attack_method:
            case "fgsm":
                iterations = 1
                perturbed_image = self._apply_fgsm_method(
                    image, epsilon, target_class, iterations=iterations
                )
            case "bim":
                perturbed_image = self._apply_fgsm_method(image, epsilon, target_class)
            case "pgdm":
                perturbed_image = self._apply_pgd_method(image, target_class)
            case other:
                raise exceptions.AdversarialMethodNotSupportedError(attack_method)

        perturbation = perturbed_image - image

        (
            self.adversarial_probabilities,
            self.adversarial_prediction,
            self.adversarial_confidence_score,
        ) = self._forward_pass(perturbed_image)

        self.original_image, self.perturbation_image, self.adversarial_image = (
            AdversarialHelper.transform_tensors_to_images(
                [image, perturbation, perturbed_image]
            )
        )

        self.visualise_attack(
            epsilon=epsilon, attack_method=attack_method, iterations=iterations
        )

        return self.adversarial_class_id, self.adversarial_class_name
