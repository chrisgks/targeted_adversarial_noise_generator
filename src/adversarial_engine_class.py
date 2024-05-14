import logging

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from adversarial_helper_class import AdversarialHelper
import exceptions


logging.basicConfig(level=logging.INFO)


class AdversarialEngine:
    def __init__(self, model_name="resnet50") -> None:
        # Regardless of the attack method, the following attributes will always be needed

        self.model = AdversarialHelper.load_torchvision_pre_trained_model(
            model_name=model_name
        )
        self.classes = AdversarialHelper.load_imagenet_classes()

        self.original_prediction = None
        self.adversarial_prediction = None
        self.original_confidence_score = None
        self.adversarial_confidence_score = None

        self.original_image = None
        self.perturbation_image = None
        self.adversarial_image = None
        logging.info("Adversarial Engine is up and running...")

    def _forward_pass(self, image):
        output = self.model(image)
        probabilities = F.softmax(output, dim=1)
        prediction = output.max(1, keepdim=True)[1]
        confidence_score = probabilities.max().item() * 100

        return probabilities, prediction, confidence_score

    def _apply_fgsm_method(self, image, epsilon, target_class, iterations=15):

        # for iterations=1 this function inmplements the Fast Sign Gradient Method (FGSM)
        # for iterations=n, where n>1, this function implements the Basic Iterative Method (BIM)
        # by applying iterations of the Fast Sign Gradiend Method (FGSM)

        adversarial_image = image.clone().detach()
        adversarial_image.requires_grad = True

        for _ in range(iterations):
            output = self.model(adversarial_image)
            loss = -torch.nn.functional.cross_entropy(
                output, torch.tensor([target_class], device=output.device)
            )
            self.model.zero_grad()
            loss.backward()
            data_grad = adversarial_image.grad.data

            sign_data_grad = data_grad.sign()
            adversarial_image = (
                adversarial_image + epsilon * sign_data_grad / iterations
            )
            adversarial_image = torch.clamp(adversarial_image, 0, 1)

            adversarial_image = adversarial_image.detach()
            adversarial_image.requires_grad = True

        return adversarial_image

    def _apply_pgd_method(self, image, target_class):
        raise NotImplementedError

    def visualise_attack(self, epsilon, iterations, attack_method):
        logging.info("Visualising attack...")

        original_class_name = (
            self.classes[self.original_prediction.item()]
            .split(",")[0]
            .replace("'", "")
            .strip()
        )
        adversarial_class_name = (
            self.classes[self.adversarial_prediction.item()]
            .split(",")[0]
            .replace("'", "")
            .strip()
        )

        fig, axs = plt.subplots(1, 3, figsize=(15, 8))
        axs[0].imshow(self.original_image)
        axs[0].title.set_text(
            f"Original Prediction \nClass name: {original_class_name}\nConfidence score: {self.original_confidence_score:.3f}%"
        )
        axs[0].axis("off")

        axs[1].imshow(self.perturbation_image)
        axs[1].title.set_text(
            f"Perturbation (epsilon={epsilon})\nAttack method: {attack_method}\nIterations: {iterations}"
        )
        axs[1].axis("off")

        axs[2].imshow(self.adversarial_image)
        axs[2].title.set_text(
            f"Adversarial Prediction \nClass name: {adversarial_class_name}\nConfidence score: {self.adversarial_confidence_score:.3f}%"
        )
        axs[2].axis("off")
        plt.savefig("adversarial_outputs/adversarial_example.jpg")

        plt.show()

    def perform_adversarial_attack(
        self, image_path, epsilon, target_class, attack_method, iterations
    ):
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
                perturbed_image = self._apply_pgd_mathod(image, target_class)
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
