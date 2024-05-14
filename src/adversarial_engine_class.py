import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from adversarial_helper_class import AdversarialHelper


class AdversarialEngine:
    def __init__(self, model_name="resnet50") -> None:
        self.model = AdversarialHelper.load_torchvision_pre_trained_model(
            model_name=model_name
        )
        self.classes = AdversarialHelper.load_imagenet_classes()

        self.original_prediction = None
        self.adversarial_prediction = None

        self.original_image = None
        self.perturbation_image = None
        self.adversarial_image = None

        self.original_confidence_score = None
        self.adversarial_confidence_score = None

    def _forward_pass(self):
        pass

    def apply_fgsm_method(self, image, epsilon, target_class, iterations=15):

        # for num_steps=1 this function inmplements the Fast Sign Gradient Method (FGSM)
        # for num_steps=n, where n>1, this function implements the Basic Iterative Method (BIM) 
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
            adversarial_image = adversarial_image + epsilon * sign_data_grad / iterations
            adversarial_image = torch.clamp(adversarial_image, 0, 1)

            adversarial_image = adversarial_image.detach()
            adversarial_image.requires_grad = True

        return adversarial_image

    def apply_projected_gradient_descent_method(self):
        raise NotImplementedError

    def visualise_attack(
        self, epsilon, original_image, perturbation_image, adversarial_image, iterations
    ):

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
        axs[0].imshow(original_image)
        axs[0].title.set_text(
            f"Original Prediction \nClass name: {original_class_name}\nConfidence score: {self.original_confidence_score:.3f}%"
        )
        axs[0].axis("off")

        axs[1].imshow(perturbation_image)
        axs[1].title.set_text(f"Perturbation (epsilon={epsilon})\n Iterations: {iterations}")
        axs[1].axis("off")

        axs[2].imshow(adversarial_image)
        axs[2].title.set_text(
            f"Adversarial Prediction \nClass name: {adversarial_class_name}\nConfidence score: {self.adversarial_confidence_score:.3f}%"
        )
        axs[2].axis("off")
        plt.savefig("adversarial_outputs/adversarial_example.jpg")

        plt.show()

    def perform_adversarial_attack(self, image_path, epsilon, target_class, iterations):
        image = AdversarialHelper.load_and_transform_image_to_tensor(
            image_path=image_path
        )
        original_output = self.model(image)
        self.original_probabilities = F.softmax(original_output, dim=1)
        self.original_prediction = original_output.max(1, keepdim=True)[1]
        self.original_confidence_score = self.original_probabilities.max().item() * 100

        perturbed_image = self.apply_fgsm_method(image, epsilon, target_class, iterations)

        perturbation = perturbed_image - image

        adversarial_output = self.model(perturbed_image)
        self.adversarial_probabilities = F.softmax(adversarial_output, dim=1)
        self.adversarial_prediction = adversarial_output.max(1, keepdim=True)[1]
        self.adversarial_confidence_score = (
            self.adversarial_probabilities.max().item() * 100
        )

        original_image, perturbation_image, adversarial_image = (
            AdversarialHelper.transform_tensors_to_images(
                [image, perturbation, perturbed_image]
            )
        )

        self.visualise_attack(
            epsilon=epsilon,
            original_image=original_image,
            perturbation_image=perturbation_image,
            adversarial_image=adversarial_image,
            iterations=iterations
        )
