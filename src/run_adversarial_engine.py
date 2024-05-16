from adversarial_engine_class import AdversarialEngine

attack_engine = AdversarialEngine()

img_path = "/home/chris/Desktop/targeted_adversarial_perturbations/notebooks/data/example_images/vito1.jpg"
target_class = 291
epsilon = 0.02

iterations = 10
attack_method = "bim"
print(
    attack_engine.perform_adversarial_attack(
        image_path=img_path,
        epsilon=epsilon,
        target_class=target_class,
        attack_method=attack_method,
        iterations=iterations,
    )
)
