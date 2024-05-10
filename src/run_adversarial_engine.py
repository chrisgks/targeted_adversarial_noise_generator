from adversarial_engine_class import AdversarialEngine

attack_engine = AdversarialEngine()

img_path = "/home/chris/Desktop/targeted_adversarial_perturbations/notebooks/data/example_images/junky2.jpeg"
target_class = 291
epsilon = 0.02
attack_engine.perform_adversarial_attack(image_path=img_path, epsilon=epsilon, target_class=target_class)
