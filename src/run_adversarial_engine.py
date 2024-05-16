import argparse
import logging

from adversarial_engine_class import AdversarialEngine
from exceptions import ImageTypeNotSupportedError


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Adversarial Engine")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument(
        "target_class_id", type=int, help="Target class for misclassification"
    )
    parser.add_argument(
        "attack_method",
        type=str,
        nargs="?",
        const="bim",
        default="bim",
        help="Target class for misclassification",
    )
    args = parser.parse_args()

    if args.image_path.endswith(".png"):
        logging.info("Please use either .jpeg or .jpg images.")
        raise ImageTypeNotSupportedError

    attack_engine = AdversarialEngine()

    attack_engine.perform_adversarial_attack(
        image_path=args.image_path,
        epsilon=0.02,
        target_class=args.target_class_id,
        attack_method=args.attack_method,
        iterations=10,
    )
