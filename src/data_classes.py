from pydantic import BaseModel as EnrichedPydanticBaseModel
from PIL import Image


class BaseModel(EnrichedPydanticBaseModel):
    # a way fiddle with pydantic's settings https://docs.pydantic.dev/latest/concepts/config/#change-behaviour-globally
    # this was neccessary so pydantic allows for FloatTensor and Image types
    class Config:
        arbitrary_types_allowed = True


class VisualisationData(BaseModel):
    # Taking advantage of pydantic's auto type validation feature
    original_image: Image
    original_class_name: str
    original_class_id: str
    original_confidence_score: float
    perturbation_image: Image
    epsilon: float
    attack_method: str
    iterations: int
    adversarial_image: Image
    adversarial_class_name: str
    adversarial_class_id: str
    adversarial_confidence_score: float
