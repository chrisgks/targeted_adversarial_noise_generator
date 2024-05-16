import pytest
from pathlib import Path

base_path = Path(__file__).parent
test_fixtures_directory = (base_path / "fixtures").resolve()


def create_image_tensor():
    pass


def write_image_tensor_to_disc():
    pass


@pytest.fixture
def load_image_tensor_from_disc(image_name: str):
    file_path = (test_fixtures_directory / image_name).resolve()
    pass
