import pytest
from pathlib import Path

base_path = Path(__file__).parent
test_fixtures_directory = (base_path / "fixtures").resolve()


@pytest.fixture
def create_tensofr_from_image(image_name: str):
    file_path = (test_fixtures_directory / image_name).resolve()

    return df
