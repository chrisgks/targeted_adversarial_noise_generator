conda create -n adversarial_engine python=3.12 -y
conda activate adversarial_engine
pip install -r requirements.txt
pytest
python src/run_adversarial_engine.py notebooks/data/example_images/vito1.jpg 291
conda env remove -n adversarial_engine
