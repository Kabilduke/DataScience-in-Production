import numpy as np

from keras import models
from pathlib import Path
from loguru import logger

from build.pipeline.Model import build_model
from config.config import settings


class UseModel():
    def __init__(self) -> None:
        self.model = None

    def load(self) -> None:
        model_path = Path(f'{settings.model_path}/{settings.model_name}')

        if not model_path.exists():
            logger.warning(f'Model file not found at {model_path}. Building model...')
            build_model()

        self.model = models.load_model(f'{settings.model_path}/{settings.model_name}')

    def predict(self, input_array):
        if self.model is None:
            raise ValueError('Model is not Loaded!')
        
        input_array = np.array(input_array).reshape(1, -1)  # into tensor
        return self.model.predict([input_array])

