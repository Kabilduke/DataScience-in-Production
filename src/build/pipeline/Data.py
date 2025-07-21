import pandas as pd

from loguru import logger

#from config.config import settings
from config.config import engine
from database.db_model import classification
from sqlalchemy import select


""" 
This function is commented out because it is not used in the current implementation.

def data(path = settings.data_file_name): #dataset_path
    logger.info('Loading data from:', path)
    return pd.read_csv(path)
"""


def load_db() -> pd.DataFrame:
    logger.info('Loading data from database...')
    query = select(classification)
    return pd.read_sql(query, engine)

