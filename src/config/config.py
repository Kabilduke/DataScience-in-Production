from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import FilePath, DirectoryPath
from sqlalchemy import create_engine

from loguru import logger


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file = 'config/.env', 
        env_file_encoding = 'utf-8'
    )
    #data_file_name: FilePath 
    model_path: DirectoryPath 
    model_name: str
    db_host: str
    table_name: str


# pyright: reportCallIssue=false
settings = Settings()

logger.add(
    'logs/app.log', 
    rotation = '1 day', 
    retention = '2 days',
    compression = 'zip'
    )

engine = create_engine(settings.db_host)

