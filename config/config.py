from pydantic import BaseModel
from pydantic_settings import BaseSettings
import os

from dotenv import load_dotenv

load_dotenv()

class ApiConfig(BaseModel):
    api_key: str
    server_url: str

class Settings(BaseSettings):
    api: ApiConfig = ApiConfig(api_key=os.getenv("API_KEY"), server_url=os.getenv("MAIN_SERVER_URL"))

settings = Settings()