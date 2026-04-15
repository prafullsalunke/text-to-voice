from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_id: str = "openbmb/VoxCPM2"
    port: int = 8000
    text_max_length: int = 500

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
