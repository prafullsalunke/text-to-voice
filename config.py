from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_id: str = "openbmb/VoxCPM2"
    port: int = 8000
    text_max_length: int = 500
    audio_dir: str = "./audio"
    api_secret: str | None = None  # RSA/EC private key PEM — used to sign JWTs
    api_token: str | None = None   # RSA/EC public key PEM — used to verify JWT signatures

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
