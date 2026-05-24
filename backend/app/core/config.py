from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DEVICE: str = "cpu"
    MODEL_CHECKPOINT_DIR: str = "./training/checkpoints"
    MAX_IMAGE_SIZE_MB: int = 15
    ELA_QUALITY: int = 90
    ELA_SCALE: int = 10
    NOISE_SIGMA: float = 2.0
    LOG_LEVEL: str = "INFO"

    # Load environment variables with fallback to a .env file
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
