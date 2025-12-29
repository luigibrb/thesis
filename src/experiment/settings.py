from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    base_path: Path = Path(__file__).resolve().parent.parent.parent
    data_path: Path = base_path / "data" / "bodmas"
    results_path: Path = base_path / "src" / "experiment" / "results"
    plot_experiment_name: str

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
