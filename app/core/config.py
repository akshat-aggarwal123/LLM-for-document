from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    hf_token: str = os.getenv("HF_TOKEN")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    llama_model: str = os.getenv("LLAMA_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
    chroma_persist_dir: Path = Path(os.getenv("CHROMA_PERSIST_DIR", "./data/vector_store"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "512"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.3"))
    device: str = "cuda"                            # torch will fallback to cpu if unavailable