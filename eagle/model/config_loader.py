# config_loader.py
import yaml
from pathlib import Path
import os
from transformers import AutoTokenizer

class Config:
    """Configuration loader for StarSD system.
    
    Note: SAC/RL related features have been removed. This class now only
    provides basic configuration for the distributed inference system.
    """

    
    def __init__(self):
        with open(Path(__file__).parent.parent.parent / "config.yaml") as f:
            self.config = yaml.safe_load(f) or {}
        
        self.device_base = self.config["device_base"]
        self.device_draft = self.config["device_draft"]
        
        self.tokenizer = None
        self._init_tokenizer()

    def _init_tokenizer(self):
        try:
            model_path = self.base_model_path
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            print("✅ Global tokenizer initialized successfully")
        except Exception as e:
            print(f"⚠️ Failed to initialize global tokenizer: {e}")
            self.tokenizer = None

    @property
    def communication_port(self) -> int:
        return self.config["communication_port"]
    
    @property
    def draft_model_path(self) -> str:
        return self.config["draft_model_path"]
    
    @property
    def base_model_path(self) -> str:
        return self.config["base_model_path"]



config = Config()