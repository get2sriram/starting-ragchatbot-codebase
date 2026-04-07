"""Test configuration loading"""

import os
from backend.config import Config


def test_config_loading():
    """Test that config loads correctly"""
    config = Config()
    assert config.EMBEDDING_MODEL == "all-MiniLM-L6-v2"
    assert config.CHUNK_SIZE == 800
    assert config.CHUNK_OVERLAP == 100


if __name__ == "__main__":
    test_config_loading()
    print("Config test passed!")
