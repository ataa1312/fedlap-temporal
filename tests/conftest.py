import sys, os
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent   # the fedlap root
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

import pytest
from config.config import get_default_config

@pytest.fixture
def config():
    import src
    cfg = get_default_config()
    src.config._registry.clear()
    src.config._registry.update(cfg._registry)
    return src.config
