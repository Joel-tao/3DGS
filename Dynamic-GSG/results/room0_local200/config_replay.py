import json
from pathlib import Path
config = json.loads(Path(__file__).with_name("config.resolved.json").read_text())
