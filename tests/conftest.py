import os
import sys

# Ensure src/ is on the path when running pytest directly (without pip install -e .)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
