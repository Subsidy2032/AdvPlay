import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from advplay.utils.load_classes import load_required_classes

load_required_classes()
