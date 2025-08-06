import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from advplay.utils.load_template_builders import import_all_template_classes

import_all_template_classes()
