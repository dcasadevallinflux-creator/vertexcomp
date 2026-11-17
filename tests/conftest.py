import os
import sys

# Use the in-tree package (developer build) unless testing an installed wheel.
if not os.environ.get("VC_INSTALLED"):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
os.environ.setdefault("PYSIDE6_OPTION_LAZY", "0")
