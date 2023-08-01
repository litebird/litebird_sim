# -*- encoding: utf-8 -*-

"""
Modify PYTHONPATH so that it is possible to import `litebird_sim`
"""

from pathlib import Path
import sys

# Path(__file__): "litebird_sim/notebooks/fixpythonpath.py"
# Path(__file__).parent: "litebird_sim/notebooks/"
# Path(__file__): "litebird_sim/"
curpath = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(curpath))
