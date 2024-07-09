import os

import gdsfactory as gf
from cspdk.si220 import PDK

PDK.activate()

PICS_PATH = os.path.join("/", os.environ.get("GIT_REPO", ""), "pics")
PDK.register_cells_yaml(PICS_PATH, update=True)

yaml_str = """
instances:
  mzi1:
    component: mzi2
  mzi2:
    component: mzi2
connections:
  mzi2,o1: mzi1,o4
"""

c = gf.read.from_yaml(yaml_str)
c.show()
