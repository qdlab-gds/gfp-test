import os

import gdsfactory as gf
from cspdk.si220 import PDK
from dodesign.show import show

PDK.activate()

PICS_PATH = os.path.join("/", os.environ.get("GIT_REPO", ""), "pics")
PDK.register_cells_yaml(PICS_PATH, update=True)

yaml_str = """
instances:
  mzi1:
    component: mzi2
  mzi2:
    component: mzi2
routes:
  bundle:
    links:
      mzi1,o3: mzi2,o1
placements:
  mzi2:
    x: 200
    y: 200
"""

c = gf.read.from_yaml(yaml_str)

for inst in c.insts:
    print(inst.name, inst.dx, inst.dy)


show(c.get_netlist())
show(c)
