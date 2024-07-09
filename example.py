import gdsfactory as gf
from cspdk.si220 import PDK

PDK.activate()

mzi = gf.get_component("mzi_sc")
mzi.show()