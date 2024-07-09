from cspdk.si220 import PDK
import os
import gdsfactory as gf
import sax
import jax.numpy as jnp
import matplotlib.pyplot as plt

PDK.activate()

PICS_PATH = os.path.join("/", os.environ.get("GIT_REPO", ""), "pics")
PDK.register_cells_yaml(PICS_PATH, update=True)

c = gf.get_component("mzi2")
netlist = sax.netlist(c.get_netlist(recursive=True))

circuit, _ = sax.circuit(
    netlist,
    models={
        "bend_euler": PDK.models["bend_euler"],
        "coupler": PDK.models["coupler"],
        "straight": PDK.models["straight"],
    },
)

wl = jnp.linspace(1.5, 1.6, 100)
result = sax.sdict(circuit(wl=wl))
print(list(result))

p = plt.plot(wl, result["o1", "o3"])
q = plt.plot(wl, result["o1", "o4"])
plt.show()
