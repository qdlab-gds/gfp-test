import os

import gdsfactory as gf
import jax.numpy as jnp
import matplotlib.pyplot as plt
import sax
from cspdk.si220 import PDK

PDK.activate()

PICS_PATH = os.path.join("/", os.environ.get("GIT_REPO", ""), "pics")
PDK.register_cells_yaml(PICS_PATH, update=True)

c = gf.get_component("lattice")
netlist = sax.netlist(c.get_netlist(recursive=True))
print(
    {
        k: {k: c.component for k, c in v.instances.items()}
        for k, v in netlist.root.items()
    }
)

circuit, _ = sax.circuit(
    netlist,
    models=PDK.models,
)

wl = jnp.linspace(1.5, 1.6, 1000)
result = sax.sdict(circuit(wl=wl))
print(list(result))


plt.plot(wl, 10 * jnp.log10(abs(result["o1", "o2"]) ** 2))
plt.plot(wl, 10 * jnp.log10(abs(result["o1", "o3"]) ** 2))
plt.show()
