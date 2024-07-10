import os

import gdsfactory as gf
import jax.numpy as jnp
import matplotlib.pyplot as plt
import sax
from cspdk.si220 import PDK
from dosax.circuit import circuit_plot

PDK.activate()

PICS_PATH = os.path.join("/", os.environ.get("GIT_REPO", ""), "pics")
PDK.register_cells_yaml(PICS_PATH, update=True)

wl = jnp.linspace(1.5, 1.6, 1000)
c = gf.get_component("lattice")
netlist = sax.netlist(c.get_netlist(recursive=True), with_placements=False)
circuit = circuit_plot(
    netlist,
    pdk="cspdk.si220",
    op="dB",
    which="html",
    port_in="o1",
    host="https://dosax.docode.doplaydo.com",
)
with open("plot.html", "w") as file:
    file.write(circuit(wl=wl))

circuit, _ = sax.circuit(netlist, models=PDK.models)
result = sax.sdict(circuit(wl=wl))
plt.plot(wl, 10 * jnp.log10(abs(result["o1", "o1"]) ** 2))
plt.plot(wl, 10 * jnp.log10(abs(result["o1", "o2"]) ** 2))
plt.plot(wl, 10 * jnp.log10(abs(result["o1", "o3"]) ** 2))
plt.show()
