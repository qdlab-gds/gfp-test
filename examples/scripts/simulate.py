import os

import altair as alt
import gdsfactory as gf
import jax.numpy as jnp
import pandas as pd
import sax
from cspdk.si220 import PDK
from gdsfactoryplus.show import show
from gdsfactoryplus.simulate import circuit_plot

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

# plt.plot(wl, 10 * jnp.log10(abs(result["o1", "o1"]) ** 2))
# plt.plot(wl, 10 * jnp.log10(abs(result["o1", "o2"]) ** 2))
# plt.plot(wl, 10 * jnp.log10(abs(result["o1", "o3"]) ** 2))
# show()

# from bokeh.plotting import figure
# p = figure(title="Bokeh Plot", x_axis_label='Wavelength (um)', y_axis_label='dB')
# p.line(wl, 10 * jnp.log10(abs(result["o1", "o1"]) ** 2), color="red")
# p.line(wl, 10 * jnp.log10(abs(result["o1", "o2"]) ** 2), color="green")
# p.line(wl, 10 * jnp.log10(abs(result["o1", "o3"]) ** 2), color="blue")
# show(p)


dfs: list[pd.DataFrame] = []
for (p, q), arr in result.items():
    if p == "o1" and q != "o1":
        dB = 20 * jnp.log10(abs(arr))
        dfs.append(pd.DataFrame({"wl": wl, "dB": dB, "port_out": q}))
df = pd.concat(dfs, axis=0)
sel = alt.selection_point(fields=["port_out"], bind="legend", toggle="true", empty=True)
c = (
    alt.Chart(df)
    .mark_line()
    .encode(
        x=alt.X("wl"),
        y="dB",
        color="port_out",
        opacity=alt.condition(sel, alt.value(0.9), alt.value(0.1)),  # type: ignore
    )
    .add_params(sel)
    .properties(width="container", height=250)
    .interactive()
)

show(c)
