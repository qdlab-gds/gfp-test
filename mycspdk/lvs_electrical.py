"""LVS demo."""

from __future__ import annotations

import gdsfactory as gf
from cspdk.si220 import cells


@gf.cell
def pads_correct(pad=cells.pad, cross_section="metal_routing") -> gf.Component:
    """Returns 2 pads connected with metal wires."""
    c = gf.Component()

    xs = gf.get_cross_section(cross_section)
    layer = gf.get_layer(xs.layer)

    pad = gf.get_component(pad)
    tl = c << pad
    bl = c << pad

    tr = c << pad
    br = c << pad

    tl.dmove((0, 300))
    br.dmove((500, 0))
    tr.dmove((500, 500))

    c.add_label("tl", position=tl.dcenter, layer=layer)
    c.add_label("tr", position=tr.dcenter, layer=layer)
    c.add_label("br", position=br.dcenter, layer=layer)
    c.add_label("bl", position=bl.dcenter, layer=layer)

    ports1 = [bl.ports["e3"], tl.ports["e3"]]
    ports2 = [br.ports["e1"], tr.ports["e1"]]
    gf.routing.route_bundle_electrical(c, ports1, ports2, cross_section=cross_section)
    return c


@gf.cell
def pads_shorted(pad=cells.pad, cross_section="metal_routing") -> gf.Component:
    """Returns 2 pads connected with metal wires."""
    c = gf.Component()
    pad = gf.get_component(pad)
    xs = gf.get_cross_section(cross_section)
    layer = gf.get_layer(xs.layer)

    tl = c << pad
    bl = c << pad

    tr = c << pad
    br = c << pad

    tl.dmove((0, 300))
    br.dmove((500, 0))
    tr.dmove((500, 500))

    c.add_label("tl", position=tl.dcenter, layer=layer)
    c.add_label("tr", position=tr.dcenter, layer=layer)
    c.add_label("br", position=br.dcenter, layer=layer)
    c.add_label("bl", position=bl.dcenter, layer=layer)

    ports1 = [bl.ports["e3"], tl.ports["e3"]]
    ports2 = [br.ports["e1"], tr.ports["e1"]]
    gf.routing.route_bundle_electrical(c, ports1, ports2, cross_section=cross_section)

    gf.routing.route_single_electrical(
        c, bl.ports["e2"], tl.ports["e4"], cross_section=cross_section
    )
    return c


@gf.cell
def demo_astar_optical() -> gf.Component:
    c = gf.Component()
    cross_section_name = "xs_sc"
    port_prefix = "o"
    bend = gf.components.bend_euler

    cross_section = gf.get_cross_section(cross_section_name, radius=5)
    w = gf.components.straight(cross_section=cross_section)
    left = c << w
    right = c << w
    right.rotate(90)  # type: ignore[arg-type]
    right.move((168, 63))

    obstacle = gf.components.rectangle(size=(250, 3), layer="PAD")
    obstacle1 = c << obstacle
    obstacle2 = c << obstacle
    obstacle3 = c << obstacle
    obstacle4 = c << obstacle
    obstacle4.rotate(90)  # type: ignore[arg-type]
    obstacle1.ymin = 50
    obstacle1.xmin = -10
    obstacle2.xmin = 35
    obstacle3.ymin = 42
    obstacle3.xmin = 72.23  # type: ignore
    obstacle4.xmin = 200
    obstacle4.ymin = 55
    port1 = left.ports[f"{port_prefix}1"]
    port2 = right.ports[f"{port_prefix}2"]

    gf.routing.route_astar(
        component=c,
        port1=port1,
        port2=port2,
        cross_section=cross_section,
        resolution=15,
        distance=12,
        avoid_layers=("PAD",),
        bend=bend,
    )
    return c


@gf.cell
def demo_astar_electrical() -> gf.Component:
    c = gf.Component()
    cross_section_name = "metal_routing"
    port_prefix = "e"
    bend = gf.components.wire_corner

    cross_section = gf.get_cross_section(cross_section_name)
    w = gf.components.straight(cross_section=cross_section)
    left = c << w
    right = c << w
    right.rotate(90)  # type: ignore[arg-type]
    right.move((168, 63))

    obstacle = gf.components.rectangle(size=(250, 3), layer="PAD")
    obstacle1 = c << obstacle
    obstacle2 = c << obstacle
    obstacle3 = c << obstacle
    obstacle4 = c << obstacle
    obstacle4.rotate(90)  # type: ignore[arg-type]
    obstacle1.ymin = 50
    obstacle1.xmin = -10
    obstacle2.xmin = 35
    obstacle3.ymin = 42
    obstacle3.xmin = 72.23  # type: ignore
    obstacle4.xmin = 200
    obstacle4.ymin = 55
    port1 = left.ports[f"{port_prefix}1"]
    port2 = right.ports[f"{port_prefix}2"]

    gf.routing.route_astar(
        component=c,
        port1=port1,
        port2=port2,
        cross_section=cross_section,
        resolution=10,
        distance=12,
        avoid_layers=("PAD",),
        bend=bend,
    )
    return c


if __name__ == "__main__":
    c = demo_astar_electrical()
    c.show()
