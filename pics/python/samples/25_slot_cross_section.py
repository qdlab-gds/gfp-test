"""Small demonstration of the slot cross_section utilizing add_center_section=False."""

from __future__ import annotations

from dodesign.show import show
import gdsfactory as gf

if __name__ == "__main__":
    c = gf.components.straight(length=10, width=0.8, cross_section="slot")
    c.show()  # show it in klayout
    show(c)
