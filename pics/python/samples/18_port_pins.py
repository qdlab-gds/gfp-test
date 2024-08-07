"""You can define a function to add pins."""

from dodesign.show import show

import gdsfactory as gf

if __name__ == "__main__":
    c = gf.components.straight()
    c.draw_ports()
    c.show()
    show(c)
