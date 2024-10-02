"""You can define a function to add pins."""

import gdsfactory as gf
from gdsfactoryplus.show import show

if __name__ == "__main__":
    c = gf.components.straight()
    c.draw_ports()
    c.show()
    show(c)
