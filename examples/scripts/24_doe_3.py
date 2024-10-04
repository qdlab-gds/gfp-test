"""Design of Experiment (DOE) with custom add_fiber_array function.

In this case add_fiber_array does not add labels.
"""

import gdsfactory as gf
from gdsfactoryplus.show import show

if __name__ == "__main__":
    c = gf.components.pack_doe_grid(
        gf.components.straight,
        settings={"length": [5, 5]},
        function=gf.routing.add_fiber_array,
    )
    c.show()
    show(c)
