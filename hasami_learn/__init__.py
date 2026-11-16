"""Optional learning package for Hasami Lab.

Everything here depends on torch and numpy. The core ``hasami`` package never
imports this package; the GUI and arena reach learned agents only through the
``AgentRegistry`` plugin contract (``hasami_learn.inference:create_agent``).
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
