"""Algorithmic trading package (Zerodha Kite)."""
# Suppress Twisted signal-handler registration globally
import twisted.internet.base
twisted.internet.base.installSignalHandlers = lambda *args, **kwargs: None
from importlib import metadata
__version__: str = metadata.version(__name__) if metadata else "0.1.0"

from .config  import load_config, CONFIG_PATH            # noqa: F401
from .broker  import KiteWrapper                         # noqa: F401
from .features import add_indicators                     # noqa: F401
from .model    import load_or_train                      # noqa: F401
from .signals  import generate_signal                    # noqa: F401
from .logger   import Order, TradeLogger                 # noqa: F401
from .runner   import run_live                           # noqa: F401