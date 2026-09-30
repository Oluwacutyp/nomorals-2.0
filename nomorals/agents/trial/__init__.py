"""Single-account trial workflow: research → one real signup → encrypted
credential vault → delivery on WhatsApp/Telegram (whichever are live)."""

from .flow import TrialFlow, active_delivery_platforms, new_id
from .vault import TrialVault

__all__ = ["TrialFlow", "TrialVault", "active_delivery_platforms", "new_id"]
