"""Challenge pack modules — imported for their ``register_topic_pack`` side effects.

Packs: ``chall_systems`` (systems/distributed/concurrency/databases/
networking/compilers/runtimes/packaging), ``chall_web`` (web/api/auth/
realtime/mobile/termux/performance/memory), ``chall_ml`` (security/
threat-modeling/ml-ops/evals/finetune/RAG/agents), ``chall_product``
(ux/latency/routing/media/ocr/vision-edit/research/source-trust/
distillation). ~400 challenges with verifiable acceptance criteria.
"""

from . import ml as ml  # noqa: F401
from . import product as product  # noqa: F401
from . import systems as systems  # noqa: F401
from . import web as web  # noqa: F401

__all__ = ["ml", "product", "systems", "web"]
