"""The ai-core shape (Craft-Libs port of the ``ai-core`` crate).

One seam every organ calls when it wants a model to do something:
validated ids, secrets that never print, the error kinds, the
capability and payload types, the model and request shapes, the
job lifecycle, endpoints, connections, roles, consent, history,
the store file, picks, declarative specs and providers — with a
mock provider and the conformance checks every provider must
pass.
"""

from . import (
               capability,
               conformance,
               connection,
               consent,
               endpoint,
               error,
               history,
               ids,
               job,
               local,
               mock,
               model,
               payload,
               picks,
               provider,
               request,
               role,
               secret,
               spec,
               store,
)
from .capability import WELL_KNOWN, CapabilityId, Modality
from .conformance import ConformanceError, run_cancel, run_lifecycle, run_rejects, run_unknown_job
from .connection import Auth, Billing, Connection, ConnectionError, Locality
from .consent import ConsentLedger, ConsentRecord, Disclosure
from .endpoint import Endpoint
from .error import AiError, ErrorKind
from .history import (
               EndRecord,
               HistoryEntry,
               HistoryLog,
               HistoryRecord,
               Outcome,
               StartRecord,
               Usage,
               clip_prompt,
               merge,
)
from .ids import AppId, ConnectionId, IdError, JobId, ModelId, RoleId
from .job import JobSnapshot, JobState
from .local import LocalMotionProvider, suggest_easing, suggest_motion
from .mock import MockProvider
from .model import CapabilitySpec, Features, InputRole, ModelInfo, Pricing
from .payload import Limits, Payload, PayloadKind
from .picks import Catalog, CatalogEntry, Pick, Picked, choose
from .provider import Provider
from .request import Request
from .role import Binding, Resolved, ResolveError, Role, eligible
from .secret import SecretString, redact
from .spec import Spec
from .store import StoreError, StoreFile

__all__ = [
    "AiError",
    "AppId",
    "Auth",
    "Billing",
    "Binding",
    "CapabilityId",
    "CapabilitySpec",
    "Catalog",
    "CatalogEntry",
    "ConformanceError",
    "Connection",
    "ConnectionError",
    "ConnectionId",
    "ConsentLedger",
    "ConsentRecord",
    "Disclosure",
    "EndRecord",
    "Endpoint",
    "ErrorKind",
    "Features",
    "HistoryEntry",
    "HistoryLog",
    "HistoryRecord",
    "IdError",
    "InputRole",
    "JobId",
    "JobSnapshot",
    "JobState",
    "Limits",
    "LocalMotionProvider",
    "Locality",
    "MockProvider",
    "ModelId",
    "ModelInfo",
    "Modality",
    "Outcome",
    "Payload",
    "PayloadKind",
    "Pick",
    "Picked",
    "Pricing",
    "Provider",
    "Request",
    "Resolved",
    "ResolveError",
    "Role",
    "RoleId",
    "SecretString",
    "Spec",
    "StartRecord",
    "StoreError",
    "StoreFile",
    "Usage",
    "WELL_KNOWN",
    "capability",
    "choose",
    "clip_prompt",
    "conformance",
    "connection",
    "consent",
    "eligible",
    "endpoint",
    "error",
    "history",
    "ids",
    "job",
    "local",
    "merge",
    "mock",
    "model",
    "payload",
    "picks",
    "provider",
    "redact",
    "request",
    "role",
    "run_cancel",
    "run_lifecycle",
    "run_rejects",
    "run_unknown_job",
    "secret",
    "spec",
    "store",
    "suggest_easing",
    "suggest_motion",
]
