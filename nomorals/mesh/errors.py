"""Mesh errors."""

from __future__ import annotations


class MeshError(Exception):
    """Base for all mesh errors."""


class NodeUnknown(MeshError):
    """A node id was referenced that is not registered."""


class TransportError(MeshError):
    """The transport failed to deliver or fetch."""
