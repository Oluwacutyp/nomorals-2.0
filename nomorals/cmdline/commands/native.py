"""``nm native`` — native-extension status surface."""

from __future__ import annotations

from ..emit import _emit



def _cmd_native(args, context) -> int:
    """Native hot-path status: which kernels run on C++, which fell back."""
    from ... import native as native_mod

    if getattr(args, "build", False):
        ok, msg = native_mod.build(force=True)
        print(f"native build: {'OK' if ok else 'FAILED'} \u2014 {msg}")
    if getattr(args, "benchmark", False):
        bench = native_mod.benchmark()
        agreement = "agreement: True" if bench.get("match") else "agreement: False (top-10 index lists differ)"
        _emit(args, {"benchmark": bench},
              "\n".join([
                  f"benchmark: {bench['vectors']} vectors x {bench['dim']} dim, "
                  f"top-10, backend {bench.get('backend')}",
                  f"  python: {bench['python_ms']} ms | native: {bench['native_ms']} ms"
                  + (f" | speedup: {bench['speedup']}x" if bench.get('speedup') else ""),
                  agreement,
              ]))
        return 0
    info = native_mod.info()
    mlp = info.get("mlp") or {}
    search_backend = info.get("backend", "pure-python")
    train_backend = mlp.get("backend", "pure-python")
    _emit(args, info,
          f"native vector search: {search_backend}\n"
          f"native mlp training: {train_backend}\n"
          f"compiler: {info.get('compiler') or 'not found (pure-python fallback is fine)'}")
    return 0
