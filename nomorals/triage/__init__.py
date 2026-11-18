"""Issue triage — the autocraft port (lane 83).

Autocraft (storytold/autocraft, dual MIT/Apache-2.0) mirrors a
GitHub backlog into a local database and runs a model pipeline over
it. The portable core lives here: sync the mirror, find duplicate
groups, classify type + flags, assign P0–P4 priority, rank the work,
and ask the owner the scope questions. Review/Fix/Verify/Merge are
not ported — they need the Claude CLI + sandbox workers, and Devon
has her own code-fix machinery.

Submodules: ``store`` (durable mirror + decisions + write audit),
``similarity`` (duplicate candidates), ``classify`` (types, flags,
priority, work rank), ``engine`` (the pipeline), ``surface`` (chat
renders), ``github`` (vault-backed client), ``jobs`` (hourly tick).
"""
