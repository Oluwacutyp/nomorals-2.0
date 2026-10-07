"""``nm money`` — money surfaces."""

from __future__ import annotations



def _cmd_money(args, context):
    """Money-making opportunities hunter, from the shell."""
    from ...agents.opportunities import handle_money_command
    action = getattr(args, "money_action", "") or "list"
    if action == "scan":
        tail = f"scan {args.kind}".strip()
    elif action == "list":
        tail = "new" if args.new else "list"
    elif action == "apply":
        tail = f"apply {' '.join(args.apply_args)}".strip()
    elif action == "applications":
        tail = f"applications {args.app_status}".strip()
    else:
        tail = "profile"
    print(handle_money_command(tail, context))
    return 0
