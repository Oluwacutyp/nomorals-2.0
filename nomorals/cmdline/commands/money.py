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
    else:
        tail = "profile"
    print(handle_money_command(tail, context))
    return 0
