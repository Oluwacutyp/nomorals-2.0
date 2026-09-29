#!/usr/bin/env bash
# merge_env.sh — fold .env.example into your live .env WITHOUT touching
# anything you already set.
#
# What it does:
#   - reads every KEY=VALUE line in the example file
#   - if that KEY already exists in your env file (commented or not,
#     anywhere): your line wins, the example line is skipped
#   - missing keys are appended at the bottom, grouped under a
#     "merged from .env.example" marker, with their section comments
#
# Your existing values, your ordering, your comments: untouched.
#
# Usage (Termux or desktop):
#   bash scripts/merge_env.sh                  # ~/.nomorals/.env + ./.env.example
#   bash scripts/merge_env.sh --dry-run        # show what would be added
#   bash scripts/merge_env.sh --env /path/to/.env --example /path/to/.env.example
set -euo pipefail

ENV_FILE="$HOME/.nomorals/.env"
EXAMPLE=".env.example"
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --env) ENV_FILE="$2"; shift 2 ;;
        --example) EXAMPLE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//' | tail -n +14
            exit 0 ;;
        *) echo "unknown flag: $1 (use --env / --example / --dry-run)" >&2; exit 2 ;;
    esac
done

[ -f "$EXAMPLE" ] || { echo "example file not found: $EXAMPLE" >&2; exit 2; }

# collect keys already present in the live env (KEY= at line start,
# `export KEY=`, and commented `# KEY=`)
have_keys() {
    [ -f "$ENV_FILE" ] || return 0
    sed -nE 's/^(export[[:space:]]+)?#?[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=.*/\2/p' "$ENV_FILE"
}

mapfile -t HAVE < <(have_keys)

have() {
    local k="$1"
    local h
    for h in "${HAVE[@]:-}"; do
        [ "$h" = "$k" ] && return 0
    done
    return 1
}

# walk the example, buffering section comments with the first key they
# introduce so newly-added keys keep their context
pending_comments=()
to_add=()
added=0
skipped=0

while IFS= read -r line || [ -n "$line" ]; do
    # strip a leading "export "
    stripped="${line#export }"
    if [[ "$stripped" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
        key="${stripped%%=*}"
        key="${key%%#*}"
        if have "$key"; then
            # key exists (even if commented): drop any pending section
            # comment that was only collected for it
            :
            skipped=$((skipped + 1))
        else
            to_add+=("${pending_comments[@]:-}" "$line")
            pending_comments=()
            added=$((added + 1))
        fi
    elif [[ "$line" =~ ^[[:space:]]*# ]]; then
        pending_comments+=("$line")
    elif [[ -z "${line//[[:space:]]/}" ]]; then
        :   # blank lines inside the example: not worth carrying over
    else
        pending_comments+=("$line")
    fi
done < "$EXAMPLE"

if [ "$added" -eq 0 ]; then
    echo "nothing to add — your $ENV_FILE already defines every key."
    exit 0
fi

render_block() {
    printf '\n# ── merged from .env.example (keys you had not set yet) ──\n\n'
    printf '%s\n' "${to_add[@]}"
}

if [ "$DRY_RUN" -eq 1 ]; then
    echo "would append $added key(s) to $ENV_FILE (skipping $skipped you already set):"
    echo "──────────────────────────────────────────────"
    render_block
    echo "──────────────────────────────────────────────"
    exit 0
fi

mkdir -p "$(dirname "$ENV_FILE")"
[ -f "$ENV_FILE" ] || touch "$ENV_FILE"
render_block >> "$ENV_FILE"
echo "merged: +$added new key(s) appended to $ENV_FILE — your $skipped existing setting(s) untouched."
echo "review the tail:  tail -n 40 \"$ENV_FILE\""
