# Phase 5: Social — Mining Report

## Repo survey

### Adapters (`nomorals/social/chat/`)
| File | Lines | State |
|------|-------|-------|
| telegram.py | 2511 | Bot API: 17 admin methods, 11 tgbot_* tools, CHANNEL support. MTProto: personal account. |
| whatsapp.py | 1179 | Baileys bridge: full group/community/channel admin (25+ methods), GROUP_CAPABILITIES table. |
| discord.py | 634 | Personal account (token-gated). Basic. |
| sms.py | 388 | SMS adapter. |
| gateway.py | 730 | Multi-adapter routing, rate limiting, per-hour windows. |
| control.py | 2357 | Command registry, telegram menu, whatsapp_menu (data-driven). |
| style.py | 138 | Platform-aware menu primitives (menu_item/section/divider). |
| tgbot_buttons.py | 242 | Inline keyboard helpers for Telegram bot. |
| side_chats.py | 305 | Side chat routing. |

### Permissions (`nomorals/partner/`)
- `group_roles.py` — resolve_group_role() per platform, 60s cache, fail-closed.
- `social_gate.py` — SocialGrant tiers: owner / group-admin / member. Public/private matrix.

### Menus
- Telegram: `help_text()` / menu from LIST_GROUPS registry — data-driven, never drifts.
- WhatsApp: `whatsapp_menu()` — OWN menu, text-first, paginated ("reply 3"), sections from registry + GROUP_CAPABILITIES.

## What's strong
1. Adapter coverage is real — not stubs.
2. Menus are registry-driven (can't drift from actual commands).
3. Role-based permissions work per-platform.
4. Style primitives are platform-aware.

## What's weak (the build targets)
1. **Group/channel outputs are raw dicts.** `whatsapp_group_members` → `{"ok": True, "members": [...]}`. The brain improvises formatting every time. No systematic rich rendering.
2. **No contextual group cards.** A members list is just names. No roles shown, no activity, no join context, no stats.
3. **Menus are static.** Same menu for everyone, every time. Doesn't adapt to time, usage, or chat context.
4. **No cross-platform output consistency.** Telegram group info and WhatsApp group info look nothing alike.
5. **Channel outputs minimal.** Post confirmations are bare.

## Best-in-class (outside)
- Telegram native: inline keyboards, not text walls. Rich formatting via HTML.
- WhatsApp native: text-first, short, emoji-led. No buttons (personal account).
- Discord native: embeds with fields, colors, thumbnails.
- Best bots: output adapts to platform idioms. Same data, native feel each time.

## Build plan
1. **`nomorals/social/render.py`** (new) — systematic output renderer:
   - `render_group_card()` — group info: name, member count, admins, description, settings, activity.
   - `render_members()` — members with roles, join context.
   - `render_community()` — community with subgroups, announcement channel.
   - `render_channel()` — channel info + recent posts.
   - Platform-aware: Telegram (HTML), WhatsApp (text+emoji), Discord (embed dict).
2. **Wire into tools** — group/channel tools return rendered output alongside raw data.
3. **Adaptive menus** — menus highlight contextually relevant sections (recent commands, time-of-day, chat kind).
4. **Tests** — rendering per platform, no raw dict leaks.

## Trash with gold
- `tgbot_buttons.py` (242 lines) — inline keyboard helpers that were barely used. Gold: the button layout patterns for Telegram-native menus.
- `whatsapp_cost.py` — cost tracking for WhatsApp sends. Gold: the per-message cost model could inform output chunking.
- Old `style.py` section() — simple but the emoji-led headers are the right WhatsApp idiom.
