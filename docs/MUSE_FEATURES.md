# Muse-Level Features Implementation

This document describes the comprehensive feature set being built to match (and exceed) Meta's Muse AI capabilities.

## Architecture Overview

### 1. Account Management System ✅ BUILT
**Location:** `nomorals/accounts/`

Secure credential vault with encrypted storage for all service integrations.

**Components:**
- **CredentialVault** - AES-256 encrypted storage for passwords, API keys, OAuth tokens
- **AccountManager** - High-level account lifecycle management
- **AccountCreator** - Automated account creation where possible
- **SessionManager** - Cookie persistence, OAuth token refresh, session state

**Features:**
- Encrypted at rest using AES-256-CTR with HMAC authentication
- Master key derived from passphrase via PBKDF2 (200k iterations)
- Support for multiple credential types (password, API key, OAuth token)
- Automatic credential rotation and expiry tracking
- Session persistence across bot restarts

**Usage:**
```python
from nomorals.accounts import CredentialVault, AccountManager

vault = CredentialVault(db, master_passphrase="your-secret")
manager = AccountManager(vault)

# Store credentials
vault.store("gmail", "bot@example.com", "app-password", tags=["email"])

# Retrieve with auto-decryption
cred = manager.get_credential("gmail", "bot@example.com")
print(cred.password)  # Decrypted on access
```

### 2. Email Integration ✅ BUILT
**Location:** `nomorals/integrations/email_integration.py`

Multi-backend email support with automatic backend selection.

**Backends:**
1. **Gmail API** (OAuth) - Fastest, most reliable for Gmail accounts
2. **IMAP/SMTP** - Works with any email provider
3. **Browser automation** - Fallback for webmail (future)

**Capabilities:**
- Send emails with HTML/plain text
- Read inbox with filtering (unread, folders)
- Search emails (Gmail search syntax or IMAP)
- Mark as read, delete, archive
- Attachment support (future)

**Usage:**
```python
from nomorals.integrations import EmailIntegration

email = EmailIntegration(account_manager, session_manager)

# Send email
await email.send(
    to="friend@example.com",
    subject="Hello",
    body="Hi there!",
    account="bot@example.com"
)

# Read inbox
messages = await email.read_inbox(account="bot@example.com", limit=10)

# Search
results = await email.search(query="from:user@example.com", account="bot@example.com")
```

### 3. Calendar Integration ✅ BUILT
**Location:** `nomorals/integrations/calendar_integration.py`

Multi-backend calendar support for scheduling and reminders.

**Backends:**
1. **Google Calendar API** (OAuth) - Full feature support
2. **CalDAV** - Works with iCloud, Nextcloud, etc. (future)
3. **Browser automation** - Fallback for web calendars (future)

**Capabilities:**
- Create events with attendees, reminders, location
- List upcoming events
- Update and delete events
- Recurring events (future)
- Free/busy queries (future)

**Usage:**
```python
from nomorals.integrations import CalendarIntegration

calendar = CalendarIntegration(account_manager, session_manager)

# Add event
event_id = await calendar.add_event(
    title="Meeting with Alice",
    start="2026-09-30T14:00:00",
    end="2026-09-30T15:00:00",
    account="bot@gmail.com",
    description="Discuss project",
    location="Zoom"
)

# List events
events = await calendar.list_events(account="bot@gmail.com", days_ahead=7)
```

### 4. Shopping Integration ✅ BUILT
**Location:** `nomorals/integrations/shopping_integration.py`

Multi-retailer product search and price comparison.

**Supported Retailers:**
- Amazon (scraping)
- eBay (scraping)
- Walmart (scraping)
- Best Buy (scraping)
- Shopify stores (via connector pattern, future)

**Capabilities:**
- Product search across multiple retailers
- Price comparison
- Add to cart via browser automation
- Checkout automation (future)
- Order tracking (future)

**Usage:**
```python
from nomorals.integrations import ShoppingIntegration

shopping = ShoppingIntegration(account_manager, session_manager, browser)

# Search products
results = await shopping.search("wireless headphones", max_results=10)

# Compare prices
comparison = await shopping.compare_prices("Sony WH-1000XM5")
print(f"Best price: ${comparison.best_price.price} at {comparison.best_price.retailer}")

# Add to cart
await shopping.add_to_cart(product_url, account="amazon@bot.com")
```

### 5. Agent Planning Engine 🚧 TODO
**Location:** `nomorals/agents/planner.py`

Goal decomposition and multi-step task execution.

**Planned Features:**
- Break down complex goals into subtasks
- Parallel task execution where possible
- State tracking for long-running tasks
- Proactive suggestion engine
- Task prioritization

**Example:**
```
User: "Plan a dinner party for 6 people this Saturday"

Agent breaks this down:
1. Check calendar for Saturday availability
2. Create guest list from contacts
3. Send invitations via email
4. Plan menu based on dietary restrictions
5. Create shopping list
6. Order groceries online
7. Set reminders for prep timeline
```

### 6. Smart Home Integration 🚧 TODO
**Location:** `nomorals/integrations/smarthome_integration.py`

Control smart devices via multiple backends.

**Planned Backends:**
- Home Assistant API
- Direct device APIs (Philips Hue, TP-Link, etc.)
- Voice bridge (via Alexa/Google Home)

**Capabilities:**
- Control lights, thermostats, locks
- Create automation scenes
- Monitor device status
- Voice control bridge

### 7. Payment Integration 🚧 TODO
**Location:** `nomorals/integrations/payment_integration.py`

Handle payments across multiple methods.

**Planned Methods:**
- Crypto wallets (Bitcoin, Ethereum, etc.)
- Payment APIs (Stripe, PayPal)
- Browser automation for web payments
- Virtual card generation (Privacy.com, etc.)

### 8. Voice Messages 🚧 TODO
**Location:** `nomorals/voice/messages.py`

Send and receive voice messages via Telegram.

**Planned Features:**
- Speech-to-text (Whisper)
- Text-to-speech (multiple voices)
- Voice message transcription
- Voice command recognition

### 9. Proactive Suggestions 🚧 TODO
**Location:** `nomorals/agents/proactive.py`

Agent suggests actions based on patterns and context.

**Planned Features:**
- Pattern recognition (e.g., "You usually order coffee on Mondays")
- Context-aware suggestions (e.g., "Your flight is tomorrow, should I check you in?")
- Reminder optimization
- Habit tracking

### 10. Health Tracking 🚧 TODO
**Location:** `nomorals/integrations/health_integration.py`

Track health metrics and integrate with fitness apps.

**Planned Integrations:**
- Google Fit API
- Apple Health (via export)
- Manual logging via chat
- Medication reminders

## Implementation Status

| Feature | Status | Priority |
|---------|--------|----------|
| Account Management | ✅ Complete | High |
| Email Integration | ✅ Complete | High |
| Calendar Integration | ✅ Complete | High |
| Shopping Integration | ✅ Complete | Medium |
| Agent Planning Engine | 🚧 Todo | High |
| Smart Home | 🚧 Todo | Medium |
| Payments | 🚧 Todo | Low |
| Voice Messages | 🚧 Todo | Medium |
| Proactive Suggestions | 🚧 Todo | Medium |
| Health Tracking | 🚧 Todo | Low |

## Security Model

### Credential Storage
- All credentials encrypted at rest with AES-256-CTR
- HMAC authentication prevents silent corruption
- Master key derived from passphrase (PBKDF2, 200k iterations)
- Each credential gets unique encryption key (master + credential ID)

### Session Management
- Cookies and tokens persisted securely
- Automatic OAuth token refresh
- Session expiry tracking
- Cleanup of expired sessions

### Privacy
- User controls which services bot connects to
- Opt-out of training data usage
- No data shared with ad systems
- Encrypted VM option (future)

## Next Steps

1. **Build Agent Planning Engine** - This is critical for complex multi-step tasks
2. **Add Voice Messages** - Quick win for Telegram integration
3. **Implement Smart Home** - High value for IoT users
4. **Create Proactive Suggestions** - Makes bot feel intelligent
5. **Add Payment Integration** - Complex but important for full autonomy

## Testing

Each integration should have:
- Unit tests for core logic
- Integration tests with mock services
- End-to-end tests with real services (using test accounts)

## Configuration

All features are configurable via environment variables:

```bash
# Account Management
NM_VAULT_PASSPHRASE=your-secret-passphrase

# Email
NM_EMAIL_DEFAULT_PROVIDER=gmail

# Calendar
NM_CALENDAR_DEFAULT_PROVIDER=google

# Shopping
NM_SHOPPING_RETAILERS=amazon,ebay,walmart,bestbuy
```

## Future Enhancements

- **Multi-account support** - Bot manages multiple identities
- **Account sharing** - Share credentials securely between users
- **Audit logging** - Track all actions for security
- **Backup/restore** - Export/import credential vault
- **Plugin system** - Third-party integrations
