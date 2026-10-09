# Content Notes — monetization reality check for Devon's short-form system

Researched 2026-10-09 from the platforms' own docs (links below). Numbers
change — re-verify before building a money plan on them.

## YouTube Partner Program (YPP) — verified from YouTube Help

Source: https://support.google.com/youtube/answer/72851 (crawled 2026-10-09)

Eligibility (meet ONE threshold path):

1. **1,000 subscribers** + **4,000 qualified public watch hours** in the
   past 12 months (long-form), OR
2. **1,000 subscribers** + **10 million qualified public Shorts views**
   in the past 90 days.

Also required: follow the Channel Monetisation Policies, live in a
YPP-available country, no active Community Guidelines strikes, 2-step
verification on the Google account, advanced features enabled, and one
active AdSense for YouTube account linked.

Fine print that matters for a content system:

- Shorts-feed watch hours do **not** count toward the 4,000-hour path —
  the two paths are separate.
- Lower "fan funding" tier: 500 subs + 3 public uploads in 90 days +
  3,000 watch hours in 12 months OR 3M Shorts views in 90 days (Supers,
  memberships — not full ad revenue).
- Shorts ad revenue: creators keep ~45% of the allocated revenue pool.
- **July 15, 2025 policy update**: YouTube tightened "inauthentic
  content" enforcement — mass-produced / repetitious / templated videos
  are demonetized more aggressively. Borrowed content must be
  *significantly altered* to count as original; repetitive content must
  be entertaining or educational beyond view-farming. AI-voiced,
  low-effort compilations are explicitly in the crosshairs.

## Reused-content policy — why original/AI visuals are the safe path

YouTube's reused-content policy demonetizes channels whose content is
substantially someone else's work re-uploaded with minimal
transformation (compilations, re-uploaded clips, slideshows of others'
images). For this system that means:

- Original rendered visuals + original/generated audio = safe.
- Re-uploading other creators' clips, even edited, = YPP rejection risk.
- AI-generated content is allowed, but YouTube requires disclosure of
  realistic AI-generated content, and low-effort AI slop fails the
  "inauthentic content" bar above. Originality + real editing is the
  moat, not just "AI made it."

## Per-platform originality rules (short version)

- **YouTube**: reused-content policy + July 2025 inauthentic-content
  update (above). Original visuals/audio per video; don't mass-produce
  identical templates.
- **TikTok**: originality is enforced by the recommendation system more
  than by written policy — reposted/duplicated content gets suppressed
  distribution. Creator Rewards (the monetization program) requires
  original videos over 1 minute with qualified views — verify current
  terms at https://www.tiktok.com/creators/creator-center before
  counting on it.
- **Instagram**: original content is ranked above reposted content
  (Meta's ranking explicitly demotes aggregators/repost accounts).
  Monetization is via gifts, subscriptions, and branded content —
  the Reels Play bonus program ended; verify current options at
  https://creators.instagram.com.
- **X**: ad revenue sharing exists for eligible creators (Premium
  subscribers meeting impression thresholds) — verify current terms at
  https://help.x.com; original video posts outperform link/reshare
  posts in distribution.

## Posting costs to keep in mind

- **YouTube Data API**: `videos.insert` = 1,600 quota units out of the
  default 10,000/day → ~6 uploads/day on a fresh project. Request a
  quota increase in the Cloud console for a real schedule. Thumbnail
  (50) + playlist add (50) are the other upload-adjacent costs.
- **TikTok**: no quota cost, but unaudited apps are private-only +
  5 users/24h until the Content Posting API audit passes (see
  `publish/tiktok.py`).
- **Instagram**: ~100 API-published posts per rolling 24h per account
  (raised from 50 — verify current docs); Reels REQUIRE a public HTTPS
  video URL (Meta fetches it).
- **X**: video upload + tweet are API calls on the paid tier —
  no free quota to manage, but the paid tier is the cost.
