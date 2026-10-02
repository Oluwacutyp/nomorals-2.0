"""Split from nomorals.games.games.cases (Wave H3) — content unchanged."""
from __future__ import annotations

from typing import Any

# ══════════════════════════════════════════════════════════════════════
# God-tier case metadata: title, tier, briefing, motives, red herrings
# and the fair-play solution for each of the 30 original bank cases,
# keyed by bank order.  Applied to _RAW_CASES at import by _enrich().
# ══════════════════════════════════════════════════════════════════════
_BANK_META: tuple[dict[str, Any], ...] = (
    # ── 0 · the vault map ─────────────────────────────────────────
    {
        "title": "The Vault Map",
        "tier": "medium",
        "briefing": "During the gala, a rare first-edition map vanished "
                    "from the museum's sealed vault. Four people had "
                    "access that night — and one of them walked out "
                    "with it.",
        "motives": {
            "the curator": "about to be fired over the gala budget "
                           "shortfall; a quiet sale would cover it.",
            "the security guard": "drowning in gambling debt and "
                                  "recently asked for an advance.",
            "the caterer": "was promised a bonus that never came; "
                           "resents the museum.",
            "the donor's assistant": "the donor collects maps and has "
                                     "paid finders' fees before.",
        },
        "red_herrings": [
            "The curator was inside the vault at 21:41 — the exact "
            "minute of the theft. It looks damning until the "
            "inventory sheet's timestamp matches the alarm system "
            "to the second: she was working, not stealing.",
        ],
        "solution": "The vault log shows a keycard swipe at 21:41, but "
                    "the guard's card was reported lost at 20:00 — "
                    "someone used it anyway. The caterer's van and the "
                    "assistant's call log clear them; the curator's "
                    "inventory sheet clears her. The second, unissued "
                    "fob on the guard's desk — with the vault number "
                    "scratched into it — names the security guard.",
    },
    # ── 1 · the championship trophy ───────────────────────────────
    {
        "title": "The Championship Trophy",
        "tier": "medium",
        "briefing": "The night of the final, the championship trophy "
                    "was stolen from the trophy room. The room needs "
                    "a key or a code — and four people knew both.",
        "motives": {
            "the head coach": "the loss would be blamed on him; "
                              "stealing the trophy muddies the story.",
            "the groundskeeper": "twenty years of service, no pension; "
                                 "the trophy would fetch plenty.",
            "the trophy designer": "unpaid for the restoration work; "
                                   "the trophy is leverage.",
            "the team photographer": "sells prints on the side and "
                                     "knows exactly who buys stolen "
                                     "silver.",
        },
        "red_herrings": [
            "The head coach had the code and was seen near the room "
            "that night — but the press-box camera holds him at "
            "22:15, 23:30 and 00:45. He never left his seat.",
        ],
        "solution": "The door was unlocked from the inside with the "
                    "code — someone who knew it. The groundskeeper's "
                    "radio log, the designer's 18:40 exit and the "
                    "coach's press-box camera all clear them. The "
                    "photographer's camera holds 40 dark-room photos "
                    "never uploaded to the team server — the trophy "
                    "room, shot in the dark. The team photographer "
                    "took it.",
    },
    # ── 2 · the cold-storage breach ───────────────────────────────
    {
        "title": "The Cold-Storage Breach",
        "tier": "easy",
        "briefing": "Twenty years of company archives were drained "
                    "from cold storage in one night. The transfer "
                    "needed an encryption key — and four people had "
                    "one.",
        "motives": {
            "the sysadmin": "passed over for promotion twice; the "
                            "breach embarrasses the CTO who passed "
                            "him over.",
            "the CFO": "the archives hold the audit trail of her "
                       "expense irregularities.",
            "the intern": "was told the internship ends Friday; a "
                          "data theft would impress a rival firm.",
            "the external auditor": "paid by a competitor to find "
                                    "dirt — or manufacture it.",
        },
        "red_herrings": [
            "The sysadmin rotated the service keys that morning — "
            "suspicious timing, until you see the rotation killed "
            "the old keys two hours before the breach window. He "
            "locked the thief out, not in.",
        ],
        "solution": "The transfer ran under a service account from a "
                    "laptop — but the old keys were dead by 09:00. "
                    "The auditor's disk was signed clean, the CFO's "
                    "tokens were all in one city. The intern's build "
                    "machine had a 4 TB drive connected at 02:12, "
                    "still plugged in at 06:00. The intern drained "
                    "the archive.",
    },
    # ── 3 · the walking statue ────────────────────────────────────
    {
        "title": "The Walking Statue",
        "tier": "easy",
        "briefing": "On a Tuesday, the museum's bronze statue walked "
                    "out the service gate. The gate needs a badge — "
                    "four were issued that day.",
        "motives": {
            "the docent": "facing redundancy; a scandal might save "
                          "her department's funding.",
            "the restorer": "owes the conservation supplier and "
                            "knows private bronze buyers.",
            "the security contractor": "up for contract renewal; a "
                                       "theft proves the museum needs "
                                       "him — or pays him off.",
            "the school group teacher": "her school's art program "
                                         "was cut; the statue would "
                                         "fund it for a decade.",
        },
        "red_herrings": [
            "Badge 7 swiped twice, ten minutes apart — the docent's "
            "tour was in the hallway on camera the whole time. "
            "Badge 7 belongs to the conservation department, not "
            "to her.",
        ],
        "solution": "Badge 7 — the conservation department's — swiped "
                    "at 11:38 and 11:52: in with the badge, out with "
                    "the statue. The docent is on hallway camera, the "
                    "teacher's group was photographed outside, the "
                    "contractor was on a live check-in. The "
                    "restorer's bench holds bronze dust and a crate "
                    "cut to the statue's plinth. The restorer took "
                    "it.",
    },
    # ── 4 · the captain's safe ────────────────────────────────────
    {
        "title": "The Captain's Safe",
        "tier": "easy",
        "briefing": "While the yacht sat at the marina during the "
                    "party, the captain's safe was opened. It takes "
                    "the combination and one key — and the key never "
                    "leaves the captain.",
        "motives": {
            "the first mate": "owed back pay and passed over for his "
                              "own command.",
            "the harbor master": "the captain reported him for "
                                 "taking bribes; revenge pays.",
            "the catering skipper": "losing the catering contract "
                                    "next season; the safe held the "
                                    "new tender.",
            "the captain's daughter": "cut out of the will last "
                                      "spring; the safe held the "
                                      "revised papers.",
        },
        "red_herrings": [
            "The captain's daughter streams for a living and was "
            "online all night — but a stream can be pre-recorded. "
            "Hers wasn't: the tournament archive is public and "
            "live, with chat reacting in real time.",
        ],
        "solution": "The safe was opened with the key, not the "
                    "combination — no dialing wear. The harbor "
                    "master is on pier camera, the skipper's GPS was "
                    "at the fuel dock, the daughter's stream is "
                    "public and live. The first mate's locker holds "
                    "a freshly cut duplicate key with the safe's "
                    "serial scratched into the tang. The first mate "
                    "opened it.",
    },
    # ── 5 · the cold-room swap ────────────────────────────────────
    {
        "title": "The Cold-Room Swap",
        "tier": "medium",
        "briefing": "A batch of insulin in the pharmacy's cold room "
                    "was swapped for room-temperature decoys. It took "
                    "a key fob and a cold-chain override — four "
                    "people had both.",
        "motives": {
            "the head pharmacist": "the batch was about to expire; a "
                                   "swap hides the loss from the "
                                   "board.",
            "the night stocker": "moonlights for a discount supplier "
                                 "who pays for diverted stock.",
            "the courier driver": "behind on van payments; pharma "
                                  "vials sell fast.",
            "the regional auditor": "auditing this exact pharmacy; a "
                                    "planted fault justifies her "
                                    "consulting contract.",
        },
        "red_herrings": [
            "The courier scanned in and out at 05:12 — inside the "
            "window. But the chain-of-custody scan is a handoff, "
            "not a stay: the driver never left the dock, and the "
            "swap was detected hours later.",
        ],
        "solution": "The override was a manual hold from inside the "
                    "cold room. The courier's scan was a dock "
                    "handoff, the auditor's tablet was signed "
                    "read-only, the pharmacist was two cities away. "
                    "The night stocker's fob shows one more cold-room "
                    "swipe at 05:47 than his shift log accounts for. "
                    "The night stocker made the swap.",
    },
    # ── 6 · the pirate frequency ──────────────────────────────────
    {
        "title": "The Pirate Frequency",
        "tier": "easy",
        "briefing": "At 3 a.m., a pirate frequency took over the "
                    "station's output for eleven minutes — a "
                    "pre-recorded message sent from the studio. Only "
                    "four people had the studio key.",
        "motives": {
            "the program director": "the board is selling the "
                                    "station; a scandal tanks the "
                                    "price so her buyout group wins.",
            "the night engineer": "fired next month in the merger; "
                                  "the broadcast is his farewell.",
            "the freelance DJ": "was cut from the schedule; the "
                                "message names the new lineup.",
            "the building superintendent": "the station is behind on "
                                           "his maintenance invoices; "
                                           "eleven minutes of fame "
                                           "is leverage.",
        },
        "red_herrings": [
            "The freelance DJ's hospital visit at 03:00 — eleven "
            "minutes after the broadcast began — looked like a "
            "cover story, until the nurse's log confirmed her "
            "arrival. She was being treated, not transmitting.",
        ],
        "solution": "The transmitter was driven from the studio "
                    "console itself. The engineer's intercom is "
                    "unbroken, the director is on office camera three "
                    "times, the DJ's hospital visit is logged at "
                    "03:00. The superintendent's fob opened the "
                    "studio door at 02:58, and his radio — off its "
                    "charger — sat in the studio bin. The building "
                    "superintendent broadcast it.",
    },
    # ── 7 · the marked deck ───────────────────────────────────────
    {
        "title": "The Marked Deck",
        "tier": "medium",
        "briefing": "After the last hand, a dealer bag was found "
                    "holding a marked deck. The bag is locked; the "
                    "key stays with the floor manager.",
        "motives": {
            "the floor manager": "in debt to the house; a marked "
                                 "deck lets him skim without the "
                                 "cameras seeing.",
            "the head dealer": "was skimmed by the house on tips; "
                               "the deck is payback.",
            "the pit boss": "runs a side game upstairs where marked "
                            "decks are currency.",
            "the security captain": "selling blind spots to a crew; "
                                    "the deck tests the coverage.",
        },
        "red_herrings": [
            "The floor manager's key is the only one — but the key "
            "log shows one checkout and one return, and the cage "
            "camera never lost sight of the bag. The deck was "
            "swapped with the seal tape unbroken.",
        ],
        "solution": "The deck was swapped with the bag shut — seal "
                    "tape unbroken — so the key was never the "
                    "method. The manager's key log is clean, the "
                    "dealer was on table camera, the captain was on "
                    "the roof circuit. The pit boss's office holds a "
                    "card case with the same marking ink and a "
                    "print-shop receipt for custom overlays. The pit "
                    "boss marked the deck.",
    },
    # ── 8 · the double-sold painting ──────────────────────────────
    {
        "title": "The Double-Sold Painting",
        "tier": "medium",
        "briefing": "A painting sold at 18:00 was 'resold' at 21:00 "
                    "to a second buyer — and the first invoice was "
                    "paid twice. Only four people could have forged "
                    "the second sale.",
        "motives": {
            "the lead auctioneer": "paid on commission; two sales "
                                   "mean two commissions.",
            "the registrar": "the catalog had a valuation error; a "
                             "second sale buries it.",
            "the private client's secretary": "the client wanted the "
                                              "painting; a forged "
                                              "sale reroutes it.",
            "the security chief": "owes money all over town; a "
                                  "double payment is easy to skim.",
        },
        "red_herrings": [
            "The lead auctioneer ran the whole evening — but the "
            "broadcast feed and the house clock put him at the "
            "podium, word for word, to the minute. He never touched "
            "the registration terminal.",
        ],
        "solution": "The second certificate was printed from the "
                    "first certificate's own template — same lot "
                    "barcode. The auctioneer is on broadcast, the "
                    "secretary's call log is unbroken, the chief is "
                    "on two timestamped feeds. The registrar's "
                    "drawer holds the template file, opened at 19:22 "
                    "with the barcode still embedded. The registrar "
                    "forged the second sale.",
    },
    # ── 9 · the prize orchid ──────────────────────────────────────
    {
        "title": "The Prize Orchid",
        "tier": "easy",
        "briefing": "The prize orchid was dead by 06:00 — cut at the "
                    "base and dressed to look like disease. Someone "
                    "with the grow-room key and a knife did it.",
        "motives": {
            "the head grower": "the orchid wasn't his cultivar; its "
                               "win would eclipse his life's work.",
            "the horticulture intern": "her own entry was disqualified; "
                                       "the favorite had to fall.",
            "the judge": "was bribed to favor another grower; the "
                         "orchid was in the way.",
            "the delivery driver": "paid by a rival nursery to make "
                                   "the exhibit fail.",
        },
        "red_herrings": [
            "The judge arrived at 05:45 to a dead bloom — early "
            "enough to have done it. But the gate camera holds her "
            "arrival at 05:40, and the bloom was already dressed by "
            "then. She found it, she didn't kill it.",
        ],
        "solution": "The cut was dressed with a fungicide the grow "
                    "room doesn't stock. The judge arrived to a dead "
                    "bloom, the driver's GPS was at the depot, the "
                    "grower was at a conference. The intern's bench "
                    "holds a blade with the same cut width and a "
                    "receipt for that fungicide from two towns over. "
                    "The horticulture intern killed the orchid.",
    },
    # ── 10 · the lighthouse lens ──────────────────────────────────
    {
        "title": "The Lighthouse Lens",
        "tier": "medium",
        "briefing": "During the storm, the lighthouse's first-order "
                    "lens cracked from the inside — a stone from the "
                    "gallery rail, thrown by someone on the island.",
        "motives": {
            "the keeper": "the automation crew arrives next month; a "
                          "broken lens delays his redundancy.",
            "the assistant keeper": "was denied the keeper's post; "
                                    "the lens is the keeper's pride.",
            "the supply boat pilot": "smuggles with the light out; a "
                                     "dark night is business.",
            "the island ranger": "wants the island closed to "
                                 "visitors; a dead lighthouse does "
                                 "it.",
        },
        "red_herrings": [
            "The keeper's log has hourly entries — easy to fake "
            "after the fact. But the supply boat's radar ping "
            "independently fixes the impact at 22:30, and the "
            "keeper was logging the storm, not the gallery.",
        ],
        "solution": "The fracture came from below the gallery — and "
                    "only the assistant held the gallery key that "
                    "night. The keeper's log is corroborated by "
                    "radar, the pilot was at sea, the ranger's power "
                    "died at 19:30. The assistant's locker holds a "
                    "stone matching the one from the gallery — and a "
                    "second, already broken. The assistant keeper "
                    "threw it.",
    },
    # ── 11 · the vault panel ──────────────────────────────────────
    {
        "title": "The Vault Panel",
        "tier": "easy",
        "briefing": "A false floor panel was lifted from the bank "
                    "vault's corner office — one swipe of the vault "
                    "master key — and gone before the night audit.",
        "motives": {
            "the vault manager": "the audit was coming; the panel "
                                 "hid a cash-count discrepancy.",
            "the branch teller": "being squeezed by a loan shark; "
                                 "the panel hid a dead drop.",
            "the armored truck driver": "runs parcels off the books; "
                                        "the panel made a perfect "
                                        "smuggling floor.",
            "the building electrician": "wiring the vault for a "
                                        "bugging job on the side.",
        },
        "red_herrings": [
            "The vault manager's key swiped in and out — the master "
            "key, the exact method. But the cage camera shows the "
            "panel still in place at the 02:00 audit, after his "
            "swipes. He opened the vault; he didn't take the panel.",
        ],
        "solution": "The panel left in a roll cage — wheel tread "
                    "matches the drive. The manager's swipes predate "
                    "the 02:00 audit with the panel in place, the "
                    "teller is on lobby camera, the electrician's fob "
                    "never went near the vault. The driver's cage "
                    "holds a panel cut to the vault's dimensions, "
                    "and his fob swiped at 01:47. The armored truck "
                    "driver took it.",
    },
    # ── 12 · the dubbed reel ──────────────────────────────────────
    {
        "title": "The Dubbed Reel",
        "tier": "easy",
        "briefing": "A reel of the 1968 premiere was found with a "
                    "dubbed scene — swapped for a frame-identical "
                    "fake. Four people touched the reel after the "
                    "screening.",
        "motives": {
            "the projectionist": "the original scene implicates his "
                                 "father; the fake rewrites it.",
            "the archivist": "paid by a collector for the original "
                             "scene, one frame at a time.",
            "the film festival director": "the dubbed scene flatters "
                                           "the festival's sponsor; "
                                           "the premiere must shine.",
            "the security chief": "selling access to the vault; the "
                                  "swap covers a viewing.",
        },
        "red_herrings": [
            "The archivist had the reel alone from 18:00 to 18:40 "
            "— but the shelf camera shows it intact on return. "
            "Access isn't opportunity when the camera watches "
            "the shelf.",
        ],
        "solution": "The fake's splice tape is discontinued "
                    "everywhere except one editing suite. The "
                    "projectionist's booth log is clean, the "
                    "archivist's shelf camera shows the reel intact, "
                    "the chief was on the gallery camera. The "
                    "director's suite holds the same tape and a note: "
                    "'premiere reel — swap after screening'. The "
                    "film festival director swapped it.",
    },
    # ── 13 · the reserve vintage ──────────────────────────────────
    {
        "title": "The Reserve Vintage",
        "tier": "medium",
        "briefing": "A barrel of the reserve vintage was swapped for "
                    "a barrel of the lower cuvee, the cork re-waxed "
                    "to look original. Four people had the cellar "
                    "key.",
        "motives": {
            "the cellar master": "the reserve was his mistake vintage; "
                                 "a swap hides the bad year.",
            "the enology intern": "her thesis needs reserve samples; "
                                  "the swap funds her research.",
            "the sommelier": "in league with a merchant who sells "
                             "'reserve' at reserve prices.",
            "the delivery driver": "paid per 'lost' barrel by a "
                                   "fence in the trade.",
        },
        "red_herrings": [
            "The cellar master's key opened the cellar at 09:00 — "
            "but the rack camera shows the barrel intact at 10:00 "
            "when he left. He was tasting, not swapping.",
        ],
        "solution": "The fresh wax carries a stamp die that matches "
                    "the intern's lab — not the cellar's. The "
                    "master's rack camera clears him, the sommelier "
                    "was at a tasting, the driver's GPS was at the "
                    "dock. The intern's lab holds the same wax batch "
                    "and a receipt for vintage label stock. The "
                    "enology intern swapped the barrel.",
    },
    # ── 14 · the scraped star ─────────────────────────────────────
    {
        "title": "The Scraped Star",
        "tier": "hard",
        "briefing": "A photographic plate from the 1954 comet survey "
                    "was scraped — one star, erased. Four people had "
                    "the archive vault key.",
        "motives": {
            "the archive keeper": "the plate proves his predecessor's "
                                  "discovery; erasing it rewrites "
                                  "the credit.",
            "the visiting professor": "her paper claims the star "
                                      "doesn't exist; the plate says "
                                      "otherwise.",
            "the graduate student": "his thesis depends on the comet "
                                    "trail matching the original — "
                                    "unless the original changes.",
            "the security chief": "paid by a private collector who "
                                  "wants the plate devalued.",
        },
        "red_herrings": [
            "The visiting professor's whole career turns on that "
            "star not existing — the perfect motive. But her flight "
            "landed at 18:00, and the scrape was detected at 15:30. "
            "Motive without opportunity is just a grudge.",
        ],
        "solution": "The scrape is under the emulsion — needle work, "
                    "not a blade. The keeper's shelf camera shows the "
                    "plate intact, the professor was still in the "
                    "air, the chief was on control-room camera. The "
                    "needle's gauge matches the graduate student's "
                    "kit, and his own note reads 'plate 1954-07 — "
                    "comet trail — verify against original'. He "
                    "didn't verify it. He edited it. The graduate "
                    "student scraped the star.",
    },
    # ── 15 · the prize loaf ───────────────────────────────────────
    {
        "title": "The Prize Loaf",
        "tier": "easy",
        "briefing": "The prize loaf from the morning bake was swapped "
                    "for a decoy — the display case seal cut with a "
                    "knife. Four people had the case key.",
        "motives": {
            "the head baker": "his own loaf lost last year; this "
                              "year the prize stays in-house.",
            "the morning assistant": "her sister's bakery is the "
                                     "rival entry; family first.",
            "the judge": "was paid to favor the decoy's baker.",
            "the delivery driver": "delivers for the rival bakery; a "
                                   "swap is a favor repaid.",
        },
        "red_herrings": [
            "The head baker's ovens ran all morning — he had the "
            "means and the hour. But the oven log agrees with the "
            "judge's 06:30 arrival, and the swap was made by 05:45. "
            "He was baking, not swapping.",
        ],
        "solution": "The seal was cut with a knife whose pattern "
                    "matches the morning assistant's bread knife. The "
                    "baker's ovens account for him, the judge "
                    "arrived after the swap, the driver's GPS was at "
                    "the depot. The assistant's bench holds the "
                    "matching knife — and the decoy loaf, still warm, "
                    "under the bench. The morning assistant swapped "
                    "the loaf.",
    },
    # ── 16 · the locked cabin ─────────────────────────────────────
    {
        "title": "The Locked Cabin",
        "tier": "medium",
        "briefing": "A first-class cabin was ransacked on the night "
                    "express — locked from the inside. Four people "
                    "held the master override.",
        "motives": {
            "the conductor": "the passenger reported him for "
                             "rudeness; the ransacking is revenge.",
            "the dining car steward": "the passenger never tips; the "
                                      "steward decided to collect.",
            "the ticket inspector": "runs a theft ring on the night "
                                    "express; the cabin was a target "
                                    "of opportunity.",
            "the security officer": "in debt to the conductor's "
                                    "bookmaker; the cabin held cash.",
        },
        "red_herrings": [
            "The conductor holds the master override and signed the "
            "manifest hourly — but the cabin was intact on his 22:00 "
            "round. The lock was picked, not overridden: the pick "
            "marks are on the bolt, not the override plate.",
        ],
        "solution": "The lock was picked — pick marks on the bolt, "
                    "not the override plate. The conductor's 22:00 "
                    "round found the cabin intact, the steward served "
                    "the dining car till 23:00, the officer is on "
                    "corridor camera. The ticket inspector's bag "
                    "holds a pick set with matching wear and a cabin "
                    "key card never issued to him. The ticket "
                    "inspector ransacked the cabin.",
    },
    # ── 17 · the torn page ────────────────────────────────────────
    {
        "title": "The Torn Page",
        "tier": "medium",
        "briefing": "A page was torn from a first edition in the "
                    "special collections reading room — a clean, "
                    "single pass with a bone folder. Four people had "
                    "the reading room key that night.",
        "motives": {
            "the special collections librarian": "the page proves "
                                                 "the volume is a "
                                                 "forgery; her "
                                                 "catalogue is at "
                                                 "stake.",
            "the graduate conservator": "her dissertation needs the "
                                         "page's watermark; the "
                                         "library refused access.",
            "the bookbinder": "a collector pays per stolen leaf; "
                              "the first edition is his pension.",
            "the security chief": "the page is a map; he knows a "
                                  "buyer who pays in cash.",
        },
        "red_herrings": [
            "The graduate conservator works with paper all day and "
            "was two floors down — but the lab camera holds her "
            "until 22:00. Skill with a bone folder isn't "
            "opportunity.",
        ],
        "solution": "The tear is a single bone-folder pass; the "
                    "folder's width matches the bookbinder's kit. "
                    "The librarian's case camera shows the volume "
                    "intact, the conservator is on lab camera, the "
                    "chief is on reading-room camera. The bookbinder's "
                    "bench holds the matching folder — and a torn "
                    "edge matching the missing page's gutter. The "
                    "bookbinder tore the page.",
    },
    # ── 18 · the gallery case ─────────────────────────────────────
    {
        "title": "The Gallery Case",
        "tier": "easy",
        "briefing": "A locked display case was opened during the "
                    "evening gallery — one swipe of the case key — "
                    "and re-sealed before the night guard's round.",
        "motives": {
            "the gallery curator": "the piece was about to be "
                                   "deaccessioned; a theft freezes "
                                   "the sale.",
            "the night guard": "three months behind on rent; the "
                               "case held small, sellable pieces.",
            "the docent": "writes a true-crime blog; an inside theft "
                          "is content — and cash from the fence.",
            "the security chief": "the insurance payout would cover "
                                  "the gallery's deficit, and his "
                                  "bonus with it.",
        },
        "red_herrings": [
            "The night guard's rounds are the gaps the thief used — "
            "but his tablet photos show the case intact at 19:30, "
            "20:30 and 21:30, all signed. He walked past a sealed "
            "case three times.",
        ],
        "solution": "The lock shows a swipe, not a pry — a key was "
                    "used. The curator's gallery camera shows the "
                    "case intact at 19:00, the guard's tablet photos "
                    "confirm it through 21:30, the chief was in the "
                    "control room. The docent's bag holds a case key "
                    "never issued to her — and a photo of the case "
                    "contents timestamped 19:45. The docent opened "
                    "the case.",
    },
    # ── 19 · the lifted engine ────────────────────────────────────
    {
        "title": "The Lifted Engine",
        "tier": "medium",
        "briefing": "A boat's engine was lifted from its cradle at "
                    "the boatyard overnight — one crane lift, and "
                    "the crane log shows only the yard's operator.",
        "motives": {
            "the yard foreman": "the boat's owner refused his "
                                "repair quote; the engine is "
                                "leverage.",
            "the crane operator": "sells marine parts off the books; "
                                  "an engine is a month's wages.",
            "the yard electrician": "the engine holds a prototype "
                                    "wiring loom he wants to copy.",
            "the harbor pilot": "the boat cut him off last season; "
                                "an engineless boat is revenge.",
        },
        "red_herrings": [
            "The crane log shows one lift at 02:00 from the "
            "operator's station — but the log records the station, "
            "not the hands. The foreman was at his desk on office "
            "camera from 21:00 to 03:00; he never touched it.",
        ],
        "solution": "The lift came from the operator's station — not "
                    "the remote. The foreman is on office camera, "
                    "the electrician's fob never went near the crane "
                    "station, the pilot's vessel left at 19:00. The "
                    "crane operator's locker holds a crane remote "
                    "never issued to him, and his fob swiped at "
                    "01:58. The crane operator lifted the engine.",
    },
    # ── 20 · the vanished Stradivarius ────────────────────────────
    {
        "title": "The Vanished Stradivarius",
        "tier": "easy",
        "briefing": "The Stradivarius vanished from the concert "
                    "hall's green room during intermission. The room "
                    "locks from the inside; four people had the spare "
                    "key.",
        "motives": {
            "the concertmaster": "the soloist got her chair; the "
                                 "violin is revenge.",
            "the page turner": "conservatory debt collectors are "
                               "calling; a Stradivarius answers.",
            "the stage manager": "the insurance would cover the "
                                 "hall's deficit — his job with it.",
            "the instrument tech": "knows every fence in the "
                                   "classical trade.",
        },
        "red_herrings": [
            "The instrument tech polishes violins for a living and "
            "was two rooms away — but the workshop camera holds "
            "him at 20:52, timestamped, polishing a cello. "
            "Proximity isn't presence.",
        ],
        "solution": "The spare key turned once, clean, at 20:52. "
                    "The concertmaster is on the broadcast feed, the "
                    "stage manager's radio never went quiet, the tech "
                    "is on workshop camera. The page turner's satchel "
                    "holds a rosin cloth monogrammed with the "
                    "soloist's initials — and a pawn ticket dated "
                    "that night. The page turner took the "
                    "Stradivarius.",
    },
    # ── 21 · the swapped tapes ────────────────────────────────────
    {
        "title": "The Swapped Tapes",
        "tier": "medium",
        "briefing": "The data center's backup tapes were swapped for "
                    "blanks during the night shift. The tape library "
                    "needs a badge and a PIN.",
        "motives": {
            "the night-shift operator": "the backups prove he "
                                        "deleted the wrong array "
                                        "last month; blanks bury it.",
            "the facilities engineer": "paid by a rival to make the "
                                       "data center look unreliable.",
            "the courier": "the blanks are worthless; the real "
                           "tapes are worth plenty to the right "
                           "buyer.",
            "the security analyst": "auditing the backup regime; a "
                                    "planted failure proves her "
                                    "thesis.",
        },
        "red_herrings": [
            "The courier's van was at the gate at 02:14 — the exact "
            "minute of the swap. But the gate camera shows the "
            "driver inside, engine running, never leaving the van. "
            "The library door needs a badge and a PIN, and he has "
            "neither.",
        ],
        "solution": "The library door logged a clean badge-plus-PIN "
                    "at 02:14. The engineer was on the roof unit, the "
                    "courier never left his van, the analyst's SOC "
                    "session has keystrokes every minute. The "
                    "operator's locker holds a tape label printer "
                    "ribbon with the backup set's barcodes — still "
                    "warm. The night-shift operator swapped the "
                    "tapes.",
    },
    # ── 22 · the weighted saddle ──────────────────────────────────
    {
        "title": "The Weighted Saddle",
        "tier": "easy",
        "briefing": "The favorite's saddle was swapped for a weighted "
                    "replica before the derby. The tack room was "
                    "locked; four people had the combination.",
        "motives": {
            "the head groom": "bet against his own horse; four extra "
                              "kilos is the margin.",
            "the stablehand": "paid by a syndicate to sink the "
                              "favorite's odds.",
            "the jockey's agent": "the jockey's contract renews on a "
                                  "win; a loss moves him to a better "
                                  "stable.",
            "the track vet": "the favorite failed a quiet vetting; a "
                             "weighted loss hides the lameness.",
        },
        "red_herrings": [
            "The head groom buys feed in bulk and was twenty minutes "
            "away at 05:15 — but the receipt is timestamped, and the "
            "swap needed the tack room's hanging scale at 05:20. He "
            "can't be in two places.",
        ],
        "solution": "The replica weighs 4 kg more — the swap needed "
                    "the tack room's hanging scale, used at 05:20. "
                    "The groom's feed-store receipt is twenty minutes "
                    "away, the agent was on a recorded call, the vet "
                    "was drawing blood in barn C. The stablehand's "
                    "trunk holds lead sheeting cut to saddle panels — "
                    "and the favorite's real stirrup leathers. The "
                    "stablehand weighted the saddle.",
    },
    # ── 23 · the penny black ──────────────────────────────────────
    {
        "title": "The Penny Black",
        "tier": "easy",
        "briefing": "The one-penny black was lifted from the philately "
                    "exhibition's case during the members' hour. The "
                    "case opens with a key and a code.",
        "motives": {
            "the exhibit designer": "designed the exhibit for free; "
                                    "the stamp is his fee.",
            "the society president": "the society is bankrupt; the "
                                      "insurance would save it.",
            "the case maker": "built the case; knows its flaws and "
                              "his invoice is unpaid.",
            "the evening guard": "collects penny blacks; this one "
                                 "completes the plate.",
        },
        "red_herrings": [
            "The case maker's van broke down on the motorway at "
            "18:40 — a convenient alibi, manufactured? The recovery "
            "invoice is timed and the breakdown crew confirms it. "
            "Inconvenient for him, exonerating for us.",
        ],
        "solution": "The case was opened with key AND code — and the "
                    "code log shows the designer's personal code at "
                    "19:12. The president was mid-address before two "
                    "hundred members, the case maker was on the "
                    "motorway, the guard's rounds photos show the "
                    "case sealed. The designer's flat holds a stock "
                    "book with the penny black hinged in — and the "
                    "exhibition's spare case key. The exhibit "
                    "designer lifted the stamp.",
    },
    # ── 24 · the photographed recipes ─────────────────────────────
    {
        "title": "The Photographed Recipes",
        "tier": "easy",
        "briefing": "The chef's recipe book was photographed page by "
                    "page in the locked office during service. Four "
                    "people had the office key.",
        "motives": {
            "the sous-chef": "opening her own place next month; the "
                             "book is the menu.",
            "the line cook": "sells kitchen secrets to a rival "
                             "restaurant group.",
            "the restaurant critic": "writing an exposé; the book "
                                     "proves the 'secret' recipes "
                                     "are bought in.",
            "the delivery driver": "paid per photo by a food blogger.",
        },
        "red_herrings": [
            "The critic filed her review at 20:30 — from the dining "
            "room, timestamped. A critic with a camera is "
            "suspicious; a critic with a timestamped filing is "
            "just a critic.",
        ],
        "solution": "The office camera was unplugged at 20:03 — a "
                    "clean pull by someone who knew the blind spot. "
                    "The sous-chef's ticket rail covers her till "
                    "22:00, the critic filed at 20:30, the driver's "
                    "route log brackets the window. The line cook's "
                    "phone holds 47 photos of handwritten recipe "
                    "pages, taken 20:04 to 20:19. The line cook "
                    "photographed the book.",
    },
    # ── 25 · the razored logbook ──────────────────────────────────
    {
        "title": "The Razored Logbook",
        "tier": "medium",
        "briefing": "A logbook page covering the comet's closest "
                    "approach was razored out of the observatory's "
                    "bound volume. Four people had the dome key that "
                    "night.",
        "motives": {
            "the resident astronomer": "the page credits a rival; "
                                      "the razor rewrites history.",
            "the visiting researcher": "her grant depends on the "
                                        "comet data being hers alone.",
            "the telescope tech": "sold the page to a collector; the "
                                  "razor is the delivery.",
            "the night watchman": "paid to make a problem disappear "
                                  "by persons unknown.",
        },
        "red_herrings": [
            "The visiting researcher's hire car sat at the motorway "
            "services from 23:00 to 01:00 — a long stop for coffee. "
            "Long enough to drive back? The GPS says the car never "
            "moved. A boring alibi is still an alibi.",
        ],
        "solution": "The cut is a single razor pass — the blade "
                    "width matches the tech's box cutter, not the "
                    "archive scalpel. The astronomer's school group, "
                    "the researcher's GPS and the watchman's photo "
                    "log all clear them. The tech's toolbox holds the "
                    "cutter with paper dust in the slide — and the "
                    "folded logbook page inside a star chart. The "
                    "telescope tech razored the page.",
    },
    # ── 26 · the missing mold ─────────────────────────────────────
    {
        "title": "The Missing Mold",
        "tier": "medium",
        "briefing": "The high-denomination chip mold went missing "
                    "from the casino's cage during the count. The "
                    "cage needs two keys turned together.",
        "motives": {
            "the cage cashier": "the mold lets him mint chips at "
                                "home; the cage is his mint.",
            "the count supervisor": "the count was short; the mold "
                                    "is the scapegoat's price.",
            "the chip runner": "owes the tables; a mold is worth "
                               "more than the debt.",
            "the surveillance operator": "sells blind spots; the mold "
                                          "proves the cameras missed "
                                          "it.",
        },
        "red_herrings": [
            "The surveillance operator watched the cage feed all "
            "night — on her own console log. Watching the feed "
            "isn't the same as watching the cage; but the log is "
            "corroborated by the count room's own recording. She "
            "watched, and the camera watched her watching.",
        ],
        "solution": "The mold left in a chip rack — RFID logged the "
                    "cage door at 03:12, then nothing. The "
                    "supervisor's count-room camera agrees to the "
                    "minute, the runner's route sheet is signed by "
                    "two dealers, the operator's console log is "
                    "corroborated. The cashier's car holds a chip "
                    "rack with the mold's serial etched inside, "
                    "wrapped in a cage towel. The cage cashier took "
                    "the mold.",
    },
    # ── 27 · the swapped negative ─────────────────────────────────
    {
        "title": "The Swapped Negative",
        "tier": "easy",
        "briefing": "The original negative of the finale was swapped "
                    "for a dupe in the editing suite overnight. The "
                    "suite needs a fob after hours.",
        "motives": {
            "the lead editor": "the finale exposes his affair; the "
                               "dupe cuts the scene.",
            "the assistant editor": "was cut from the credits; the "
                                    "negative is leverage.",
            "the colorist": "paid by the studio to bury the "
                            "director's cut.",
            "the night cleaner": "finds things; sells them.",
        },
        "red_herrings": [
            "The night cleaner's cart GPS shows the third floor from "
            "01:00 to 02:00 — near the suite, after hours. Near "
            "isn't in: the cart never entered the suite, and the "
            "suite needs a fob she doesn't have.",
        ],
        "solution": "The dupe's edge code is one generation off — "
                    "swapped by someone who knew which can to take. "
                    "The lead editor's fob is in the parking garage "
                    "at 23:40, the colorist's render farm ran till "
                    "04:00, the cleaner's cart never entered the "
                    "suite. The assistant editor's drawer holds the "
                    "original negative in a mislabeled can — and a "
                    "pawn ticket for camera gear. The assistant "
                    "editor swapped the negative.",
    },
    # ── 28 · the lifted globe ─────────────────────────────────────
    {
        "title": "The Lifted Globe",
        "tier": "medium",
        "briefing": "The 17th-century globe was lifted from the "
                    "library's map room during the storm. The room "
                    "was locked; four people had the master.",
        "motives": {
            "the map curator": "the globe was misattributed; its "
                               "theft buries his error.",
            "the night shelver": "a collector offered him a year's "
                                  "wages for one quiet night.",
            "the restoration volunteer": "believes the globe belongs "
                                          "to her family's estate.",
            "the security guard": "the storm was cover; the globe "
                                  "is his retirement.",
        },
        "red_herrings": [
            "The restoration volunteer's timesheet ends at 17:00 — "
            "but she was seen near the map room at 16:30. Seen near "
            "isn't seen inside: the exit gate logged her out at "
            "17:04, hours before the storm peaked.",
        ],
        "solution": "The globe's cradle was unscrewed, not forced — "
                    "a screwdriver job by someone who knew the mount. "
                    "The curator was at a donors' dinner, the "
                    "volunteer was gated out at 17:04, the guard's "
                    "rounds put him in the east wing. The shelver's "
                    "cart holds brass screws matching the cradle — "
                    "and a shipping label made out to a private "
                    "collector. The night shelver lifted the globe.",
    },
    # ── 29 · the swapped chip ─────────────────────────────────────
    {
        "title": "The Swapped Chip",
        "tier": "easy",
        "briefing": "The race leader's timing chip was swapped at the "
                    "marathon's halfway point, erasing her split. "
                    "Four marshals worked that station.",
        "motives": {
            "the station chief": "her daughter runs second; a lost "
                                 "split is a lost record.",
            "the course marshal": "paid by a betting syndicate to "
                                  "muddy the timing.",
            "the timing tech": "the system glitched last year; a "
                               "manual swap covers the bug.",
            "the water volunteer": "dating the second-place runner; "
                                   "love is a motive.",
        },
        "red_herrings": [
            "The timing tech's laptop shows the swap as a manual "
            "override from the station terminal — his system, his "
            "crime? The override came from the station terminal, "
            "not his laptop. He reported it; he didn't make it.",
        ],
        "solution": "The chip was swapped, not lost — the dead chip "
                    "was found in the station's bin, wiped. The "
                    "chief's radio net has her voice every five "
                    "minutes, the tech's laptop shows the override "
                    "came from the station terminal, the volunteer "
                    "was photographed two stations down. The "
                    "marshal's vest holds the leader's live chip, "
                    "still pinging, in the inner pocket. The course "
                    "marshal swapped the chip.",
    },
)
