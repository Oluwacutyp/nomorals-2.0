"""Split from nomorals.games.games.cases (Wave H3) — content unchanged."""
from __future__ import annotations

from typing import Any

# ══════════════════════════════════════════════════════════════════════
# New bank cases (bank:30 …): full god-tier schema inline — title, tier,
# briefing, suspects with motives, ordered clues, explicit red herrings,
# fair-play solution, and interview statements.
# ══════════════════════════════════════════════════════════════════════
_NEW_CASES: tuple[dict[str, Any], ...] = (
    # ── 30 · easy ─────────────────────────────────────────────
    {
        "title": "The Missing Mascot",
        "tier": "easy",
        "briefing": "The night before the championship final, the "
                    "Wildcats' beloved mascot costume vanished from "
                    "the locked equipment room. Four people had the "
                    "equipment-room key — and the mascot was due on "
                    "the field at noon.",
        "story": "The Wildcats' mascot costume vanished from the "
                 "locked equipment room the night before the final. "
                 "Four people had the key.",
        "suspects": ["the head coach", "the assistant coach",
                     "the team captain", "the janitor"],
        "motives": {
            "the head coach": "the mascot stunt embarrasses his "
                              "serious program; no costume, no stunt.",
            "the assistant coach": "bet the rival school's boosters "
                                   "the mascot wouldn't appear — and "
                                   "collected.",
            "the team captain": "hazed by the mascot performer last "
                                "year; this is payback.",
            "the janitor": "tired of cleaning up after games; no "
                           "mascot, no mess.",
        },
        "culprit": "the assistant coach",
        "clues": [
            "The equipment-room door was propped with a wedge, not "
            "forced — someone with the key held it open.",
            "The head coach ran film session in the AV room from "
            "19:00 to 21:00; thirty players saw him.",
            "The team captain was at the library until 21:30; the "
            "sign-in sheet agrees.",
            "The janitor's cart was seen outside the equipment room "
            "at 20:15 — but the work log shows he was restocking "
            "the far bathrooms, and the cart GPS agrees.",
            "In the assistant coach's car trunk: the mascot head, "
            "and a betting slip from the rival school's bookmaker "
            "dated that night.",
        ],
        "red_herrings": [
            "The janitor's cart outside the equipment room at 20:15 "
            "— until the work log and the cart GPS put him at the "
            "far bathrooms, restocking.",
        ],
        "solution": "The door was propped, not forced: a key-holder. "
                    "The head coach's film session and the captain's "
                    "library sheet clear them; the janitor's cart GPS "
                    "clears him. The mascot head in the assistant "
                    "coach's trunk — beside a rival bookmaker's "
                    "betting slip — closes it: the assistant coach "
                    "took the mascot.",
        "statements": {
            "the head coach": "Film session, 19:00 to 21:00. Thirty "
                              "players saw me.",
            "the assistant coach": "I left at 19:30. A mascot head in "
                                   "my trunk? Someone's pranking me.",
            "the team captain": "Library till 21:30. The sign-in "
                                "sheet has me.",
            "the janitor": "I was restocking the far bathrooms. The "
                           "cart GPS knows.",
        },
    },
    # ── 31 · easy ─────────────────────────────────────────────
    {
        "title": "The Vanishing Lunch",
        "tier": "easy",
        "briefing": "Three weeks running, meal-prep containers have "
                    "vanished from the office fridge. The fridge "
                    "opens with a keypad code — and exactly four "
                    "people know it.",
        "story": "Meal-prep containers keep vanishing from the office "
                 "fridge, which opens with a keypad code. Four people "
                 "know the code.",
        "suspects": ["the office manager", "the temp worker",
                     "the intern", "the security guard"],
        "motives": {
            "the office manager": "skipping lunches to cover her "
                                  "mother's care bills; free food "
                                  "helps.",
            "the temp worker": "unpaid this month over a payroll "
                               "error; the fridge is back pay.",
            "the intern": "bulking for tryouts; protein is protein.",
            "the security guard": "night shifts with no dinner "
                                  "break; the fridge is a buffet.",
        },
        "culprit": "the temp worker",
        "clues": [
            "The fridge code was entered, not bypassed — the keypad "
            "log shows the code at 12:04 each day.",
            "The office manager was at a client lunch off-site; the "
            "receipt is timestamped 12:10.",
            "The intern was at the gym; the turnstile log has him at "
            "11:50.",
            "The security guard's fingerprints are all over the "
            "fridge — but he restocks the water cooler beside it "
            "every morning, logged.",
            "In the temp worker's desk drawer: three labeled "
            "containers with the victims' names still on the lids.",
        ],
        "red_herrings": [
            "The guard's fingerprints all over the fridge — from "
            "restocking the water cooler beside it, every morning, "
            "logged.",
        ],
        "solution": "The keypad log shows the code used at 12:04 "
                    "daily — an insider. The manager's client receipt "
                    "and the intern's gym turnstile clear them; the "
                    "guard's prints are from the water cooler. The "
                    "temp's drawer holds the victims' labeled "
                    "containers. The temp worker took the lunches.",
        "statements": {
            "the office manager": "Client lunch, off-site. The "
                                  "receipt is timestamped.",
            "the temp worker": "I bring my own lunch. The drawer? "
                               "Those are mine — same containers.",
            "the intern": "Gym at 11:50. The turnstile logged me.",
            "the security guard": "I restock the water cooler every "
                                  "morning. That's why my prints are "
                                  "there.",
        },
    },
    # ── 32 · medium ───────────────────────────────────────────
    {
        "title": "The Silent Auction",
        "tier": "medium",
        "briefing": "At the charity gala's silent auction, the "
                    "winning bid sheets were swapped overnight — top "
                    "bids reassigned to lesser lots. Four people "
                    "handled the sheets after close.",
        "story": "The silent auction's winning bid sheets were "
                 "swapped overnight; top bids were reassigned. Four "
                 "people handled the sheets.",
        "suspects": ["the charity director", "the treasurer",
                     "the auctioneer", "the volunteer"],
        "motives": {
            "the charity director": "the gala flopped; inflated "
                                    "results save her job.",
            "the treasurer": "skimming the gap between real and "
                             "recorded bids for months.",
            "the auctioneer": "paid a flat fee; a scandal gets him "
                              "rebooked as the fixer.",
            "the volunteer": "her family's item 'won' by a stranger; "
                             "she wanted it back.",
        },
        "culprit": "the treasurer",
        "clues": [
            "The sheets were reprinted, not altered — the paper "
            "stock matches the treasurer's office printer.",
            "The charity director was on stage with the mayor until "
            "midnight; the photos agree.",
            "The auctioneer's voice is on the PA recording, calling "
            "lots until 23:40.",
            "The volunteer's initials are on the original sheets — "
            "but she initialed them at 18:00 during setup, logged.",
            "In the treasurer's office: the original bid sheets, and "
            "a ledger showing the skimmed differences paid into a "
            "personal account.",
        ],
        "red_herrings": [
            "The volunteer's initials on the sheets — from the 18:00 "
            "setup log, hours before the swap.",
        ],
        "solution": "The sheets were reprinted on the treasurer's "
                    "office stock. The director's stage photos and "
                    "the auctioneer's PA recording clear them; the "
                    "volunteer's initials are from setup. The "
                    "treasurer's office holds the originals — and a "
                    "ledger of the skim. The treasurer swapped the "
                    "sheets.",
        "statements": {
            "the charity director": "On stage with the mayor till "
                                    "midnight. The photos are "
                                    "everywhere.",
            "the treasurer": "I reconciled the books at 22:00 and "
                             "went home. Reprinted sheets? Check the "
                             "paper supplier.",
            "the auctioneer": "My voice is on the PA till 23:40. I "
                              "never left the podium.",
            "the volunteer": "I initialed at setup, 18:00. The log "
                             "says so.",
        },
    },
    # ── 33 · medium ───────────────────────────────────────────
    {
        "title": "The Greenhouse Break-in",
        "tier": "medium",
        "briefing": "Rare orchid seeds vanished from the botanical "
                    "garden's greenhouse vault. The lock was picked "
                    "with a tension wrench — four people had vault "
                    "access.",
        "story": "Rare orchid seeds vanished from the greenhouse "
                 "vault; the lock was picked with a tension wrench. "
                 "Four people had access.",
        "suspects": ["the head botanist", "the volunteer coordinator",
                     "the seed supplier", "the night guard"],
        "motives": {
            "the head botanist": "her research grant depends on "
                                 "exclusive access; the seeds are "
                                 "leverage.",
            "the volunteer coordinator": "sells rare seeds to private "
                                          "collectors on the side.",
            "the seed supplier": "the vault holds his competitor's "
                                 "stock; theft kills the contract.",
            "the night guard": "three months behind on rent; seeds "
                               "are small and valuable.",
        },
        "culprit": "the volunteer coordinator",
        "clues": [
            "The vault lock was picked with a tension wrench, not "
            "drilled — someone skilled, someone patient.",
            "The head botanist gave the keynote at a conference; the "
            "stream archive is public.",
            "The seed supplier's truck GPS shows the depot all "
            "night; the tracker log is intact.",
            "The night guard's torch was found inside the vault — "
            "but he dropped it on rounds at 22:00, logged, and the "
            "vault was sealed then.",
            "At the volunteer coordinator's potting bench: seed "
            "packets with the vault's lot numbers, and an email "
            "thread with a private buyer.",
        ],
        "red_herrings": [
            "The guard's torch inside the vault — dropped on his "
            "22:00 rounds, logged, while the vault was still sealed.",
        ],
        "solution": "A tension-wrench pick: skilled hands. The "
                    "botanist's keynote stream, the supplier's depot "
                    "GPS clear them; the guard's torch was dropped "
                    "on logged rounds. The coordinator's bench holds "
                    "the vault's lot-numbered packets and a buyer's "
                    "email thread. The volunteer coordinator took "
                    "the seeds.",
        "statements": {
            "the head botanist": "Keynote, streamed. The archive is "
                                 "public.",
            "the volunteer coordinator": "I pot seedlings, that's "
                                          "all. Seed packets? We all "
                                          "have them.",
            "the seed supplier": "Depot all night. The tracker log "
                                 "is intact.",
            "the night guard": "Dropped my torch on rounds at 22:00. "
                               "It's in the log.",
        },
    },
    # ── 34 · medium ───────────────────────────────────────────
    {
        "title": "The Marathon Bib",
        "tier": "medium",
        "briefing": "An hour before the start, the elite favorite's "
                    "bib and timing chip were stolen from the "
                    "athletes' tent. Four people had tent access.",
        "story": "The elite runner's bib and timing chip were stolen "
                 "from the athletes' tent before the start. Four "
                 "people had access.",
        "suspects": ["the race director", "the physio",
                     "the rival runner", "the kit manager"],
        "motives": {
            "the race director": "the favorite insulted the event; a "
                                 "DNS embarrasses her instead.",
            "the physio": "paid by a betting ring to sideline the "
                          "favorite.",
            "the rival runner": "second place pays half; the bib is "
                                "worth the difference.",
            "the kit manager": "sells signed bibs to collectors; a "
                               "star's bib is the prize.",
        },
        "culprit": "the physio",
        "clues": [
            "The bib was lifted from the kit bag and the zipper "
            "re-closed — someone who knew the bag's layout.",
            "The race director was on the start-line PA; the "
            "recording is timestamped.",
            "The rival runner was on the warm-up camera from 06:00 "
            "to 07:00, timestamped.",
            "The kit manager's scissors were found by the tent — but "
            "she spent the morning cutting tape, logged on the "
            "supply sheet.",
            "In the physio's treatment bag: the favorite's bib, and "
            "a burner phone with one call to a known bookmaker.",
        ],
        "red_herrings": [
            "The kit manager's scissors by the tent — she spent the "
            "morning cutting athletic tape, logged on the supply "
            "sheet.",
        ],
        "solution": "The zipper was re-closed: someone who knew the "
                    "kit bag. The director's PA recording and the "
                    "rival's warm-up camera clear them; the kit "
                    "manager's scissors cut tape all morning. The "
                    "physio's bag holds the bib — and a burner phone "
                    "with one call to a bookmaker. The physio stole "
                    "the bib.",
        "statements": {
            "the race director": "Start-line PA, timestamped. I never "
                                 "left the gantry.",
            "the physio": "I was taping ankles all morning. A bib in "
                          "my bag? Must have been mixed in.",
            "the rival runner": "Warm-up camera, 06:00 to 07:00. "
                                "Check it.",
            "the kit manager": "Cutting tape all morning. The supply "
                               "sheet has me.",
        },
    },
    # ── 35 · medium ───────────────────────────────────────────
    {
        "title": "The Forged Hold Slip",
        "tier": "medium",
        "briefing": "A hold slip for the Gutenberg fragment was "
                    "forged, rerouting the treasure to a private "
                    "pickup. Four people use the hold system.",
        "story": "A hold slip for the Gutenberg fragment was forged, "
                 "rerouting it to a private pickup. Four people use "
                 "the hold system.",
        "suspects": ["the head librarian", "the circulation clerk",
                     "the rare-book dealer", "the student assistant"],
        "motives": {
            "the head librarian": "the fragment's insurance lapsed; "
                                  "a theft triggers the payout.",
            "the circulation clerk": "a dealer offered him a year's "
                                     "salary for one quiet reroute.",
            "the rare-book dealer": "has a buyer waiting; the "
                                    "fragment completes a set.",
            "the student assistant": "failing out; the fragment's "
                                     "sale pays the tuition.",
        },
        "culprit": "the circulation clerk",
        "clues": [
            "The forged slip's barcode was generated at the staff "
            "terminal — not the public kiosk.",
            "The head librarian was in a board meeting; the minutes "
            "are signed.",
            "The rare-book dealer was at an estate sale across town; "
            "the invoice is timed.",
            "The student assistant's login generated the slip — but "
            "she left her terminal unlocked, and her phone GPS puts "
            "her at the café across town.",
            "In the circulation clerk's locker: the real hold slip, "
            "and the dealer's business card with a price written on "
            "the back.",
        ],
        "red_herrings": [
            "The student assistant's login on the forged slip — her "
            "terminal sat unlocked while her phone GPS had her at "
            "the café across town.",
        ],
        "solution": "The barcode came from the staff terminal. The "
                    "librarian's board minutes and the dealer's "
                    "estate-sale invoice clear them; the assistant's "
                    "login was an unlocked terminal. The clerk's "
                    "locker holds the real slip — and the dealer's "
                    "card with a price on the back. The circulation "
                    "clerk forged the slip.",
        "statements": {
            "the head librarian": "Board meeting. The minutes are "
                                  "signed.",
            "the circulation clerk": "I process holds all day. A "
                                     "forgery? Check the terminal "
                                     "logs.",
            "the rare-book dealer": "Estate sale across town. The "
                                    "invoice is timed.",
            "the student assistant": "I was at the café. My phone GPS "
                                     "knows — and I left my terminal "
                                     "unlocked, my mistake.",
        },
    },
    # ── 36 · hard ─────────────────────────────────────────────
    {
        "title": "The Clockmaker's Alibi",
        "tier": "hard",
        "briefing": "The 200-year-old regulator clock was smashed — "
                    "but first its chime hammers were loosened, so it "
                    "died silently. Four people had the workshop key.",
        "story": "The regulator clock was smashed — its chime hammers "
                 "loosened first so it died silently. Four people had "
                 "the workshop key.",
        "suspects": ["the master clockmaker", "the apprentice",
                     "the rival collector", "the insurance assessor"],
        "motives": {
            "the master clockmaker": "the clock was his one failure; "
                                     "its destruction buries the flaw.",
            "the apprentice": "the master's will leaves the shop to "
                              "his son; the apprentice inherits "
                              "nothing.",
            "the rival collector": "wanted the clock's movement for "
                                   "his own restoration.",
            "the insurance assessor": "the payout would be the year's "
                                      "biggest; fraud pays twice.",
        },
        "culprit": "the apprentice",
        "clues": [
            "The chime hammers were loosened with a left-handed turn "
            "— the silencing took a steady, practiced hand.",
            "The master clockmaker was at the guild dinner; the "
            "photos are timestamped.",
            "The rival collector was at an auction in another city; "
            "the paddle records agree.",
            "The insurance assessor sent a threatening letter about "
            "the policy — dated four months ago, settled and "
            "countersigned since.",
            "Brass dust under the apprentice's fingernails matches "
            "the hammer screws — and the loosening turns are "
            "left-handed, like the apprentice.",
        ],
        "red_herrings": [
            "The assessor's threatening letter — four months old, "
            "settled and countersigned. A paper trail to nowhere.",
        ],
        "solution": "The hammers were loosened left-handed before "
                    "the smashing — silencing first means planning. "
                    "The master's dinner photos and the collector's "
                    "paddle records clear them; the assessor's letter "
                    "is ancient history. The brass dust under the "
                    "apprentice's nails matches the hammer screws, "
                    "and the turns are left-handed — like the "
                    "apprentice. The apprentice silenced and smashed "
                    "the clock.",
        "statements": {
            "the master clockmaker": "Guild dinner. The photos are "
                                     "timestamped.",
            "the apprentice": "I was oiling the longcase all night. "
                              "Brass dust? It's a workshop.",
            "the rival collector": "Another city, paddle in hand. The "
                                   "records agree.",
            "the insurance assessor": "That letter is four months "
                                      "old and settled. Check the "
                                      "countersignature.",
        },
    },
    # ── 37 · hard ─────────────────────────────────────────────
    {
        "title": "The Poisoned Pen",
        "tier": "hard",
        "briefing": "For a month, poison-pen letters typed on the "
                    "paper's own machines have been ruining "
                    "reputations. Four people have after-hours "
                    "access.",
        "story": "Poison-pen letters, typed on the paper's own "
                 "machines, have ruined reputations for a month. Four "
                 "people have after-hours access.",
        "suspects": ["the editor-in-chief", "the advice columnist",
                     "the typesetter", "the paper's owner"],
        "motives": {
            "the editor-in-chief": "the letters target his rivals; "
                                   "the chaos sells papers.",
            "the advice columnist": "her column was cut for space; "
                                    "the letters are her real voice.",
            "the typesetter": "passed over for the columnist's job "
                              "twice; the letters discredit her.",
            "the paper's owner": "shorting the targets' companies; "
                                 "the letters move markets.",
        },
        "culprit": "the advice columnist",
        "clues": [
            "Every letter was typed on the office's 1962 Olympia — "
            "the ribbon wear is identical across all of them.",
            "The editor-in-chief was at a press conference each "
            "mailing night; the broadcasts agree.",
            "The paper's owner was on a cruise; the ship manifest is "
            "stamped.",
            "The typesetter's fingerprints are on the Olympia — but "
            "he services it weekly, logged in the maintenance book.",
            "The advice columnist's drafts drawer holds the same "
            "misspelling — 'recieve' — as the letters; and the "
            "Olympia's ribbon was replaced the day she complained "
            "hers was running dry.",
        ],
        "red_herrings": [
            "The typesetter's fingerprints on the Olympia — he "
            "services it weekly, logged in the maintenance book.",
        ],
        "solution": "One machine, one ribbon, one typist's tell: "
                    "'recieve'. The editor's broadcasts and the "
                    "owner's cruise clear them; the typesetter's "
            "prints are maintenance. The columnist's drafts share "
                    "the misspelling, and the ribbon was changed the "
                    "day she complained hers ran dry. The advice "
                    "columnist wrote the letters.",
        "statements": {
            "the editor-in-chief": "Press conferences, broadcast. "
                                   "Every mailing night.",
            "the advice columnist": "I write advice, not poison. A "
                                    "misspelling? Everyone makes "
                                    "them.",
            "the typesetter": "I service the Olympia weekly. The "
                              "maintenance book says so.",
            "the paper's owner": "I was on a cruise. The manifest is "
                                 "stamped.",
        },
    },
    # ── 38 · hard ─────────────────────────────────────────────
    {
        "title": "The Flooded Cellar",
        "tier": "hard",
        "briefing": "The reserve cellar flooded overnight when the old "
                    "valve was opened — then closed again, to flood "
                    "slowly and silently. Four people knew where the "
                    "valve was.",
        "story": "The reserve cellar flooded when the old valve was "
                 "opened overnight, then closed again. Four people "
                 "knew the valve's location.",
        "suspects": ["the vineyard owner", "the cellar hand",
                     "the wine critic", "the insurance broker"],
        "motives": {
            "the vineyard owner": "the vintage was corked; a flood "
                                  "writes it off.",
            "the cellar hand": "fired last week, rehired on appeal; "
                               "the flood is his grievance.",
            "the wine critic": "panned the vintage; a ruined cellar "
                               "proves her right.",
            "the insurance broker": "the flood policy pays double "
                                    "for 'slow' water damage.",
        },
        "culprit": "the cellar hand",
        "clues": [
            "The valve was opened with a pipe wrench, then closed "
            "again — timed to flood slowly while everyone slept.",
            "The vineyard owner was at a wedding; the photos are "
            "timestamped.",
            "The wine critic was filing from abroad; the submission "
            "is timestamped.",
            "The insurance broker sold the flood policy last month "
            "— but the underwriter confirms it predates the sale "
            "listing by a year.",
            "The cellar hand's boots carry silt from the cellar "
            "floor — and the wrench marks on the valve match the "
            "wrench missing from his kit.",
        ],
        "red_herrings": [
            "The broker's flood policy, sold last month — the "
            "underwriter confirms it predates the sale listing by "
            "a year. Old coverage, not a setup.",
        ],
        "solution": "Opened, then closed: a timed flood, not an "
                    "accident. The owner's wedding photos and the "
                    "critic's filing clear them; the broker's policy "
                    "is a year old. The cellar hand's boots hold the "
                    "cellar's silt, and the valve's wrench marks "
                    "match the wrench missing from his kit. The "
                    "cellar hand flooded the cellar.",
        "statements": {
            "the vineyard owner": "Wedding, all night. The photos are "
                                  "timestamped.",
            "the cellar hand": "I haven't been near the valve in "
                               "weeks. My wrench was stolen.",
            "the wine critic": "Filing from abroad. The submission "
                               "is timestamped.",
            "the insurance broker": "That policy is a year old. Ask "
                                    "the underwriter.",
        },
    },
    # ── 39 · hard ─────────────────────────────────────────────
    {
        "title": "The Rigged Raffle",
        "tier": "hard",
        "briefing": "The charity raffle's winning ticket came from a "
                    "drum that had been loaded with duplicates. Four "
                    "people handled the drum that night.",
        "story": "The charity raffle was drawn from a drum loaded "
                 "with duplicate tickets. Four people handled the "
                 "drum.",
        "suspects": ["the charity chair", "the master of ceremonies",
                     "the ticket printer", "the auditor"],
        "motives": {
            "the charity chair": "her nephew 'won'; the prize stays "
                                 "in the family.",
            "the master of ceremonies": "takes a cut from every "
                                        "'lucky' winner he crowns.",
            "the ticket printer": "paid per thousand tickets; "
                                  "duplicates are pure margin.",
            "the auditor": "auditing the charity; a rigged raffle "
                           "justifies her fee.",
        },
        "culprit": "the master of ceremonies",
        "clues": [
            "The drum's inner panel was unscrewed and re-sealed — "
            "duplicates slipped in through the service hatch.",
            "The charity chair was on stage all night; the video "
            "agrees.",
            "The ticket printer's delivery dockets are signed and "
            "timestamped.",
            "The auditor flagged the raffle last year — but she "
            "flagged it to fix it, and the fixes are documented.",
            "The master of ceremonies' cufflink was found inside the "
            "drum — and his 'random' draw reached for the exact "
            "corner where the service hatch opens.",
        ],
        "red_herrings": [
            "The auditor's report flagging last year's raffle — she "
            "flagged it to fix it, and the fixes are documented.",
        ],
        "solution": "Duplicates entered through the service hatch — "
                    "the panel was unscrewed and re-sealed. The "
                    "chair's stage video and the printer's dockets "
                    "clear them; the auditor's flag led to fixes. "
                    "The master of ceremonies' cufflink sat inside "
                    "the drum, and his draw went straight for the "
                    "hatch corner. The master of ceremonies rigged "
                    "the raffle.",
        "statements": {
            "the charity chair": "On stage all night. The video "
                                 "agrees.",
            "the master of ceremonies": "I draw what I draw. A "
                                        "cufflink? I lose them "
                                        "everywhere.",
            "the ticket printer": "Dockets signed and timestamped. "
                                  "Count them.",
            "the auditor": "I flagged it to fix it. The fixes are "
                           "documented.",
        },
    },
    # ── 40 · hard ─────────────────────────────────────────────
    {
        "title": "The Burned Ledger",
        "tier": "hard",
        "briefing": "The firm's paper ledger archive was torched — "
                    "but the fire started in the one cabinet holding "
                    "the expense records. Four people had the archive "
                    "key.",
        "story": "The ledger archive was torched — the fire started "
                 "in the cabinet holding the expense records. Four "
                 "people had the archive key.",
        "suspects": ["the managing partner", "the night auditor",
                     "the cleaning supervisor", "the IT admin"],
        "motives": {
            "the managing partner": "the expense records show his "
                                    "skimming; fire is cheaper than "
                                    "prison.",
            "the night auditor": "she found the skimming and was "
                                 "paid to un-find it.",
            "the cleaning supervisor": "her crew is blamed for every "
                                       "loss; the fire is leverage "
                                       "for a raise.",
            "the IT admin": "the paper archive proves his digital "
                            "migration failed; ash hides it.",
        },
        "culprit": "the night auditor",
        "clues": [
            "The fire started with lighter fluid on the expense "
            "cabinet only — targeted, not random.",
            "The managing partner was at a client dinner; the "
            "receipt is timestamped.",
            "The cleaning supervisor's cart GPS shows another floor "
            "all night.",
            "The IT admin's remote access at 02:00 was a scheduled "
            "backup — logged and routine.",
            "The night auditor's timesheet shows her clocked out at "
            "01:00 — but the archive camera caught her badge at "
            "01:40; she re-entered and never logged it.",
        ],
        "red_herrings": [
            "The IT admin's 02:00 remote access — a scheduled "
            "backup, logged and routine.",
        ],
        "solution": "Lighter fluid on one cabinet: arson with a "
                    "reading list. The partner's dinner receipt and "
                    "the supervisor's cart GPS clear them; the "
                    "admin's access was a scheduled backup. The "
                    "night auditor clocked out at 01:00 — then her "
                    "badge opened the archive at 01:40, unlogged. "
                    "The night auditor burned the ledger.",
        "statements": {
            "the managing partner": "Client dinner. The receipt is "
                                    "timestamped.",
            "the night auditor": "I clocked out at 01:00. The "
                                 "camera must be wrong.",
            "the cleaning supervisor": "Another floor all night. The "
                                       "cart GPS knows.",
            "the IT admin": "Scheduled backup at 02:00. It's in the "
                            "log.",
        },
    },
    # ── 41 · hard ─────────────────────────────────────────────
    {
        "title": "The Switched Slides",
        "tier": "hard",
        "briefing": "Minutes before the keynote, the lecture slides "
                    "were swapped for embarrassing fakes. Four people "
                    "knew the podium laptop's password.",
        "story": "The keynote slides were swapped for fakes minutes "
                 "before the lecture. Four people knew the podium "
                 "laptop's password.",
        "suspects": ["the professor", "the teaching assistant",
                     "the AV technician", "the rival lecturer"],
        "motives": {
            "the professor": "stage fright; a ruined talk cancels "
                             "the tenure review.",
            "the teaching assistant": "the professor took credit for "
                                      "her research; humiliation is "
                                      "payback.",
            "the AV technician": "the department cut his hours; the "
                                 "chaos proves he's needed.",
            "the rival lecturer": "up for the same chair; a "
                                  "disaster is an opening.",
        },
        "culprit": "the teaching assistant",
        "clues": [
            "The swap came through the podium's USB port — the "
            "laptop never left the hall.",
            "The professor was in the green room; three witnesses "
            "agree.",
            "The AV technician was on the balcony mixer; the console "
            "log agrees.",
            "The rival lecturer's feud with the professor is public "
            "— but he was live on a panel in another city, "
            "streamed.",
            "The teaching assistant's laptop holds the fake slides' "
            "source files — and the USB drive's serial matches the "
            "one she signed out of the media cage.",
        ],
        "red_herrings": [
            "The rival lecturer's very public feud — while he was "
            "live on a streamed panel in another city.",
        ],
        "solution": "A USB swap on a laptop that never left the "
                    "hall: an insider with the password. The "
                    "professor's witnesses and the tech's console log "
                    "clear them; the rival was streamed from another "
                    "city. The teaching assistant's laptop holds the "
                    "fakes' source files, and the USB serial matches "
                    "her media-cage sign-out. The teaching assistant "
                    "switched the slides.",
        "statements": {
            "the professor": "Green room. Three witnesses.",
            "the teaching assistant": "I was printing handouts. The "
                                      "USB? I sign out drives all "
                                      "the time.",
            "the AV technician": "Balcony mixer. The console log "
                                 "agrees.",
            "the rival lecturer": "Live panel, another city. It's "
                                  "streamed.",
        },
    },
    # ── 42 · hard ─────────────────────────────────────────────
    {
        "title": "The Counterfeit Tickets",
        "tier": "hard",
        "briefing": "Counterfeit tickets flooded the sold-out show — "
                    "printed on the theater's own stock. Four people "
                    "could reach the paper stock.",
        "story": "Counterfeit tickets flooded the sold-out show, "
                 "printed on the theater's own stock. Four people "
                 "could reach the stock.",
        "suspects": ["the theater manager", "the box-office temp",
                     "the usher", "the printing contractor"],
        "motives": {
            "the theater manager": "the show's profits were short; "
                                   "extra 'sales' fill the gap.",
            "the box-office temp": "paid per ticket by a scalping "
                                   "ring; the stock is the press.",
            "the usher": "takes bribes at the door; counterfeits "
                         "need a blind eye.",
            "the printing contractor": "overprinted the run and "
                                       "sells the extras himself.",
        },
        "culprit": "the box-office temp",
        "clues": [
            "The fakes are on genuine stock — the theft was paper, "
            "not tickets.",
            "The theater manager was at a donors' dinner; the "
            "photos are timestamped.",
            "The printing contractor's delivery dockets are signed "
            "and counted.",
            "The usher's locker holds torn ticket stubs — but his "
            "job is tearing tickets, and every stub is genuine.",
            "The box-office temp's bag holds a ream wrapper with the "
            "stock's lot number — and the box-office camera shows "
            "her printing 'test pages' after close.",
        ],
        "red_herrings": [
            "The usher's locker of torn stubs — his job is tearing "
            "tickets, and every stub is genuine.",
        ],
        "solution": "Genuine stock, fake print: someone stole paper. "
                    "The manager's dinner photos and the contractor's "
                    "dockets clear them; the usher's stubs are all "
                    "genuine. The temp's bag holds the stock's lot "
                    "wrapper, and the camera caught her printing "
            "'test pages' after close. The box-office temp printed "
                    "the counterfeits.",
        "statements": {
            "the theater manager": "Donors' dinner. The photos are "
                                   "timestamped.",
            "the box-office temp": "I print test pages to calibrate. "
                                   "Everyone does.",
            "the usher": "I tear tickets. That's the job — check "
                         "the stubs.",
            "the printing contractor": "Dockets signed and counted. "
                                        "Every ream.",
        },
    },
    # ── 43 · hard ─────────────────────────────────────────────
    {
        "title": "The Jammed Safe",
        "tier": "hard",
        "briefing": "The shop's safe was jammed with glue — not "
                    "robbed, just sealed shut. Four people knew the "
                    "safe's make and its weakness.",
        "story": "The shop's safe was jammed with glue — sealed, not "
                 "robbed. Four people knew the safe's weakness.",
        "suspects": ["the shop owner", "the locksmith's apprentice",
                     "the rival locksmith", "the night guard"],
        "motives": {
            "the shop owner": "the insurance pays for a drilling job; "
                              "a jammed safe is a claim.",
            "the locksmith's apprentice": "drumming up drilling work "
                                          "on the side; glue is "
                                          "advertising.",
            "the rival locksmith": "the shop uses his competitor; a "
                                   "jammed safe is a sales pitch.",
            "the night guard": "fell asleep on shift; a sealed safe "
                               "hides what he missed.",
        },
        "culprit": "the locksmith's apprentice",
        "clues": [
            "Glue injected through the dial spindle — a locksmith's "
            "trick to force an expensive drilling job.",
            "The shop owner was home; the alarm log agrees.",
            "The rival locksmith was at a trade show; the badge "
            "scans agree.",
            "The night guard was asleep at 03:00 on camera — but the "
            "jamming happened at 01:00, and the safe was already "
            "sealed when he dozed.",
            "The locksmith's apprentice's kit is missing its glue "
            "syringe — and the work order for a 'drilling job' was "
            "written in his hand before the owner even called.",
        ],
        "red_herrings": [
            "The night guard asleep at 03:00 — two hours after the "
            "01:00 jamming. Embarrassing, not incriminating.",
        ],
        "solution": "Glue through the spindle is a locksmith's "
                    "trick — it forces a drilling job. The owner's "
                    "alarm log and the rival's badge scans clear "
                    "them; the guard dozed at 03:00, two hours late. "
                    "The apprentice's kit is missing its glue "
                    "syringe, and he wrote up the drilling work "
                    "order before the owner called. The locksmith's "
                    "apprentice jammed the safe.",
        "statements": {
            "the shop owner": "Home. The alarm log agrees.",
            "the locksmith's apprentice": "I was home asleep. My "
                                          "syringe? Lent it out.",
            "the rival locksmith": "Trade show. The badge scans "
                                   "agree.",
            "the night guard": "I dozed at 03:00, I admit it. But "
                               "the safe was sealed by then.",
        },
    },
    # ── 44 · expert ───────────────────────────────────────────
    {
        "title": "The Vanishing Ink",
        "tier": "expert",
        "briefing": "Signed settlement contracts reverted to blank "
                    "overnight. The ink was designed to disappear — "
                    "and four people handled the originals.",
        "story": "Signed settlement contracts reverted to blank "
                 "overnight — disappearing ink. Four people handled "
                 "the originals.",
        "suspects": ["the senior partner", "the paralegal",
                     "the opposing counsel", "the courier"],
        "motives": {
            "the senior partner": "the settlement was his mistake; "
                                  "blank contracts reopen the case.",
            "the paralegal": "paid by the other side to make the "
                             "settlement evaporate.",
            "the opposing counsel": "voided contracts mean a better "
                                    "deal for her client.",
            "the courier": "paid per 'lost' document; blank ones are "
                           "deniable.",
        },
        "culprit": "the paralegal",
        "clues": [
            "The ink was a commercial disappearing formulation — it "
            "fades in exactly 48 hours, and the signing was 48 hours "
            "ago.",
            "The senior partner was on a flight; the manifest is "
            "stamped.",
            "The courier's route log has timestamps every twenty "
            "minutes, unbroken.",
            "The opposing counsel benefits most from voided "
            "contracts — but her office's ink order is standard "
            "stock, verified, and she handled only copies.",
            "The paralegal's desk drawer holds an empty vial of the "
            "same formulation — and she insisted the signing happen "
            "Thursday, exactly 48 hours before the hearing.",
        ],
        "red_herrings": [
            "The opposing counsel, who profits most from voided "
            "contracts — her office's ink is verified standard "
            "stock, and she never touched the originals.",
        ],
        "solution": "Disappearing ink fades in 48 hours; the signing "
                    "was 48 hours before the hearing — the timing is "
                    "the weapon. The partner's flight and the "
                    "courier's log clear them; opposing counsel never "
                    "touched the originals. The paralegal's drawer "
                    "holds the empty vial, and she set the signing "
                    "for Thursday — the 48-hour fuse. The paralegal "
                    "inked the vanishing contracts.",
        "statements": {
            "the senior partner": "On a flight. The manifest is "
                                  "stamped.",
            "the paralegal": "I just prepare the paperwork. Ink? "
                             "The office buys it in bulk.",
            "the opposing counsel": "I handled copies, never the "
                                    "originals. Check the chain of "
                                    "custody.",
            "the courier": "Route log, every twenty minutes. "
                           "Unbroken.",
        },
    },
    # ── 45 · expert ───────────────────────────────────────────
    {
        "title": "The Double-Booked Vault",
        "tier": "expert",
        "briefing": "Two parties were assigned the same safe-deposit "
                    "box — and one's contents vanished. Four people "
                    "could assign boxes.",
        "story": "Two parties were assigned the same safe-deposit "
                 "box; one's contents vanished. Four people could "
                 "assign boxes.",
        "suspects": ["the bank manager", "the vault officer",
                     "the auditor", "the safe-deposit client"],
        "motives": {
            "the bank manager": "the missing contents embarrass the "
                                "bank; a 'clerical error' buries his "
                                "negligence.",
            "the vault officer": "knew the first holder was abroad; "
                                 "an empty box is an opportunity.",
            "the auditor": "the audit was coming; a vanished box "
                           "proves the controls failed — her "
                           "report, her fee.",
            "the safe-deposit client": "the second assignee; claims "
                                       "the box was empty when she "
                                       "got it.",
        },
        "culprit": "the vault officer",
        "clues": [
            "The box was assigned twice in the ledger — the second "
            "entry backdated by three weeks.",
            "The bank manager was at a conference; the badge scans "
            "agree.",
            "The auditor's review covered another branch that week; "
            "the report is filed.",
            "The safe-deposit client's key fits the box — of course "
            "it does; she was issued it. Her entry log shows one "
            "visit, the box untouched.",
            "The vault officer's handwriting matches the backdated "
            "entry — and only she knew the first holder was abroad "
            "for a month.",
        ],
        "red_herrings": [
            "The client's key fitting the box — she was issued it. "
            "One logged visit, the box untouched.",
        ],
        "solution": "A backdated second assignment: the ledger was "
                    "rewritten, not mistaken. The manager's badge "
                    "scans and the auditor's filed report clear "
                    "them; the client's single visit is logged. The "
                    "backdated entry is in the vault officer's hand "
                    "— and only she knew the first holder was abroad "
                    "for a month. The vault officer double-booked "
                    "the vault.",
        "statements": {
            "the bank manager": "Conference. The badge scans agree.",
            "the vault officer": "I assign boxes all day. Backdated? "
                                 "The ledger clerk must have erred.",
            "the auditor": "I reviewed another branch that week. "
                           "The report is filed.",
            "the safe-deposit client": "One visit. The box was empty "
                                        "when I got it.",
        },
    },
    # ── 46 · expert ───────────────────────────────────────────
    {
        "title": "The Ghost Bidder",
        "tier": "expert",
        "briefing": "All season, a ghost bidder has driven prices up "
                    "— bids from an account that never pays. Four "
                    "people could be running it.",
        "story": "A ghost bidder drove auction prices up all season "
                 "from an account that never pays. Four people could "
                 "run it.",
        "suspects": ["the auction house owner", "the seller's brother",
                     "the collector", "the platform engineer"],
        "motives": {
            "the auction house owner": "higher hammer prices mean "
                                       "higher commissions.",
            "the seller's brother": "his brother's lots keep "
                                    "'selling' high; the family "
                                    "keeps the spread.",
            "the collector": "drives up rivals' prices, then buys "
                             "low elsewhere.",
            "the platform engineer": "sells the bidding bot to "
                                     "whoever pays.",
        },
        "culprit": "the seller's brother",
        "clues": [
            "The ghost account bids in the last three seconds, every "
            "time — a script, or a very fast thumb.",
            "The auction house owner's accounts were frozen in a "
            "divorce; the court records are public.",
            "The platform engineer's deploy log shows no out-of-hours "
            "access; every deploy is signed.",
            "The collector wins often and pays late — but his "
            "payments clear eventually. Slow isn't ghost.",
            "The seller's brother's phone holds the auction app's "
            "admin token — and the ghost account only ever bids on "
            "his brother's lots, never anyone else's.",
        ],
        "red_herrings": [
            "The collector, winning often and paying late — his "
            "payments clear eventually. Slow isn't ghost.",
        ],
        "solution": "Three-second bids on every lot, never a payment: "
                    "a shill script. The owner's frozen accounts and "
                    "the engineer's signed deploys clear them; the "
                    "collector pays, eventually. The ghost only bids "
                    "on the seller's brother's lots — and the "
                    "brother's phone holds the app's admin token. "
                    "The seller's brother ran the ghost bidder.",
        "statements": {
            "the auction house owner": "My accounts are frozen. The "
                                       "court records are public.",
            "the seller's brother": "I just watch my brother's "
                                    "lots. An admin token? I don't "
                                    "know what that is.",
            "the collector": "I pay late, I pay. Check the "
                             "clearances.",
            "the platform engineer": "Every deploy signed. The log "
                                     "is clean.",
        },
    },
    # ── 47 · expert ───────────────────────────────────────────
    {
        "title": "The Cold Open",
        "tier": "expert",
        "briefing": "Opening night: sand in the stage trap, and the "
                    "star fell through on cue. Four people knew the "
                    "trap's timing.",
        "story": "Opening night: sand in the stage trap — the star "
                 "fell through on cue. Four people knew the trap's "
                 "timing.",
        "suspects": ["the director", "the understudy",
                     "the stagehand", "the critic"],
        "motives": {
            "the director": "the star upstaged him for years; a "
                            "public fall is a lesson.",
            "the understudy": "one fall and the role is hers — "
                              "opening night, no less.",
            "the stagehand": "the star got him fired from the last "
                             "production; this is the grudge.",
            "the critic": "a disaster is the review of a lifetime.",
        },
        "culprit": "the understudy",
        "clues": [
            "Sand was poured into the trap mechanism — it jammed "
            "halfway, on the star's cue exactly.",
            "The director was in the booth; the recording agrees.",
            "The critic was in her seat; the usher's log agrees.",
            "The stagehand's fingerprints are on the trap — but he "
            "maintains it nightly, logged in the fly book.",
            "The understudy's script has the star's cue marked "
            "'mine' — and she was the only one who asked for the "
            "trap to be 'tested' that afternoon.",
        ],
        "red_herrings": [
            "The stagehand's fingerprints on the trap — he "
            "maintains it nightly, logged in the fly book.",
        ],
        "solution": "Sand in the mechanism, jammed on the star's cue "
                    "exactly: sabotage with a script. The director's "
                    "booth recording and the critic's usher log clear "
                    "them; the stagehand's prints are nightly "
                    "maintenance. The understudy marked the star's "
                    "cue 'mine' — and asked for the trap to be "
                    "'tested' that afternoon. The understudy set the "
                    "cold open.",
        "statements": {
            "the director": "In the booth. The recording agrees.",
            "the understudy": "I was warming up backstage. 'Mine'? "
                              "It's just a note.",
            "the stagehand": "I maintain the trap nightly. The fly "
                             "book says so.",
            "the critic": "In my seat. The usher's log agrees.",
        },
    },
    # ── 48 · expert ───────────────────────────────────────────
    {
        "title": "The Salted Assay",
        "tier": "expert",
        "briefing": "The assay samples were salted — gold added to "
                    "make a dead mine look rich. Four people handled "
                    "the samples.",
        "story": "The assay samples were salted with gold to make a "
                 "dead mine look rich. Four people handled the "
                 "samples.",
        "suspects": ["the mine owner", "the assayer's assistant",
                     "the geologist", "the buyer"],
        "motives": {
            "the mine owner": "a rich assay sells the mine; a dead "
                              "one bankrupts him.",
            "the assayer's assistant": "paid per 'promising' result "
                                       "by the owner's broker.",
            "the geologist": "her reputation rides on the find; a "
                             "rich assay makes her career.",
            "the buyer": "wants the price low; a salted assay he "
                         "exposes buys it cheap.",
        },
        "culprit": "the assayer's assistant",
        "clues": [
            "The gold in the samples is placer gold, not vein gold "
            "— it was added, not found.",
            "The mine owner's phone records put him in another "
            "state; the tower pings agree.",
            "The buyer's flight records show him abroad all month.",
            "The geologist's report was optimistic — but her samples "
            "came from the same salted batch; she read what she was "
            "given.",
            "The assayer's assistant's pan shows placer-gold residue "
            "— and he ordered 'test weights' of placer gold last "
            "month.",
        ],
        "red_herrings": [
            "The geologist's optimistic report — written from the "
            "same salted batch everyone else read. She read what "
            "she was given.",
        ],
        "solution": "Placer gold in vein samples: the gold was "
                    "added, not found. The owner's tower pings and "
                    "the buyer's flights clear them; the geologist "
                    "read the salted batch she was given. The "
                    "assistant's pan carries placer-gold residue, "
                    "and he ordered placer 'test weights' last "
                    "month. The assayer's assistant salted the "
                    "assay.",
        "statements": {
            "the mine owner": "Another state. The tower pings agree.",
            "the assayer's assistant": "I prep samples, that's all. "
                                       "Test weights are standard.",
            "the geologist": "I reported what the samples showed. "
                             "They were salted before I saw them.",
            "the buyer": "Abroad all month. The flight records "
                         "agree.",
        },
    },
    # ── 49 · expert ───────────────────────────────────────────
    {
        "title": "The Phantom Trolley",
        "tier": "expert",
        "briefing": "For a month, the museum trolley tour's takings "
                    "were skimmed — cash fares that never reached "
                    "the box. Four people rode the route.",
        "story": "The trolley tour's takings were skimmed for a "
                 "month — cash fares that never reached the box. "
                 "Four people rode the route.",
        "suspects": ["the museum director", "the conductor",
                     "the ticket seller", "the mechanic"],
        "motives": {
            "the museum director": "the tour loses money; skimmed "
                                   "cash hides the shortfall.",
            "the conductor": "pocketing cash fares is a month's "
                             "wages, tax-free.",
            "the ticket seller": "her drawer never balances; the "
                                 "trolley covers it.",
            "the mechanic": "the trolley's repairs are his invoices; "
                            "cash keeps them quiet.",
        },
        "culprit": "the conductor",
        "clues": [
            "The skim matches the cash fares exactly — card sales "
            "reconcile to the cent.",
            "The museum director was at fundraisers; the photos "
            "agree.",
            "The mechanic's time sheets are signed and match the "
            "work orders.",
            "The ticket seller's drawer was short twice — but both "
            "shorts were card-machine errors, documented and "
            "refunded.",
            "The conductor's cash pouch has a false bottom — and his "
            "reported cash counts match the card-sale pattern, not "
            "the passenger counts.",
        ],
        "red_herrings": [
            "The ticket seller's two short drawers — both "
            "card-machine errors, documented and refunded.",
        ],
        "solution": "Only cash went missing; cards reconcile to the "
                    "cent — the thief handled cash. The director's "
                    "fundraisers and the mechanic's time sheets clear "
                    "them; the seller's shorts were documented "
                    "machine errors. The conductor's pouch has a "
                    "false bottom, and his cash counts follow the "
                    "card pattern, not the passenger counts. The "
                    "conductor skimmed the trolley.",
        "statements": {
            "the museum director": "Fundraisers. The photos agree.",
            "the conductor": "I report what I collect. A false "
                             "bottom? It's a reinforced pouch.",
            "the ticket seller": "Both shorts were machine errors. "
                                 "Documented and refunded.",
            "the mechanic": "Time sheets signed. Match them to the "
                            "work orders.",
        },
    },
    # ── 50 · expert ───────────────────────────────────────────
    {
        "title": "The Mirrored Manuscript",
        "tier": "expert",
        "briefing": "The scanned manuscript came back mirrored — "
                    "every page flipped, hiding a forged signature. "
                    "Four people touched the scans.",
        "story": "The scanned manuscript was mirrored — every page "
                 "flipped, hiding a forged signature. Four people "
                 "touched the scans.",
        "suspects": ["the curator", "the digitization tech",
                     "the scholar", "the IT admin"],
        "motives": {
            "the curator": "the signature proves the manuscript is "
                           "hers to sell; the mirror hides the "
                           "forgery.",
            "the digitization tech": "paid by the forger to make the "
                                     "signature unreadable.",
            "the scholar": "her translation depends on the "
                           "signature being genuine.",
            "the IT admin": "the mirror covers his failed backup; "
                            "corrupted scans are deniable.",
        },
        "culprit": "the digitization tech",
        "clues": [
            "The mirroring is in the scan files themselves, not the "
            "viewer — done at capture, not in display.",
            "The curator was at a symposium; the program lists her.",
            "The IT admin's access logs are clean and complete.",
            "The scholar published on the manuscript last year — "
            "but from the physical copy, before digitization; her "
            "notes predate the scans.",
            "The digitization tech's workstation holds a batch script "
            "that mirrors on export — and he ran it the night the "
            "'calibration failed'.",
        ],
        "red_herrings": [
            "The scholar's published work on the manuscript — from "
            "the physical copy, before digitization; her notes "
            "predate the scans.",
        ],
        "solution": "Mirrored at capture, not in the viewer: the "
                    "sabotage happened at the scanner. The curator's "
                    "symposium and the admin's clean logs clear them; "
                    "the scholar worked from the physical copy. The "
                    "tech's workstation holds a mirror-on-export "
                    "script — run the night the 'calibration failed'. "
                    "The digitization tech mirrored the manuscript.",
        "statements": {
            "the curator": "Symposium. The program lists me.",
            "the digitization tech": "Calibration failed that night. "
                                     "The script is standard.",
            "the scholar": "I worked from the physical copy, last "
                           "year. My notes predate the scans.",
            "the IT admin": "Access logs clean and complete.",
        },
    },
    # ── 51 · expert ───────────────────────────────────────────
    {
        "title": "The Lost Luggage Con",
        "tier": "expert",
        "briefing": "The lost-luggage auction was rigged — the best "
                    "bags never reached the block. Four people saw "
                    "the intake.",
        "story": "The lost-luggage auction was rigged — the best "
                 "bags never reached the block. Four people saw the "
                 "intake.",
        "suspects": ["the auctioneer", "the baggage handler",
                     "the frequent flyer", "the warehouse guard"],
        "motives": {
            "the auctioneer": "skimming the premium bags before the "
                              "hammer; the block gets the dregs.",
            "the baggage handler": "cherry-picking bags at intake is "
                                   "a second salary.",
            "the frequent flyer": "her own lost bag was valuable; "
                                  "rigging its 'loss' recovers it.",
            "the warehouse guard": "the intake room is his kingdom; "
                                   "tolls are tradition.",
        },
        "culprit": "the baggage handler",
        "clues": [
            "The intake log skips numbers — bags logged out before "
            "the auction list was ever made.",
            "The auctioneer's commission records reconcile to the "
            "cent.",
            "The frequent flyer's travel records show her abroad "
            "during intake.",
            "The warehouse guard's rounds missed the warehouse twice "
            "— but the missing bags left before his shift, on the "
            "intake timestamps.",
            "The baggage handler's locker holds luggage tags matching "
            "the skipped log numbers — and his phone has photos of "
            "the bags, priced.",
        ],
        "red_herrings": [
            "The guard's two missed rounds — the missing bags left "
            "before his shift, on the intake timestamps.",
        ],
        "solution": "Skipped log numbers: bags left before the "
                    "auction list existed. The auctioneer's "
                    "commissions reconcile and the flyer was abroad; "
                    "the guard's missed rounds came after the bags "
                    "left. The handler's locker holds the skipped "
                    "tags — and his phone holds the bags' photos, "
                    "priced. The baggage handler rigged the auction.",
        "statements": {
            "the auctioneer": "Commissions reconcile to the cent.",
            "the baggage handler": "I move bags, that's all. Tags? "
                                   "Souvenirs.",
            "the frequent flyer": "Abroad during intake. The travel "
                                  "records agree.",
            "the warehouse guard": "Missed two rounds, I admit it. "
                                   "The bags were gone before my "
                                   "shift.",
        },
    },
)
