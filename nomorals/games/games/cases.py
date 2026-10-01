"""The case bank for the ``case`` game (InvestigationGame).

Fifty-two full cases across four tiers (easy/medium/hard/expert), each
with airtight internal logic:

* ``id``          — "bank:N", stable across runs.
* ``title``       — the case's name.
* ``tier``        — easy | medium | hard | expert.
* ``briefing``    — the setup, two to three sentences.
* ``story``       — the crime, one or two lines (shown at case open).
* ``suspects``    — four named suspects.
* ``motives``     — one motive per suspect (everyone has a reason).
* ``culprit``     — one of the suspects.
* ``clues``       — five, sharpest last, revealed in order.  The pattern
  every case follows:
  clue 1  = the mechanism (how it was done)
  clue 2  = an alibi that clears one innocent
  clue 3  = an alibi that clears a second innocent
  clue 4  = a red herring (looks suspicious, is innocent)
  clue 5  = the smoking gun that names the culprit
* ``red_herrings``— the misleading trails, spelled out (each names an
  innocent — never the culprit).
* ``herring_suspect`` — the suspect the red herring points at.
* ``solution``    — the fair-play deduction walkthrough: how the clues
  identify the culprit.
* ``statements``  — one line per suspect, what they say when asked.
  The culprit's statement contradicts clue 5 (the smoking gun); the
  innocents' statements are consistent with their alibis.  Interviewing
  (``ask <name>``) is how you find the contradiction before you accuse.
* ``difficulty``  — back-compat grade (expert maps to "hard").

``verify_solvability`` runs the fair-play check: the true culprit must
be deducible from the clues alone.  ``deal_case`` serves cases with
per-player anti-repeat tracking, and ``adapt_case`` tunes clue counts
to the player's solve rate.
"""
from __future__ import annotations

import time
from typing import Any

__all__ = [
    "CASES", "BANK_SIZE", "TIERS", "HINT_COST", "TIME_LIMITS",
    "TIME_BONUS_MAX", "HISTORY_VERSION",
    "generate_case", "random_case", "verify_solvability",
    "blank_history", "eligible_tiers", "solve_rate", "deal_case",
    "record_played", "adapt_case", "score_solve", "unseen_bank_cases",
]

#: difficulty tiers, easiest first
TIERS = ("easy", "medium", "hard", "expert")

#: base score paid for closing a case, by tier
TIER_BASE_SCORE = {"easy": 3, "medium": 5, "hard": 8, "expert": 12}

#: score points one hint costs
HINT_COST = 2

#: timed-mode countdown per tier, in seconds
TIME_LIMITS = {"easy": 300, "medium": 240, "hard": 180, "expert": 150}

#: maximum time bonus (full speed) per tier, in score points
TIME_BONUS_MAX = {"easy": 4, "medium": 6, "hard": 10, "expert": 16}

#: bump when the persisted history schema changes
HISTORY_VERSION = 1

#: how many seen case ids are kept per tier (anti-repeat memory)
_SEEN_CAP = 400


_RAW_CASES: tuple[dict[str, Any], ...] = (
    # ── 1 ────────────────────────────────────────────────────────────
    {
        "story": "The rare first-edition map vanished from the sealed "
                 "vault during the gala. Four people had access.",
        "suspects": ["the curator", "the security guard", "the caterer",
                     "the donor's assistant"],
        "culprit": "the security guard",
        "clues": [
            "The vault log shows a keycard swipe at 21:41 — but the "
            "guard's card was 'lost' at 20:00 and 'found' at 23:00.",
            "The caterer's van was at the dock at 21:30, off the "
            "delivery route — waiting on a run that never came.",
            "The donor's assistant was on a live video call with the "
            "donor from 21:00 to 22:30. The call log is unbroken.",
            "The curator was in the vault at 21:41 doing inventory — "
            "the timestamp on the inventory sheet matches the alarm "
            "system exactly, to the second.",
            "On the guard's desk: a second keycard fob, unissued, "
            "with the vault number scratched into the plastic.",
        ],
        "statements": {
            "the curator": "I was in the vault at 21:41 doing inventory. "
                           "The sheet's timestamp matches the alarm to "
                           "the second.",
            "the security guard": "My card was lost and I reported it at "
                                  "20:00. I was on the east wing the "
                                  "whole time.",
            "the caterer": "I was at the dock at 21:30 waiting on a "
                           "delivery that was running late.",
            "the donor's assistant": "I was on a live call with the donor "
                                     "from 21:00 to 22:30. The log is "
                                     "unbroken.",
        },
    },
    # ── 2 ────────────────────────────────────────────────────────────
    {
        "story": "The championship trophy was stolen from the trophy "
                 "room the night of the final. The room needs a key or a "
                 "code.",
        "suspects": ["the head coach", "the groundskeeper", "the trophy "
                     "designer", "the team photographer"],
        "culprit": "the team photographer",
        "clues": [
            "The door was unlocked from the inside — the code lock "
            "still showed 'entered' in the log.",
            "The groundskeeper was on the radio until 01:00; the relay "
            "log is timestamped and unbroken.",
            "The trophy designer had been asked to appraise it that "
            "afternoon — he was seen leaving at 18:40, before the "
            "stolen window.",
            "The head coach was in the stadium's press box during the "
            "final; the press-box camera caught him at 22:15, 23:30 "
            "and 00:45.",
            "The photographer's camera memory: 311 photos from the "
            "final, and 40 more from a dark room — none of them "
            "uploaded to the team server.",
        ],
        "statements": {
            "the head coach": "I was in the press box the whole final. "
                              "The camera caught me three times.",
            "the groundskeeper": "I was on the radio until 01:00. The "
                                 "relay log is timestamped.",
            "the trophy designer": "I appraised it that afternoon and "
                                   "left at 18:40, before any of it.",
            "the team photographer": "I shot the final, that's all. The "
                                     "dark room? That's the changing "
                                     "room, it's always dark.",
        },
    },
    # ── 3 ────────────────────────────────────────────────────────────
    {
        "story": "Someone drained the company's cold-storage server "
                 "archive — 20 years of data, gone in one night. "
                 "Encryption key required.",
        "suspects": ["the sysadmin", "the CFO", "the intern",
                     "the external auditor"],
        "culprit": "the intern",
        "clues": [
            "The transfer job ran under a service account — but "
            "service accounts can't log in from a laptop.",
            "The external auditor's laptop had a clean disk image "
            "signed by the security team that morning.",
            "The CFO was traveling; her login tokens were all "
            "issued in one city that night.",
            "The sysadmin rotated the service-account keys at 09:00 "
            "that same morning, two hours before the breach window — "
            "the old keys were dead.",
            "The intern's build machine: a 4 TB external drive "
            "connected at 02:12, still plugged in at 06:00.",
        ],
        "statements": {
            "the sysadmin": "I rotated the service keys at 09:00. The "
                            "old ones were dead before any of this.",
            "the CFO": "I was in one city all night. Every token says "
                       "so.",
            "the intern": "My build machine was running the nightly "
                          "job. The drive is for the dataset.",
            "the external auditor": "My disk image was signed that "
                                    "morning. It's read-only.",
        },
    },
    # ── 4 ────────────────────────────────────────────────────────────
    {
        "story": "The museum's bronze statue walked out the service "
                 "gate on a Tuesday. The gate needs a badge. Four "
                 "badges were issued that day.",
        "suspects": ["the docent", "the restorer", "the security "
                     "contractor", "the school group teacher"],
        "culprit": "the restorer",
        "clues": [
            "The gate log: badge 7 at 11:38, badge 7 again at 11:52 "
            "— one badge, two swipes, ten minutes apart.",
            "The docent led the 11:00 tour; the hallway camera has "
            "her in frame continuously from 10:55 to 11:40.",
            "The school group teacher's badge (12) was used once, "
            "10:30, and the group was photographed outside at 10:45.",
            "The security contractor (badge 3) was on a video check-in "
            "from the west wing at 11:38 — live, timestamped.",
            "Badge 7 was issued to the conservation department. The "
            "restorer's workbench: bronze dust, and a crate labeled "
            "for 'fragile, do not stack' — the exact dimensions of the "
            "statue's plinth.",
        ],
        "statements": {
            "the docent": "I led the 11:00 tour. The hallway camera has "
                          "me in frame the whole time.",
            "the restorer": "Badge 7 is the department's, it's not "
                            "mine. I was at the bench all morning.",
            "the security contractor": "I was on a live check-in from "
                                       "the west wing at 11:38.",
            "the school group teacher": "I badged in at 10:30 and the "
                                        "group was photographed outside "
                                        "at 10:45.",
        },
    },
    # ── 5 ────────────────────────────────────────────────────────────
    {
        "story": "The captain's safe was opened while the yacht sat at "
                 "the marina during the party. It takes the combination "
                 "and one key.",
        "suspects": ["the first mate", "the harbor master", "the "
                     "catering skipper", "the captain's daughter"],
        "culprit": "the first mate",
        "clues": [
            "The safe was opened with the key, not the combination — "
            "the hinge shows no dialing wear.",
            "The harbor master was on the pier security camera from "
            "19:00 to 23:00, timestamped.",
            "The catering skipper's boat GPS pinged the fuel dock "
            "until 21:40; the safe was opened at 22:15.",
            "The captain's daughter was in a live tournament from "
            "22:00 to midnight; the stream archive is public.",
            "In the first mate's locker: a duplicate key, freshly cut, "
            "with the safe's serial number scratched into the tang.",
        ],
        "statements": {
            "the first mate": "I was in the crew room all night, "
                              "polishing brass. And the key — the "
                              "captain keeps the only one.",
            "the harbor master": "I kept the pier camera running the "
                                 "whole party. Check the log.",
            "the catering skipper": "We were fueling at the dock until "
                                    "21:40 sharp, then we ran the line.",
            "the captain's daughter": "I was streaming the whole "
                                      "tournament. The clip is public.",
        },
    },
    # ── 6 ────────────────────────────────────────────────────────────
    {
        "story": "A batch of insulin in the pharmacy's cold room was "
                 "swapped with room-temperature decoy vials. Someone "
                 "with a key fob and a cold-chain override did it.",
        "suspects": ["the head pharmacist", "the night stocker",
                     "the courier driver", "the regional auditor"],
        "culprit": "the night stocker",
        "clues": [
            "The override was used from inside the cold room — the log "
            "shows a manual hold, not a system fault.",
            "The courier's delivery was scanned in and out at 05:12; "
            "the swap was detected at 09:30.",
            "The auditor's tablet has a signed disk image from the "
            "chain-of-custody office that morning.",
            "The head pharmacist was at a board meeting two cities "
            "over; the entry log and parking camera agree.",
            "The night stocker's fob: one more swipe on the cold-room "
            "door at 05:47 than his shift log accounts for.",
        ],
        "statements": {
            "the head pharmacist": "I was at the board meeting. The "
                                   "entry log has my signature.",
            "the night stocker": "I stocked at 05:30, exactly as "
                                 "logged. The room was sealed when I "
                                 "left it.",
            "the courier driver": "Scanned in at 05:12, out at 05:12. "
                                  "The chain of custody can confirm.",
            "the regional auditor": "My disk image was signed that "
                                    "morning. Everything on it is "
                                    "read-only.",
        },
    },
    # ── 7 ────────────────────────────────────────────────────────────
    {
        "story": "At 3 a.m. the pirate frequency took over the station's "
                 "output for eleven minutes — a pre-recorded message "
                 "sent from the studio. Only four had the studio key.",
        "suspects": ["the program director", "the night engineer",
                     "the freelance DJ", "the building superintendent"],
        "culprit": "the building superintendent",
        "clues": [
            "The transmitter was driven from the studio console, not "
            "remotely — the session file is on the console's disk.",
            "The night engineer's booth mic was live on the studio "
            "intercom from 02:50 to 03:40; the board log is unbroken.",
            "The program director was in the office above the booth; "
            "the office camera caught her at 02:59, 03:10 and 03:31.",
            "The freelance DJ was at a hospital visit; the nurse's log "
            "shows her arrival at 03:00.",
            "The superintendent's key fob logged the studio door at "
            "02:58 — and his radio, off its charger, was found in the "
            "studio bin.",
        ],
        "statements": {
            "the program director": "I was on the phone with the board "
                                    "in the office. The camera is "
                                    "there too.",
            "the night engineer": "My booth was live on the intercom "
                                  "the whole time. Check the board.",
            "the freelance DJ": "I was at the hospital. The nurse's log "
                                "says 03:00.",
            "the building superintendent": "My fob goes into the boiler "
                                           "room, never the studio. The "
                                           "radio's been off its charger "
                                           "a week — I asked for a new "
                                           "one.",
        },
    },
    # ── 8 ────────────────────────────────────────────────────────────
    {
        "story": "A dealer bag was found with a marked deck after the "
                 "last hand of the night. The bag is locked; the key "
                 "stays with the floor manager.",
        "suspects": ["the floor manager", "the head dealer",
                     "the pit boss", "the security captain"],
        "culprit": "the pit boss",
        "clues": [
            "The deck was swapped with the bag shut — the bag's seal "
            "tape is unbroken.",
            "The floor manager's key log shows one checkout, one "
            "return, and the cage camera shows the bag never left his "
            "line of sight.",
            "The head dealer shuffled the same bag at the table on "
            "camera from 01:00 to 02:00.",
            "The security captain was on the roof camera circuit from "
            "02:00 to 03:00; the loop is timestamped.",
            "In the pit boss's office: a card case with the same "
            "marking ink, and a print-shop receipt for 'custom "
            "overlays, 2 boxes'.",
        ],
        "statements": {
            "the floor manager": "The key log and the camera agree. The "
                                 "bag never left my sight.",
            "the head dealer": "I was shuffling that bag at the table, "
                               "on camera, all night.",
            "the pit boss": "I was at the table until my shift ended. I "
                            "have never been in the cage.",
            "the security captain": "Roof camera circuit, 02:00 to "
                                    "03:00. The loop is there.",
        },
    },
    # ── 9 ────────────────────────────────────────────────────────────
    {
        "story": "A painting sold at 18:00 was 'resold' at 21:00 to a "
                 "second buyer, and the first invoice was paid twice — "
                 "a double dip. Only four could have forged the second "
                 "sale.",
        "suspects": ["the lead auctioneer", "the registrar", "the "
                     "private client's secretary", "the security chief"],
        "culprit": "the registrar",
        "clues": [
            "The second sale certificate carries the first "
            "certificate's lot barcode — it was printed from the same "
            "template.",
            "The lead auctioneer was at the podium; the broadcast feed "
            "and the house clock agree to the minute.",
            "The secretary's phone shows a live call with the client "
            "from 20:40 to 21:30. The call log is unbroken.",
            "The security chief was on the loading-dock camera until "
            "21:15, then in the control room; both feeds are "
            "timestamped.",
            "In the registrar's drawer: the template file, opened at "
            "19:22, with the lot barcode still embedded.",
        ],
        "statements": {
            "the lead auctioneer": "The broadcast has my every word, "
                                   "with the house clock on screen.",
            "the registrar": "I closed the catalog at 19:00 and locked "
                             "the drawer. Nothing was printed after "
                             "that.",
            "the private client's secretary": "I was on the phone with "
                                              "the client the whole "
                                              "time. The log is "
                                              "there.",
            "the security chief": "Dock until 21:15, control room "
                                  "after. Both feeds.",
        },
    },
    # ── 10 ───────────────────────────────────────────────────────────
    {
        "story": "The prize orchid that was to be exhibited was dead by "
                 "06:00 — cut at the base and dressed to look like "
                 "disease. Someone with the grow-room key and a knife "
                 "did it.",
        "suspects": ["the head grower", "the horticulture intern",
                     "the judge", "the delivery driver"],
        "culprit": "the horticulture intern",
        "clues": [
            "The cut was dressed with a fungicide the grow room does "
            "not stock — the bottle cap matches the intern's bench.",
            "The judge's car was on the gate camera from 05:40 to "
            "06:10, and the bloom was already dead when she entered.",
            "The delivery driver's van GPS pinged the depot until "
            "04:50; the cut was dressed by 05:30.",
            "The head grower was at a growers' conference; the hotel "
            "entry log and the coffee receipts agree.",
            "At the intern's bench: a blade with the same cut width, "
            "and a receipt for the fungicide from a garden centre two "
            "towns over.",
        ],
        "statements": {
            "the head grower": "I was at the conference. The entry log "
                               "says so.",
            "the horticulture intern": "I'm not in the grow room before "
                                       "07:00. The fungicide? I've "
                                       "never bought it.",
            "the judge": "I arrived at 05:45 and the bloom was already "
                         "gone.",
            "the delivery driver": "We were at the depot until 04:50. "
                                   "The GPS can show it.",
        },
    },
    # ── 11 ───────────────────────────────────────────────────────────
    {
        "story": "The lighthouse's first-order lens cracked from the "
                 "inside during the storm — a stone from the gallery "
                 "rail, thrown by someone on the island.",
        "suspects": ["the keeper", "the assistant keeper", "the supply "
                     "boat pilot", "the island ranger"],
        "culprit": "the assistant keeper",
        "clues": [
            "The fracture pattern shows the impact came from below the "
            "gallery — only the assistant had the gallery key that "
            "night.",
            "The keeper's log has hourly entries, and the supply "
            "boat's radar ping puts it at sea at the moment of impact.",
            "The supply boat pilot's vessel left at 19:00; the radar "
            "shows the impact at 22:30.",
            "The island ranger's cabin was dark — the substation log "
            "shows a trip at 19:30 that no one reset.",
            "In the assistant keeper's locker: a stone the same size "
            "as the one recovered from the gallery, and a second one, "
            "already broken.",
        ],
        "statements": {
            "the keeper": "My log is hourly. The radar has the boat's "
                          "position at 22:30.",
            "the assistant keeper": "I locked the gallery at 18:00 and "
                                    "left the key in the keeper's box. "
                                    "I was in my cabin.",
            "the supply boat pilot": "We left at 19:00. The radar "
                                     "knows.",
            "the island ranger": "My power was dead after 19:30. The "
                                 "substation log is there.",
        },
    },
    # ── 12 ───────────────────────────────────────────────────────────
    {
        "story": "A false floor panel was lifted from the bank vault's "
                 "corner office — one swipe with the vault master key, "
                 "and the panel was gone before the night audit.",
        "suspects": ["the vault manager", "the branch teller", "the "
                     "armored truck driver", "the building electrician"],
        "culprit": "the armored truck driver",
        "clues": [
            "The panel was carried out in a roll cage — the cage has "
            "the bank's logo, and its wheel tread matches the drive.",
            "The vault manager's key log shows one swipe in, one out, "
            "and the cage camera shows the panel in place at the 02:00 "
            "audit.",
            "The teller was on the lobby camera from 00:00 to 04:00; "
            "the loop is unbroken.",
            "The electrician's work order covers the east wing; the "
            "vault door's access log has no electrician fob that "
            "night.",
            "In the armored truck driver's cage: a panel cut to the "
            "vault's dimensions, and a swipe record on his fob at "
            "01:47.",
        ],
        "statements": {
            "the vault manager": "One in, one out, and the panel was in "
                                 "place at the 02:00 audit. The log "
                                 "says so.",
            "the branch teller": "Lobby camera, 00:00 to 04:00. The "
                                 "loop is there.",
            "the armored truck driver": "The cage was full of cash all "
                                        "night. The panel? I've never "
                                        "seen it.",
            "the building electrician": "East wing only. My fob never "
                                        "came near the vault.",
        },
    },
    # ── 13 ───────────────────────────────────────────────────────────
    {
        "story": "A reel of the 1968 premiere was found with a dubbed "
                 "scene — swapped for a frame-identical fake. Only four "
                 "people touched the reel after the screening.",
        "suspects": ["the projectionist", "the archivist", "the film "
                     "festival director", "the security chief"],
        "culprit": "the film festival director",
        "clues": [
            "The fake reel's splice tape matches the director's "
            "editing suite — the tape stock is discontinued "
            "elsewhere.",
            "The projectionist ran the reel on camera from 19:00 to "
            "21:00; the booth log agrees.",
            "The archivist's vault access log shows one entry at 18:00 "
            "and one exit at 18:40; the reel was intact on the shelf "
            "camera.",
            "The security chief was on the gallery camera until 21:30; "
            "the loop is timestamped.",
            "In the director's editing suite: a splice bench with the "
            "same tape, and a note that reads 'premiere reel — swap "
            "after screening'.",
        ],
        "statements": {
            "the projectionist": "The booth log has my every cue. The "
                                 "reel was mine until I gave it back.",
            "the archivist": "One in, one out, 18:00 to 18:40. The "
                             "shelf camera has the reel intact.",
            "the film festival director": "I was in the hall for the "
                                          "whole premiere. The editing "
                                          "suite was locked.",
            "the security chief": "Gallery until 21:30. The loop is "
                                  "there.",
        },
    },
    # ── 14 ───────────────────────────────────────────────────────────
    {
        "story": "A barrel of the reserve vintage was swapped for a "
                 "barrel of the reserve's lower cuvee, and the cork "
                 "was re-waxed to look original. Only four had the "
                 "cellar key.",
        "suspects": ["the cellar master", "the enology intern",
                     "the sommelier", "the delivery driver"],
        "culprit": "the enology intern",
        "clues": [
            "The wax on the swapped barrel is a fresh batch — the "
            "stamp die matches the intern's lab.",
            "The cellar master's key log shows one cellar entry at "
            "09:00 and one exit at 10:00; the barrel was intact on "
            "the rack camera.",
            "The sommelier was at a tasting two towns over; the venue "
            "guest list has her signature.",
            "The delivery driver's van GPS pinged the loading dock "
            "until 11:00; the swap was detected at 13:00.",
            "In the intern's lab: a wax batch with the same stamp, and "
            "a receipt for 'vintage label stock' from a paper "
            "supplier.",
        ],
        "statements": {
            "the cellar master": "One in, one out, 09:00 to 10:00. The "
                                 "rack camera has the barrel intact.",
            "the enology intern": "I was in the lab all morning. The "
                                  "wax batch? That's for the cuvee, "
                                  "not the reserve.",
            "the sommelier": "I was at the tasting. The guest list has "
                             "my signature.",
            "the delivery driver": "Dock until 11:00. The GPS can show "
                                   "it.",
        },
    },
    # ── 15 ───────────────────────────────────────────────────────────
    {
        "story": "A photographic plate from the 1954 comet survey was "
                 "scraped — one star, erased. Only four had the archive "
                 "vault key.",
        "suspects": ["the archive keeper", "the visiting professor",
                     "the graduate student", "the security chief"],
        "culprit": "the graduate student",
        "clues": [
            "The scrape is under the emulsion — done with a needle, "
            "not a blade. The needle's gauge matches the graduate "
            "student's kit.",
            "The archive keeper's vault log shows one entry at 14:00 "
            "and one exit at 15:00; the plate was intact on the shelf "
            "camera.",
            "The visiting professor's flight landed at 18:00; the "
            "scrape was detected at 15:30.",
            "The security chief was on the control-room camera until "
            "16:00; the loop is timestamped.",
            "In the graduate student's kit: a needle with the same "
            "wear, and a note that reads 'plate 1954-07 — comet "
            "trail — verify against original'.",
        ],
        "statements": {
            "the archive keeper": "One in, one out, 14:00 to 15:00. "
                                  "The shelf camera has the plate "
                                  "intact.",
            "the visiting professor": "My flight landed at 18:00. The "
                                      "scrape was before I arrived.",
            "the graduate student": "I've only looked at the plate "
                                    "through the viewer. The needle's "
                                    "for the microscope, not for "
                                    "plates.",
            "the security chief": "Control room until 16:00. The loop "
                                  "is there.",
        },
    },
    # ── 16 ───────────────────────────────────────────────────────────
    {
        "story": "The prize loaf from the morning bake was swapped for "
                 "a decoy, and the swap was made with a knife that cut "
                 "the display case seal. Only four had the case key.",
        "suspects": ["the head baker", "the morning assistant",
                     "the judge", "the delivery driver"],
        "culprit": "the morning assistant",
        "clues": [
            "The display case seal was cut with a knife — the cut "
            "pattern matches the morning assistant's bread knife.",
            "The head baker's ovens ran from 04:00 to 06:00; the oven "
            "log agrees with the judge's arrival at 06:30.",
            "The judge's car was on the gate camera from 06:30 to "
            "07:00; the swap was detected at 07:15.",
            "The delivery driver's van GPS pinged the depot until "
            "05:00; the swap was made by 05:45.",
            "At the morning assistant's bench: a knife with the same "
            "cut pattern, and a decoy loaf under the bench, still "
            "warm.",
        ],
        "statements": {
            "the head baker": "Ovens from 04:00 to 06:00. The log "
                              "agrees with the judge's 06:30 arrival.",
            "the morning assistant": "I was at the proofing bench all "
                                     "morning. The knife's for bread, "
                                     "not for seals.",
            "the judge": "I arrived at 06:30 and the loaf looked "
                         "right.",
            "the delivery driver": "Depot until 05:00. The GPS can "
                                   "show it.",
        },
    },
    # ── 17 ───────────────────────────────────────────────────────────
    {
        "story": "A first-class cabin was ransacked on the night "
                 "express, and the cabin was locked from the inside. "
                 "Only four had the master override.",
        "suspects": ["the conductor", "the dining car steward",
                     "the ticket inspector", "the security officer"],
        "culprit": "the ticket inspector",
        "clues": [
            "The cabin lock was picked, not overridden — the pick "
            "marks are on the bolt, not the override plate.",
            "The conductor's manifest has hourly signatures, and the "
            "cabin was intact on his 22:00 round.",
            "The dining car steward served the dining car from 21:00 "
            "to 23:00; the service log agrees.",
            "The security officer was on the corridor camera from "
            "23:00 to 01:00; the loop is timestamped.",
            "In the ticket inspector's bag: a pick set with the same "
            "wear, and a cabin key card that was never issued to him.",
        ],
        "statements": {
            "the conductor": "Hourly signatures. The cabin was intact "
                             "on my 22:00 round.",
            "the dining car steward": "I served the dining car from "
                                      "21:00 to 23:00. The service "
                                      "log is there.",
            "the ticket inspector": "I was checking tickets all night. "
                                    "A pick set? I've never had one.",
            "the security officer": "Corridor camera, 23:00 to 01:00. "
                                    "The loop is there.",
        },
    },
    # ── 18 ───────────────────────────────────────────────────────────
    {
        "story": "A page was torn from a first edition in the special "
                 "collections reading room, and the tear was made with "
                 "a bone folder. Only four had the reading room key "
                 "that night.",
        "suspects": ["the special collections librarian", "the graduate "
                     "conservator", "the bookbinder", "the security "
                     "chief"],
        "culprit": "the bookbinder",
        "clues": [
            "The tear is clean, a single pass — made with a bone "
            "folder, not a blade. The folder's width matches the "
            "bookbinder's kit.",
            "The special collections librarian's access log shows one "
            "entry at 20:00 and one exit at 21:00; the volume was "
            "intact on the case camera.",
            "The graduate conservator was in the lab two floors down; "
            "the lab camera has her until 22:00.",
            "The security chief was on the reading room camera until "
            "21:30; the loop is timestamped.",
            "At the bookbinder's bench: a bone folder with the same "
            "wear, and a torn edge matching the missing page's "
            "gutter.",
        ],
        "statements": {
            "the special collections librarian": "One in, one out, "
                                                 "20:00 to 21:00. The "
                                                 "case camera has the "
                                                 "volume intact.",
            "the graduate conservator": "Lab camera, until 22:00. I "
                                        "was two floors down.",
            "the bookbinder": "I was at my bench all night. The "
                              "folder's for binding, not for tearing.",
            "the security chief": "Reading room camera until 21:30. "
                                  "The loop is there.",
        },
    },
    # ── 19 ───────────────────────────────────────────────────────────
    {
        "story": "A locked display case was opened during the evening "
                 "gallery — one swipe with the case key, and the case "
                 "was re-sealed before the night guard's round.",
        "suspects": ["the gallery curator", "the night guard",
                     "the docent", "the security chief"],
        "culprit": "the docent",
        "clues": [
            "The case lock shows a swipe, not a pry — the strike "
            "plate is clean.",
            "The gallery curator's key log shows one case entry at "
            "18:00 and one exit at 18:30; the case was intact on the "
            "gallery camera at the 19:00 round.",
            "The night guard's rounds log has a signature at 19:30, "
            "20:30 and 21:30; the case was intact in the guard's "
            "tablet photos.",
            "The security chief was on the control-room camera until "
            "22:00; the loop is timestamped.",
            "In the docent's bag: a case key that was never issued to "
            "her, and a photo of the case contents timestamped at "
            "19:45.",
        ],
        "statements": {
            "the gallery curator": "One in, one out, 18:00 to 18:30. "
                                   "The gallery camera has the case "
                                   "intact at 19:00.",
            "the night guard": "My rounds are signed, and the tablet "
                               "photos show the case intact.",
            "the docent": "I led the 19:00 tour and left at 19:30. A "
                          "key? I've never had one.",
            "the security chief": "Control room until 22:00. The loop "
                                  "is there.",
        },
    },
    # ── 20 ───────────────────────────────────────────────────────────
    {
        "story": "A boat's engine was lifted from its cradle at the "
                 "boatyard overnight — one lift with the yard crane, "
                 "and the crane log shows only the yard's operator.",
        "suspects": ["the yard foreman", "the crane operator", "the "
                     "yard electrician", "the harbor pilot"],
        "culprit": "the crane operator",
        "clues": [
            "The crane log shows one lift at 02:00 — from the "
            "operator's station, not the remote.",
            "The yard foreman's office camera has him at his desk "
            "from 21:00 to 03:00; the loop is unbroken.",
            "The yard electrician's work order covers the fuel dock; "
            "the crane station's access log has no electrician fob.",
            "The harbor pilot's vessel left at 19:00; the radar shows "
            "the yard clear at 02:00.",
            "In the crane operator's locker: a crane remote that was "
            "never issued to him, and a swipe record on his fob at "
            "01:58.",
        ],
        "statements": {
            "the yard foreman": "Office camera, 21:00 to 03:00. The "
                                "loop is there.",
            "the crane operator": "The crane sat idle all night. A "
                                  "remote? I've never had one.",
            "the yard electrician": "Fuel dock only. My fob never came "
                                    "near the crane station.",
            "the harbor pilot": "We left at 19:00. The radar has the "
                                "yard clear at 02:00.",
        },
    },
    # ── 21 ──────────────────────────────────────────────────────────
    {
        "story": "The Stradivarius vanished from the concert hall's "
                 "green room during intermission. The room locks from "
                 "the inside; four people had the spare key.",
        "suspects": ["the concertmaster", "the page turner",
                     "the stage manager", "the instrument tech"],
        "culprit": "the page turner",
        "clues": [
            "The spare key was used at 20:52 — the lock log shows one "
            "clean turn, no force.",
            "The concertmaster was on stage; the broadcast feed has "
            "her in frame from 20:40 to 21:10.",
            "The stage manager's radio log shows continuous check-ins "
            "from the wings until 21:05.",
            "The instrument tech was seen polishing a cello in the "
            "workshop at 20:52 — on the workshop camera, timestamped.",
            "In the page turner's satchel: a rosin cloth monogrammed "
            "with the soloist's initials, and a pawn ticket dated "
            "that night.",
        ],
        "statements": {
            "the concertmaster": "I was on stage the whole intermission. "
                                 "The broadcast feed proves it.",
            "the page turner": "I was fetching scores from the library. "
                               "A satchel? I carry scores, not violins.",
            "the stage manager": "My radio never went quiet. Check the log.",
            "the instrument tech": "I was polishing the cello in the "
                                   "workshop. The camera was on.",
        },
    },
    # ── 22 ──────────────────────────────────────────────────────────
    {
        "story": "The data center's backup tapes were swapped for "
                 "blanks during the night shift. The tape library "
                 "needs a badge and a PIN.",
        "suspects": ["the night-shift operator", "the facilities "
                     "engineer", "the courier", "the security analyst"],
        "culprit": "the night-shift operator",
        "clues": [
            "The library door logged a badge plus PIN at 02:14 — the "
            "PIN pad shows no smudge pattern, entered cleanly.",
            "The facilities engineer was on the roof unit from 01:00 "
            "to 03:00; the work order GPS agrees.",
            "The courier's van was at the gate at 02:14 — the gate "
            "camera shows the van, driver inside, engine running.",
            "The security analyst was in the SOC; the session log has "
            "keystrokes every minute until 04:00.",
            "In the operator's locker: a tape label printer ribbon "
            "with the backup set's barcodes, still warm.",
        ],
        "statements": {
            "the night-shift operator": "I was at the console all night. "
                                        "The tapes? I never touched the "
                                        "library.",
            "the facilities engineer": "Roof unit, 01:00 to 03:00. The "
                                       "work order has the GPS.",
            "the courier": "I was at the gate at 02:14, engine running. "
                           "The camera saw me.",
            "the security analyst": "SOC session log, keystrokes every "
                                    "minute. I never left.",
        },
    },
    # ── 23 ──────────────────────────────────────────────────────────
    {
        "story": "The favorite's saddle was swapped for a weighted "
                 "replica before the derby. The tack room was locked; "
                 "four had the combination.",
        "suspects": ["the head groom", "the stablehand", "the jockey's "
                     "agent", "the track vet"],
        "culprit": "the stablehand",
        "clues": [
            "The replica weighs 4 kg more — the swap needed the "
            "tack room's hanging scale, used at 05:20.",
            "The head groom was at the feed store; the receipt is "
            "timestamped 05:15, twenty minutes away.",
            "The jockey's agent was on a recorded call with the "
            "stewards from 05:00 to 05:40.",
            "The track vet was drawing blood in barn C; the lab "
            "requisition is timed 05:25.",
            "In the stablehand's trunk: lead sheeting cut to saddle "
            "panels, and the favorite's real stirrup leathers.",
        ],
        "statements": {
            "the head groom": "Feed store at 05:15. The receipt is "
                              "twenty minutes from the track.",
            "the stablehand": "I mucked stalls all morning. Lead "
                              "sheeting? That's for the roof repairs.",
            "the jockey's agent": "I was on with the stewards, recorded, "
                                  "05:00 to 05:40.",
            "the track vet": "Barn C, drawing blood. The lab "
                             "requisition is timed.",
        },
    },
    # ── 24 ──────────────────────────────────────────────────────────
    {
        "story": "The one-penny black was lifted from the philately "
                 "exhibition's case during the members' hour. The case "
                 "opens with a key and a code.",
        "suspects": ["the exhibit designer", "the society president",
                     "the case maker", "the evening guard"],
        "culprit": "the exhibit designer",
        "clues": [
            "The case was opened with key AND code — the code log "
            "shows the designer's personal code at 19:12.",
            "The society president was giving the opening address; "
            "two hundred members heard it.",
            "The case maker's van broke down on the motorway; the "
            "recovery invoice is timed 18:40.",
            "The evening guard's rounds log has the case sealed at "
            "19:00 and 19:30 — both with tablet photos.",
            "In the designer's flat: a stock book with the "
            "one-penny black hinged in, and the exhibition's spare "
            "case key.",
        ],
        "statements": {
            "the exhibit designer": "I set the case at noon and never "
                                    "went back. My code? Someone must "
                                    "have watched me type it.",
            "the society president": "Two hundred members heard my "
                                     "address. I was on the podium.",
            "the case maker": "My van died on the motorway. The "
                              "recovery invoice is timed.",
            "the evening guard": "Rounds logged, photos taken. The case "
                                 "was sealed at 19:00 and 19:30.",
        },
    },
    # ── 25 ──────────────────────────────────────────────────────────
    {
        "story": "The chef's recipe book was photographed page by "
                 "page in the locked office during service. Only four "
                 "had the office key.",
        "suspects": ["the sous-chef", "the line cook", "the restaurant "
                     "critic", "the delivery driver"],
        "culprit": "the line cook",
        "clues": [
            "The office camera was unplugged at 20:03 — the plug shows "
            "a clean pull, someone who knew the blind spot.",
            "The sous-chef was on the pass; the ticket rail has her "
            "initials on every order until 22:00.",
            "The critic filed her review from the dining room at "
            "20:30; the timestamp is on the submission.",
            "The delivery driver's route log shows the restaurant "
            "drop at 19:40 and the next stop at 20:10.",
            "On the line cook's phone: 47 photos of handwritten "
            "recipe pages, taken 20:04 to 20:19.",
        ],
        "statements": {
            "the sous-chef": "I was on the pass all service. The ticket "
                             "rail has my initials.",
            "the line cook": "I was on grill all night. Photos? I "
                             "photograph plating for my portfolio.",
            "the restaurant critic": "I filed from the dining room at "
                                     "20:30. The submission is timestamped.",
            "the delivery driver": "Drop at 19:40, next stop 20:10. The "
                                   "route log is there.",
        },
    },
    # ── 26 ──────────────────────────────────────────────────────────
    {
        "story": "A logbook page covering the comet's closest approach "
                 "was razored out of the observatory's bound volume. "
                 "Four had the dome key that night.",
        "suspects": ["the resident astronomer", "the visiting "
                     "researcher", "the telescope tech", "the night "
                     "watchman"],
        "culprit": "the telescope tech",
        "clues": [
            "The cut is a single razor pass — the blade width matches "
            "the tech's box cutter, not the archive scalpel.",
            "The resident astronomer was guiding a school group; the "
            "booking sheet and thirty students agree.",
            "The visiting researcher's hire car GPS shows the "
            "motorway services from 23:00 to 01:00.",
            "The night watchman's rounds log has the dome locked at "
            "midnight and 02:00, both with photos.",
            "In the tech's toolbox: a box cutter with paper dust in "
            "the slide, and a folded logbook page in a star chart.",
        ],
        "statements": {
            "the resident astronomer": "School group all night. Thirty "
                                       "students and a booking sheet.",
            "the visiting researcher": "I was at the motorway services. "
                                        "The hire car GPS knows.",
            "the telescope tech": "I was collimating the secondary all "
                                  "night. The cutter's for opening "
                                  "boxes.",
            "the night watchman": "Dome locked at midnight and 02:00. "
                                  "The photos are logged.",
        },
    },
    # ── 27 ──────────────────────────────────────────────────────────
    {
        "story": "The high-denomination chip mold went missing from "
                 "the casino's cage during the count. The cage needs "
                 "two keys turned together.",
        "suspects": ["the cage cashier", "the count supervisor",
                     "the chip runner", "the surveillance operator"],
        "culprit": "the cage cashier",
        "clues": [
            "The mold left in a chip rack — the rack's RFID logged "
            "the cage door at 03:12, then nothing.",
            "The count supervisor was in the count room; the room "
            "camera and the scale logs agree to the minute.",
            "The chip runner's route sheet has the floor run "
            "timestamped 03:10 to 03:25, signed by two dealers.",
            "The surveillance operator's console log shows the cage "
            "camera feed live on her monitor all night.",
            "In the cashier's car: a chip rack with the mold's "
            "serial etched inside, wrapped in a cage towel.",
        ],
        "statements": {
            "the cage cashier": "I was balancing the drawer all night. "
                                "A rack in my car? That's the laundry "
                                "run.",
            "the count supervisor": "Count room, on camera, scale logs "
                                    "to the minute.",
            "the chip runner": "Floor run 03:10 to 03:25, signed by two "
                               "dealers.",
            "the surveillance operator": "The cage feed was on my monitor "
                                         "all night. The console log "
                                         "proves it.",
        },
    },
    # ── 28 ──────────────────────────────────────────────────────────
    {
        "story": "The original negative of the finale was swapped for "
                 "a dupe in the editing suite overnight. The suite "
                 "needs a fob after hours.",
        "suspects": ["the lead editor", "the assistant editor",
                     "the colorist", "the night cleaner"],
        "culprit": "the assistant editor",
        "clues": [
            "The dupe's edge code is one generation off — swapped by "
            "someone who knew which can to take.",
            "The lead editor's fob logged the parking garage at "
            "23:40; the garage camera agrees.",
            "The colorist was rendering at home; the render farm log "
            "has her jobs queued until 04:00.",
            "The night cleaner's cart GPS shows the third floor from "
            "01:00 to 02:00, never the suite.",
            "In the assistant editor's drawer: the original negative "
            "in a mislabeled can, and a pawn ticket for camera gear.",
        ],
        "statements": {
            "the lead editor": "I left at 23:40. The garage camera saw "
                               "my fob.",
            "the assistant editor": "I was logging footage at my "
                                     "station. The negative? I've never "
                                     "opened that can.",
            "the colorist": "Rendering from home. The farm log has my "
                            "jobs until 04:00.",
            "the night cleaner": "Third floor all night. The cart GPS "
                                 "never went near the suite.",
        },
    },
    # ── 29 ──────────────────────────────────────────────────────────
    {
        "story": "The 17th-century globe was lifted from the library's "
                 "map room during the storm. The room was locked; four "
                 "had the master.",
        "suspects": ["the map curator", "the night shelver", "the "
                     "restoration volunteer", "the security guard"],
        "culprit": "the night shelver",
        "clues": [
            "The globe's cradle was unscrewed, not forced — a "
            "screwdriver job by someone who knew the mount.",
            "The map curator was at a donors' dinner; the seating "
            "chart and photos agree.",
            "The restoration volunteer's timesheet ends at 17:00; the "
            "exit gate logged her out at 17:04.",
            "The security guard's rounds put him in the east wing "
            "during the storm; the log is timestamped.",
            "In the shelver's cart: brass screws matching the cradle, "
            "and a shipping label made out to a private collector.",
        ],
        "statements": {
            "the map curator": "Donors' dinner all evening. The seating "
                               "chart has me.",
            "the night shelver": "I shelved returns all night. Screws? "
                                 "From the book truck repairs.",
            "the restoration volunteer": "Out at 17:04. The gate logged "
                                          "me.",
            "the security guard": "East wing during the storm. The "
                                  "rounds log is timestamped.",
        },
    },
    # ── 30 ──────────────────────────────────────────────────────────
    {
        "story": "The race leader's timing chip was swapped at the "
                 "marathon's halfway point, erasing her split. Four "
                 "marshals worked that station.",
        "suspects": ["the station chief", "the course marshal",
                     "the timing tech", "the water volunteer"],
        "culprit": "the course marshal",
        "clues": [
            "The chip was swapped, not lost — the dead chip was found "
            "in the station's bin, wiped.",
            "The station chief was on the radio net; the net log has "
            "her voice every five minutes.",
            "The timing tech's laptop shows the swap as a manual "
            "override from the station terminal at 09:12.",
            "The water volunteer was photographed handing out cups "
            "at 09:12, two stations down.",
            "In the marshal's vest: the leader's live chip, still "
            "pinging, tucked in the inner pocket.",
        ],
        "statements": {
            "the station chief": "Radio net all morning, every five "
                                 "minutes. The log has my voice.",
            "the course marshal": "I was directing runners all morning. "
                                  "A chip in my vest? Must have fallen "
                                  "in.",
            "the timing tech": "The override came from the station "
                               "terminal, not my laptop.",
            "the water volunteer": "I was two stations down at 09:12. "
                                   "The photos show it.",
        },
    },
)

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



# ══════════════════════════════════════════════════════════════════════
# Bank enrichment + god-tier case services.
#
# _RAW_CASES (the 30 originals) get their god-tier metadata from
# _BANK_META; _NEW_CASES already carry the full schema.  Every case in
# CASES therefore has: id, title, tier, briefing, story, suspects,
# motives, culprit, clues (ordered), red_herrings, herring_suspect,
# solution, statements, difficulty (back-compat), generated=False.
# ══════════════════════════════════════════════════════════════════════

def _sig_words(name: str) -> list[str]:
    """Significant words of a suspect name (for fair-play matching)."""
    words = []
    for w in (name or "").split():
        w = w.strip(".,'\"").lower()
        if w != "the" and len(w) >= 3:
            words.append(w)
    return words


def _full_name(name: str) -> str:
    """Suspect name without the leading 'the' — for implication checks."""
    n = (name or "").strip().lower()
    if n.startswith("the "):
        n = n[4:]
    return n


def _herring_suspect(case: dict[str, Any]) -> str:
    """Which innocent the red-herring clue (clues[3]) points at."""
    blob = (case.get("clues") or [""] * 4)[3].lower()
    for s in case.get("suspects") or ():
        if s == case.get("culprit"):
            continue
        if any(w in blob for w in _sig_words(s)):
            return s
    return ""


def _enrich() -> tuple[dict[str, Any], ...]:
    out: list[dict[str, Any]] = []
    raws = list(_RAW_CASES) + list(_NEW_CASES)
    for i, raw in enumerate(raws):
        case = dict(raw)
        if i < len(_BANK_META):
            for key in ("title", "tier", "briefing", "motives",
                        "red_herrings", "solution"):
                case[key] = _BANK_META[i][key]
        tier = case.get("tier") or "medium"
        case["tier"] = tier
        case["id"] = f"bank:{i}"
        # back-compat grade for older scoring paths (expert → hard)
        case["difficulty"] = tier if tier in ("easy", "medium", "hard") \
            else "hard"
        case["herring_suspect"] = _herring_suspect(case)
        case["generated"] = False
        case["suspects"] = list(case.get("suspects") or ())
        case["clues"] = list(case.get("clues") or ())
        case["statements"] = dict(case.get("statements") or {})
        case["motives"] = dict(case.get("motives") or {})
        case["red_herrings"] = list(case.get("red_herrings") or ())
        out.append(case)
    return tuple(out)


CASES = _enrich()

# ── how many hand-written cases the bank holds (the generator adds more) ──
BANK_SIZE = len(CASES)


def _copy_case(case: dict[str, Any]) -> dict[str, Any]:
    """A mutable deep-enough copy: the game adapts clues per player."""
    c = dict(case)
    c["suspects"] = list(case.get("suspects") or ())
    c["clues"] = list(case.get("clues") or ())
    c["statements"] = dict(case.get("statements") or {})
    c["motives"] = dict(case.get("motives") or {})
    c["red_herrings"] = list(case.get("red_herrings") or ())
    return c


# ── fair-play solvability ─────────────────────────────────────────────────

_CONFESS_MARKERS = ("i did it", "i'm guilty", "i confess", "it was me",
                    "my fault", "i'll admit")


def verify_solvability(case: dict[str, Any]) -> list[str]:
    """Fair-play check: can the culprit be deduced from the clues?

    Returns a list of problems; empty means the case is solvable and
    fair: the smoking gun names the culprit, at least two innocents
    are cleared by alibis, the red herring points at an innocent (never
    the culprit), everyone has a motive and a statement, and the
    solution walks the deduction.
    """
    problems: list[str] = []
    suspects = list(case.get("suspects") or ())
    culprit = case.get("culprit")
    if len(suspects) != 4 or len(set(suspects)) != 4:
        problems.append("need exactly 4 unique suspects")
    if culprit not in suspects:
        problems.append("culprit must be one of the suspects")
    for key in ("title", "tier", "briefing", "story", "motives", "clues",
                "red_herrings", "solution", "statements"):
        if not case.get(key):
            problems.append(f"missing/empty {key}")
    if case.get("tier") not in TIERS:
        problems.append("tier must be one of easy/medium/hard/expert")
    motives = case.get("motives") or {}
    for s in suspects:
        if not (motives.get(s) or "").strip():
            problems.append(f"no motive for {s}")
    clues = list(case.get("clues") or ())
    if len(clues) < 5:
        problems.append("need at least 5 ordered clues")
    cwords = _sig_words(culprit or "")
    cname = _full_name(culprit or "")
    if clues and not any(w in clues[-1].lower() for w in cwords):
        problems.append("smoking gun (last clue) must name the culprit")
    # adapted cases splice extra clues in before the smoking gun, so
    # the classic five keep their order and the gun still closes the file
    innocents = [s for s in suspects if s != culprit]
    if len(clues) >= 5:
        for idx in (1, 2):
            blob = clues[idx].lower()
            cleared = [s for s in innocents
                       if any(w in blob for w in _sig_words(s))]
            if not cleared:
                problems.append(f"clue {idx + 1} (clears an innocent) "
                                "names no innocent")
        herring_blob = clues[3].lower()
        # full-name phrase: incidental shared words ("cage camera" vs
        # "cage cashier") must not count as implicating the culprit
        if cname and cname in herring_blob:
            problems.append("red-herring clue must not name the culprit")
        if not any(any(w in herring_blob for w in _sig_words(s))
                   for s in innocents):
            problems.append("red-herring clue should point at an innocent")
    for i, blob in enumerate([case.get("story") or ""] + clues):
        if "{" in blob or "}" in blob:
            problems.append(f"unfilled template slot in story/clue {i}")
        if not blob.strip():
            problems.append(f"empty story/clue {i}")
    statements = case.get("statements") or {}
    if set(statements) != set(suspects):
        problems.append("every suspect needs exactly one statement")
    cline = (statements.get(culprit) or "").lower()
    if any(m in cline for m in _CONFESS_MARKERS):
        problems.append("culprit must not confess in their statement")
    for name, line in statements.items():
        if "{" in line or "}" in line or not line.strip():
            problems.append(f"bad statement for {name}")
    for h in case.get("red_herrings") or ():
        hl = h.lower()
        if not any(any(w in hl for w in _sig_words(s)) for s in innocents):
            problems.append(f"red herring names no innocent: {h[:48]}")
        if cname and cname in hl:
            problems.append(f"red herring implicates the culprit: {h[:48]}")
    sol = case.get("solution") or ""
    if len(sol) < 40 or not any(w in sol.lower() for w in cwords):
        problems.append("solution must walk the deduction to the culprit")
    return problems


# ── per-player history: anti-repeat, skill, streaks ────────────────────────

def blank_history() -> dict[str, Any]:
    """A fresh per-player case record (JSON-safe, stored in the player
    profile's per_game['case'])."""
    return {
        "v": HISTORY_VERSION,
        "seen": {},            # tier -> [case ids already served]
        "attempts": {},        # tier -> cases played
        "solved": {},          # tier -> cases closed by this player
        "streak": 0,           # current consecutive solves
        "best_streak": 0,
        "reshuffles": 0,       # times a tier pool was exhausted
        "hints_taken": 0,
        "timed_solved": 0,
    }


def solve_rate(history: dict[str, Any], tier: str | None = None) -> float:
    """Rolling solve rate: solved / attempted. 0.5 (neutral) until the
    player has data, so new players get unmodified cases."""
    att = history.get("attempts") or {}
    sol = history.get("solved") or {}
    if tier is not None:
        a = int(att.get(tier) or 0)
        s = int(sol.get(tier) or 0)
    else:
        a = sum(int(v) for v in att.values())
        s = sum(int(v) for v in sol.values())
    return (s / a) if a else 0.5


def eligible_tiers(history: dict[str, Any]) -> tuple[str, ...]:
    """Tiers a player may draw from. New players start on easy/medium;
    hard unlocks at a 50%+ solve rate over 3+ cases, expert at 65%+
    over 6+."""
    attempts = sum(int(v) for v in (history.get("attempts") or {}).values())
    rate = solve_rate(history)
    tiers = ["easy", "medium"]
    if attempts >= 3 and rate >= 0.50:
        tiers.append("hard")
    if attempts >= 6 and rate >= 0.65:
        tiers.append("expert")
    return tuple(tiers)


def unseen_bank_cases(tier: str, seen_ids: Any) -> list[dict[str, Any]]:
    """Bank cases of a tier the player hasn't seen yet."""
    seen = set(seen_ids or ())
    return [c for c in CASES
            if c.get("tier") == tier and c.get("id") not in seen]


def deal_case(rng: Any, history: dict[str, Any]
              ) -> tuple[dict[str, Any], str, bool]:
    """Serve one case with per-player anti-repeat.

    Picks a tier from the player's eligible tiers, then an unseen bank
    case of that tier (with a ~25% sprinkle of fresh generated cases).
    When the tier's bank pool is exhausted, the pool reshuffles
    (``reshuffled=True``) and bank cases become eligible again.
    Applies skill-adaptive clue tuning before returning.
    """
    import random as _random

    if not isinstance(rng, _random.Random):
        rng = _random.Random(rng)
    tiers = eligible_tiers(history)
    tier = rng.choice(list(tiers))
    seen = history.setdefault("seen", {}).setdefault(tier, [])
    pool = unseen_bank_cases(tier, seen)
    reshuffled = False
    if not pool:
        pool = unseen_bank_cases(tier, [])
        reshuffled = True
    if pool and rng.random() < 0.75:
        case = _copy_case(rng.choice(pool))
    else:
        case = generate_case(rng, tier)
    return adapt_case(case, rng, history), tier, reshuffled


def record_played(history: dict[str, Any], case_id: str, tier: str,
                  solved: bool, reshuffled: bool = False) -> dict[str, Any]:
    """Fold one finished case into the player's history. Returns it."""
    attempts = history.setdefault("attempts", {})
    solved_map = history.setdefault("solved", {})
    attempts[tier] = int(attempts.get(tier) or 0) + 1
    seen = history.setdefault("seen", {}).setdefault(tier, [])
    if reshuffled:
        del seen[:]
        history["reshuffles"] = int(history.get("reshuffles") or 0) + 1
    if case_id and case_id not in seen:
        seen.append(case_id)
        if len(seen) > _SEEN_CAP:
            del seen[:len(seen) - _SEEN_CAP]
    if solved:
        solved_map[tier] = int(solved_map.get(tier) or 0) + 1
        history["streak"] = int(history.get("streak") or 0) + 1
        history["best_streak"] = max(int(history.get("best_streak") or 0),
                                     history["streak"])
    else:
        history["streak"] = 0
    history["v"] = HISTORY_VERSION
    return history


# ── skill-adaptive clue counts ─────────────────────────────────────────────

_ADAPT_HERRINGS = (
    "{s} was seen arguing with the victim's associate that week — over "
    "an old debt, settled since.",
    "A witness puts {s} near the scene that night — but the witness "
    "wore no watch and the hour is off.",
    "{s} called the victim that morning — an 11-second wrong number; "
    "the carrier log shows it.",
    "{s} asked about the case unprompted — curiosity, not guilt: every "
    "question was about public facts.",
)


def adapt_case(case: dict[str, Any], rng: Any,
               history: dict[str, Any]) -> dict[str, Any]:
    """Tune clue counts to the player's solve rate.

    Struggling players (sub-40% over 3+ cases) get one extra genuine
    clue — a background check on everyone's motives, inserted before
    the smoking gun. Strong players (75%+ over 5+ cases) get extra red
    herrings (two at 90%+) targeting innocents. Everyone else plays
    the case as written.
    """
    import random as _random

    if not isinstance(rng, _random.Random):
        rng = _random.Random(rng)
    case = _copy_case(case)
    rate = solve_rate(history)
    attempts = sum(int(v) for v in (history.get("attempts") or {}).values())
    innocents = [s for s in case["suspects"] if s != case["culprit"]]
    if attempts >= 3 and rate < 0.40:
        motives = case.get("motives") or {}
        bit = "; ".join(f"{s}: {motives.get(s) or 'no known motive'}"
                        for s in case["suspects"])
        assist = ("background check — everyone had a reason: " + bit +
                  ". means and opportunity still decide it.")
        case["clues"] = case["clues"][:-1] + [assist] + case["clues"][-1:]
        case["assisted"] = True
    elif attempts >= 5 and rate > 0.75:
        n = 2 if rate > 0.90 else 1
        pool = [s for s in innocents
                if s != case.get("herring_suspect")] or innocents
        for _ in range(n):
            text = rng.choice(_ADAPT_HERRINGS).format(s=rng.choice(pool))
            case["clues"] = case["clues"][:-1] + [text] + case["clues"][-1:]
            case["red_herrings"] = list(case.get("red_herrings") or []) + [text]
        case["sharpened"] = True
    return case


# ── scoring ───────────────────────────────────────────────────────────────

def score_solve(tier: str, strikes: int = 0, hints_used: int = 0,
                streak_in: int = 0, time_bonus: int = 0) -> int:
    """Points for closing a case.

    Base by tier, minus a point per wrong accusation and HINT_COST per
    hint, multiplied by the incoming solve streak (10% per streak step,
    capped at 2x), plus any timed-mode bonus. Always at least 1.
    """
    base = TIER_BASE_SCORE.get(tier, 5)
    mult = 1.0 + 0.1 * max(0, min(int(streak_in), 10))
    raw = (base - int(strikes) - HINT_COST * int(hints_used)) * mult
    return max(1, int(round(raw))) + max(0, int(time_bonus))


# ══════════════════════════════════════════════════════════════════════
# Dynamic case generator — seeded, combinatorial, airtight by construction.
#
# Every generated case follows the same 5-clue contract as the bank:
#   clue 1 = the mechanism (how it was done)
#   clue 2 = an alibi clearing innocent A
#   clue 3 = an alibi clearing innocent B
#   clue 4 = a red herring (suspicious-looking, provably innocent)
#   clue 5 = the smoking gun, naming the culprit by name
# The culprit's statement denies the smoking gun; every innocent's
# statement is consistent with their own clue. Because the pieces are
# assembled from matched (clue, statement) pairs, a generated case
# cannot contradict itself. Generated cases also carry the full
# god-tier schema (title, tier, briefing, motives, red_herrings,
# herring_suspect, solution).
#
# Difficulty: easy → blunt herring + blunt gun; hard/expert → subtle
# herring + subtle gun; medium → mixed. ~12 scenes × C(24,4) suspect
# sets × templates = millions of distinct, solvable cases.
# ══════════════════════════════════════════════════════════════════════

_GEN_ROLES = (
    "the night guard", "the curator", "the janitor", "the sous-chef",
    "the florist", "the valet", "the electrician", "the intern",
    "the archivist", "the stagehand", "the sommelier", "the locksmith",
    "the chauffeur", "the housekeeper", "the gardener", "the bartender",
    "the dispatcher", "the lab tech", "the docent", "the projectionist",
    "the bookkeeper", "the courier", "the radio operator",
    "the pastry chef",
)

_GEN_SCENES = (
    {"item": "first-edition map", "place": "the sealed vault",
     "lock": "keycard reader"},
    {"item": "championship trophy", "place": "the trophy room",
     "lock": "code lock"},
    {"item": "sapphire necklace", "place": "the gallery case",
     "lock": "case key"},
    {"item": "leather-bound ledger", "place": "the corner office",
     "lock": "master key"},
    {"item": "concert violin", "place": "the green room",
     "lock": "spare key"},
    {"item": "reserve wine barrel", "place": "the cellar",
     "lock": "cellar key"},
    {"item": "illuminated manuscript", "place": "the reading room",
     "lock": "reading-room key"},
    {"item": "studio camera", "place": "the equipment cage",
     "lock": "studio fob"},
    {"item": "pocket watch", "place": "the wall safe",
     "lock": "combination dial"},
    {"item": "oil painting", "place": "the storage wing",
     "lock": "badge reader"},
    {"item": "field laptop", "place": "the server room",
     "lock": "biometric pad"},
    {"item": "signet ring", "place": "the bridal suite",
     "lock": "door code"},
)

_GEN_PLACE2 = (
    "the lobby", "the east wing", "the loading dock", "the courtyard",
    "the far stairwell", "the service corridor",
)

_GEN_TIMES = (
    ("19:40", "20:15"), ("20:52", "21:30"), ("21:41", "22:20"),
    ("22:15", "23:00"), ("01:47", "02:30"), ("02:14", "03:00"),
    ("05:20", "06:00"),
)

_GEN_TOOLS = ("wrench", "torch", "screwdriver", "multitool")

# (clue template, statement template) — the statement always agrees
# with the clue, so innocents can never contradict themselves.
_GEN_ALIBIS = (
    ("{s} was on the {place2} camera from {t1} to {t2}; the loop is unbroken.",
     "I was on the camera the whole time. Check the loop."),
    ("{s}'s badge logged the far wing at {t1} — the reader log is timestamped.",
     "My badge never left the far wing. The reader log proves it."),
    ("{s} was on a live call from {t1} to {t2}; the call record is unbroken.",
     "I was on a live call the whole time. The record is unbroken."),
    ("Two witnesses place {s} at {place2} at {t1}; both statements agree.",
     "Two people saw me at {place2}. Ask them."),
    ("{s}'s van GPS pinged {place2} until {t1}; the tracker log is intact.",
     "My van never moved. The tracker log is intact."),
    ("{s} clocked the {place2} register at {t1} — the tape agrees to the minute.",
     "I clocked the register at {t1}. The tape agrees."),
)

# (clue, statement, subtle) — subtle herrings look worse before the
# exonerating tail lands.
_GEN_HERRINGS = (
    ("{s} was seen near {place} at {t1} — but the {place2} camera catches "
     "{s} buying coffee at that exact minute.",
     "I was buying coffee. The camera has me.", False),
    ("A {tool} belonging to {s} was found at {place} — reported missing "
     "from {s}'s kit three days earlier.",
     "My {tool} went missing days ago. I filed the report.", False),
    ("{s} asked about the {item}'s value that afternoon — as part of an "
     "insurance inventory, filed at {t1}.",
     "It was for the insurance inventory. The filing is timestamped.", False),
    ("{s}'s fingerprints are on the {lock} — from the morning opening "
     "shift, logged at {t1}.",
     "I open up every morning. The log has my shift.", True),
    ("{s} had {item} dust on their cuffs — but the lab confirms it is from "
     "{s}'s legitimate bench work at {t1}.",
     "That is from my bench work. The lab confirmed it.", True),
    ("{s} left {place} in a hurry at {t1} — to catch the last ferry; "
     "the manifest lists {s}.",
     "I was catching the ferry. The manifest lists me.", True),
)

# (evidence, incriminating detail, evidence noun)
_GEN_EVIDENCE = (
    ("a duplicate key, freshly cut",
     "the {item}'s serial scratched into the tang", "duplicate key"),
    ("a pawn ticket dated last night",
     "the {item}'s description written on the stub", "pawn ticket"),
    ("photos of the {item}",
     "timestamps from {t1}, before anyone reported it gone", "photos"),
    ("a shipping label",
     "made out to a private buyer, in {s}'s own hand", "shipping label"),
    ("a cloth with {item} fibers",
     "monogrammed with {s}'s initials", "cloth"),
    ("a burner phone",
     "one {t1} call to a known fence, logged", "burner phone"),
)

# motives: the culprit's is pointed; everyone else gets a plausible
# but milder reason.
_GEN_MOTIVES = (
    "owes money all over town",
    "was passed over for promotion twice",
    "holds a grudge after a public demotion",
    "needs cash to cover an old debt",
    "was promised a cut by a fence",
    "is desperate to keep their job",
    "resents the victim's success",
    "has expensive tastes and an empty wallet",
)

_GEN_CULPRIT_MOTIVES = (
    "was offered good money for the {item} by a private buyer",
    "needed the {item} to cover a debt coming due",
    "had been planning to take the {item} for months",
)

# (clue template, subtle)
_GEN_GUNS = (
    ("In {s}'s locker: {ev} — {det}.", False),
    ("On {s}'s workbench: {ev}, and {det}.", False),
    ("Tucked inside {s}'s bag: {ev}; {det}.", False),
    ("{s}'s phone location pinged {place} at {t1} — and {det}.", True),
    ("The {item}'s packing straw turned up in {s}'s locker: {ev} — {det}.", True),
    ("A receipt in {s}'s name surfaced at the pawn shop: {ev}, {det}.", True),
)

_GEN_DENIALS = (
    "\"I've never touched the {item}.\"",
    "\"I was nowhere near {place} that night.\"",
    "\"That {noun} isn't mine — someone planted it.\"",
    "\"Ask anyone — I left well before {t1}.\"",
    "\"I don't even know what the {item} looks like.\"",
    "\"Check the cameras — you'll see I'm innocent.\"",
)

_RECENT_CASE_IDS: Any = None  # lazy deque, avoids import cost at module load


def _recent() -> Any:
    global _RECENT_CASE_IDS
    if _RECENT_CASE_IDS is None:
        from collections import deque

        _RECENT_CASE_IDS = deque(maxlen=12)
    return _RECENT_CASE_IDS


def generate_case(rng: Any, difficulty: str = "medium") -> dict[str, Any]:
    """Compose a fresh, solvable case from the template pools.

    ``rng`` is any ``random.Random``-like; the same seed always builds
    the same case. ``difficulty`` is ``easy`` | ``medium`` | ``hard`` |
    ``expert`` (expert uses the subtle templates, like hard).  Emits
    the full god-tier schema: id, title, tier, briefing, motives,
    red_herrings, herring_suspect and solution alongside the classic
    story/suspects/culprit/clues/statements.
    """
    import random as _random

    if not isinstance(rng, _random.Random):
        rng = _random.Random(rng)
    difficulty = difficulty if difficulty in TIERS else "medium"
    tier = difficulty

    scene = rng.choice(_GEN_SCENES)
    item, place, lock = scene["item"], scene["place"], scene["lock"]
    suspects = list(rng.sample(_GEN_ROLES, 4))
    culprit = rng.choice(suspects)
    others = [s for s in suspects]
    rng.shuffle(others)
    others = [s for s in others if s != culprit]
    cleared_a, cleared_b, herring_suspect = others[0], others[1], others[2]

    t1, t2 = rng.choice(_GEN_TIMES)
    place2 = rng.choice(_GEN_PLACE2)
    tool = rng.choice(_GEN_TOOLS)

    def fill(tpl: str, s: str = "", **kw: str) -> str:
        return tpl.format(s=s, item=item, place=place, lock=lock,
                          place2=place2, t1=t1, t2=t2, tool=tool, **kw)

    story = (f"The {item} vanished from {place} overnight. "
             f"Four people had access.")
    title = f"The {item.title()} Case"
    briefing = (story + " Four people had access — and every one of "
                "them had a motive.")
    clues = [fill(f"The {lock} shows one clean entry at {t1} — "
                   "no force, no alarm.")]

    statements: dict[str, str] = {}
    for sus, (ctpl, stpl) in zip(
            (cleared_a, cleared_b), rng.sample(_GEN_ALIBIS, 2)):
        clues.append(fill(ctpl, sus))
        statements[sus] = fill(stpl, sus)

    # Red herring: difficulty picks the subtlety.
    herring_pool = [h for h in _GEN_HERRINGS
                    if (h[2] == (difficulty in ("hard", "expert")))
                    or difficulty == "medium"]
    hclue, hstmt, _ = rng.choice(herring_pool or _GEN_HERRINGS)
    hclue_f = fill(hclue, herring_suspect)
    clues.append(hclue_f)
    statements[herring_suspect] = fill(hstmt, herring_suspect)

    # Smoking gun: always names the culprit.
    gun_pool = [g for g in _GEN_GUNS
                if (g[1] == (difficulty in ("hard", "expert")))
                or difficulty == "medium"]
    gtpl, _ = rng.choice(gun_pool or _GEN_GUNS)
    ev, det, noun = rng.choice(_GEN_EVIDENCE)
    ev_f = fill(ev, culprit)
    det_f = fill(det, culprit)
    gun = fill(gtpl, culprit, ev=ev_f, det=det_f)
    clues.append(gun)

    denial = rng.choice(_GEN_DENIALS)
    statements[culprit] = fill(denial, culprit, noun=noun)

    motives = {s: rng.choice(_GEN_MOTIVES) for s in suspects}
    motives[culprit] = fill(rng.choice(_GEN_CULPRIT_MOTIVES), culprit)

    solution = (
        f"The {lock} showed one clean entry at {t1}: an insider did "
        f"it. {cleared_a} and {cleared_b} are cleared by the camera "
        f"and call logs. {herring_suspect} looked guilty at first "
        f"glance, but the exonerating detail in the evidence cleared "
        f"them. The smoking gun: {det_f} — in {culprit}'s possession. "
        f"Only {culprit} fits all five clues."
    )

    return {
        "id": f"gen-{rng.getrandbits(32):08x}",
        "title": title,
        "tier": tier,
        "difficulty": tier if tier != "expert" else "hard",
        "briefing": briefing,
        "story": story,
        "suspects": suspects,
        "motives": motives,
        "culprit": culprit,
        "clues": clues,
        "red_herrings": [hclue_f],
        "herring_suspect": herring_suspect,
        "solution": solution,
        "statements": statements,
        "generated": True,
    }


def random_case(rng: Any, difficulty: str | None = None) -> dict[str, Any]:
    """A case for a new game: ~55% from the hand-written bank, the rest
    freshly generated. Never repeats one of the last 12 served cases,
    and always carries a ``difficulty`` grade for scoring."""
    import random as _random

    if not isinstance(rng, _random.Random):
        rng = _random.Random(rng)
    difficulty = difficulty or rng.choice(("easy", "medium", "hard"))
    recent = _recent()

    if rng.random() < 0.55:
        candidates = [(f"bank:{i}", c) for i, c in enumerate(CASES)
                      if f"bank:{i}" not in recent]
        if not candidates:
            candidates = [(f"bank:{i}", c) for i, c in enumerate(CASES)]
        cid, picked = rng.choice(candidates)
        case = dict(picked)
        case["id"] = cid
        case["statements"] = dict(picked.get("statements") or {})
        case["generated"] = False
    else:
        case = generate_case(rng, difficulty)
        cid = case["id"]

    case = dict(case)
    case["difficulty"] = difficulty
    recent.append(cid)
    return case
