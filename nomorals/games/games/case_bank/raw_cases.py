"""Split from nomorals.games.games.cases (Wave H3) — content unchanged."""
from __future__ import annotations

from typing import Any

# The 30 original hand-written cases (bank:0 … bank:29).
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
