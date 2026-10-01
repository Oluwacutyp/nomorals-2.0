"""Medium-complexity games: state machines with real phases, roles, and
economics. Same contract as the easy games — pure state logic over
``room.state``, scripted-move testable — but longer arcs: mafia's
night/day cycle, the shop's ten rounds of price drift, an
investigation's clue-by-clue reveal, an RPG's sixteen-scene campaign.
"""
from __future__ import annotations

import random
import re
import time
from typing import Any

from ..ai import GameMind
from ..players import Player
from .base import MultiGame, Room
from .cases import (  # noqa: F401 - CASES kept for back-compat
    CASES,
    HINT_COST,
    HISTORY_VERSION,
    TIME_BONUS_MAX,
    TIME_LIMITS,
    blank_history,
    deal_case,
    random_case,
    record_played,
    score_solve,
)
from .easy import TRIVIA

__all__ = ["MEDIUM_GAMES"]

# ── shared banks ─────────────────────────────────────────────────────────────

CHALLENGES: tuple[tuple[str, str, str], ...] = (
    ("What word is 'listen' spelled backwards?", "silent", "a sound"),
    ("I have cities but no houses, water but no fish. What am I?",
     "a map", "paper"),
    ("9.11 + 9.12 in decimal, to two places?", "9.23", "add them"),
    ("Which is heavier: a kilogram of feathers or a kilogram of nails?",
     "same", "trick"),
    ("How many letters in the alphabet come after 'M'?", "13", "count"),
    ("A farmer has 17 sheep and all but 9 run away. How many are left?",
     "9", "read it twice"),
    ("What comes next: 2, 6, 12, 20, 30, …?", "42", "the gaps grow"),
    ("If 5 cats catch 5 mice in 5 minutes, how many cats catch 100 mice "
     "in 100 minutes?", "5", "rate"),
    ("What 5-letter word can you make shorter by adding 2 letters?",
     "short", "a verb"),
    ("Which number is the odd one out: 3, 5, 7, 9, 11?", "9", "primes"),
    ("A train leaves at 3:15 and travels 2 h 45 m. What time does it "
     "arrive?", "6:00", "add them"),
    ("How many months have 28 days?", "12", "trick"),
    ("What is the next letter: A, C, F, J, O, …?", "U", "+1, +2, +3…"),
    ("You're 6th in a race of 12 and pass the 5th runner. Where are "
     "you now?", "5th", "think it through"),
    ("How many times can you subtract 5 from 25?", "5", "trick"),
)

STORY_OPENERS: tuple[str, ...] = (
    "The lighthouse keeper counted one more boat in the harbor than "
    "there were boats at dusk.",
    "The market opened at dawn, but the stall that always sold clocks "
    "was selling silence instead.",
    "Three letters arrived at the same house on the same morning, each "
    "signed with the same name — her own.",
    "The train was late for the first time in forty years, and "
    "everyone on the platform pretended not to notice.",
    "Under the city, the river had started flowing uphill, and only the "
    "catfish seemed worried.",
    "The auctioneer sold the mirror first, because everything it "
    "reflected had already been sold.",
    "It rained for nine days, and on the tenth morning the puddles "
    "were all facing the same direction.",
    "The library's new rule was simple: no questions after closing "
    "time. The librarian stopped answering at 4:59.",
)

STORY_CLOSERS: tuple[str, ...] = (
    "Years later, no one could say which of them had started it.",
    "The city moved on, the way cities do — carefully, and without "
    "apologizing.",
    "In the end, the truth was smaller than the story, and larger "
    "than the silence.",
    "They never spoke of it again, which is how you knew it had "
    "happened.",
)

RPG_SCENES: tuple[dict[str, Any], ...] = (
    {
        "text": "You reach the edge of the Glass Forest, where the "
                "trees ring when the wind changes. Something is "
                "following you between the trunks.",
        "choices": [
            ("Push through the thickest part of the forest",
             "You crash through the glass bark, bleeding, and the "
             "follower's footfalls stop at the tree line. The "
             "silence afterwards is worse.",
             "The ring of the trees becomes a drum. The follower "
             "picks you out of the undergrowth in two strides.",
             +5, -4),
            ("Climb a dead tree and watch from above",
             "From the height you see it: a deer, glass-eared, "
             "carrying a lantern it didn't light. It is lost, not "
             "hunting.",
             "The branch cracks halfway up. You land in a tangle of "
             "branches, and the lantern comes closer.",
             +8, 0),
            ("Call out to it, human voice",
             "It stops. The lantern turns toward you — and you see "
             "the glass ears are broken, and the thing inside is "
             "smaller than the deer it's wearing.",
             "Your voice carries too far. Three lanterns appear on "
             "the treeline, and one of them was not there before.",
             +6, -2),
        ],
    },
    {
        "text": "The village well has been dry for three days, and "
                "the women of the village are starting to look at "
                "each other instead of the well.",
        "choices": [
            ("Swear you'll find water, and mean it",
             "They give you a rope, a lantern, and the name of the "
             "old man who knows the dry riverbeds.",
             "One of the women laughs at you. The whole village "
             "hears it, and the rope is handed to someone else.",
             +4, 0),
            ("Check the well's stones for a seal",
             "Under the third stone: a wax seal, cracked. Someone "
             "plugged this well on purpose, recently.",
             "Your fingers find only wet moss. The well looks like "
             "any other dry well, and you're an hour poorer.",
             +9, 0),
            ("Ask who has been near the well at night",
             "Two women say the same name at the same time. It is "
             "the old man.",
             "The women look at each other. Nobody says a name. "
             "You've learned nothing, and they know you asked.",
             +5, 0),
        ],
    },
    {
        "text": "A merchant offers to carry your bag for 10 coins. "
                "The bag is heavy. The merchant is smiling too "
                "hard.",
        "choices": [
            ("Pay the 10 coins",
             "He carries it with one hand and sings the whole way. "
             "At the gate, the bag is lighter — you hadn't noticed "
             "it was lighter before either.",
             "He carries it one hand and the other pocketing "
             "things. You find out at the gate, when your "
             "letters are missing.",
             +3, -10),
            ("Carry it yourself, but watch him",
             "The smile stays, but his eyes go to your shoulder "
             "bag four times. You're being weighed, not helped.",
             "You walk with your hand on the bag and learn nothing "
             "but suspicion, which costs you the same coin count.",
             +6, 0),
            ("Offer him a coin for his opinion of the road ahead",
             "'The toll is doubled tonight. The bridge is closed. "
             "The river is the way, if you can swim the current.' "
             "He says it without a smile, which is worth more than "
             "the coin.",
             "'The road is fine. I'm a merchant, not a prophet.' "
             "You learn nothing, and he's gone.",
             +8, -1),
        ],
    },
    {
        "text": "The bridge is out. The river is the way. The "
                "current is exactly as bad as the merchant said.",
        "choices": [
            ("Cross at the narrow, shallow point",
             "The water is cold and the current is a wall, but the "
             "shallows hold your weight. You come out shaking, "
             "alive, and dry enough to think.",
             "The shallows are shallower than they look. The "
             "current takes you under once, twice, and the "
             "third time the bank grabs your boot.",
             +7, -6),
            ("Look for a ferry, or a forder",
             "An hour upstream, a fisherman with a flat boat and a "
             "price. He asks no questions, which is either "
             "hospitality or something else.",
             "Two hours of walking upstream. The fisherman is "
             "gone. The river is still the way.",
             +5, -2),
            ("Send one thing across first, on a line",
             "The bag makes it. Your letter, your map, your knife — "
             "across in a line, in the order you tied them. The "
             "river takes the line and you stand holding the end "
             "of it, lighter and smarter.",
             "The line holds the bag. It does not hold your "
             "letter. The river keeps what it keeps.",
             +6, -3),
        ],
    },
    {
        "text": "You find the old man by the dry riverbed. He is "
                "cooking something that smells like rain.",
        "choices": [
            ("Ask him straight where the water went",
             "It didn't go. It was taken. The seal under the third "
             "stone is my sister's mark, and she's been deaf since "
             "spring. If she can sign for it, someone paid her in a "
             "language she couldn't read.",
             "He looks at you for a long time. The water went where "
             "all water goes, he says, and turns back to his pot. "
             "You have lost the moment.",
             +8, 0),
            ("Sit down and share the meal first",
             "The food is rain, somehow. By the second bowl he is "
             "talking about the seal, the sister, and a debt paid "
             "in a language she could not read.",
             "You sit. He does not offer. The meal is cold by the "
             "time you have said enough to be interesting, and he "
             "is already asleep.",
             +7, 0),
            ("Ask about the seal on the well stones",
             "That seal is my sister's. She has been deaf since "
             "spring. If someone is signing her name to move "
             "water, they are paying her in a language she cannot "
             "read. Go find out what it says.",
             "Seals are for mail. He says it the way a man saves "
             "the rest of the conversation for later. There is no "
             "later.",
             +9, 0),
        ],
    },
    {
        "text": "The town's only clock has stopped at 3:33, and "
                "the town is starting to agree to it. Shops open at "
                "3:33. The school bell rings at 3:33. No one will "
                "say why.",
        "choices": [
            ("Ask the clockmaker",
             "The clock did not stop. It was told. The gears "
             "are frozen at a specific second, and the "
             "mainspring is wound to the day it was set. If "
             "you want the time back, find who wound it, "
             "and do not let them see you looking.",
             "The clock is old. It stops. He says it three "
             "times, and you leave.",
             +8, 0),
            ("Check the town square for a second clock",
             "There is a second clock. It is running. It says 3:33. "
             "Someone is keeping time on purpose, and the town is "
             "following.",
             "Every clock you find is gone, broken, or at 3:33. "
             "The town is a room with one door, and you've found "
             "the door, but not the key.",
             +7, 0),
            ("Go to the school and ask the children",
             "The children are not afraid. They are annoyed. 'It's "
             "the bell, we don't set it, the teacher sets it, and "
             "the teacher says the town set it, and the town says "
             "we set it, and nobody has stopped saying it.'",
             "The school is empty at 3:33. The bell is still "
             "ringing. You are standing in it.",
             +6, -2),
        ],
    },
    {
        "text": "The woman at the door has your face. She is "
                "standing in your hallway. She is holding your "
                "coat.",
        "choices": [
            ("Ask her who she is",
             "I am the one who has been here since before "
             "you. You walk in every night and I walk out "
             "every morning. We have been trading. You get "
             "the life. I get the coat.",
             "Who are you? You say it like a man who has "
             "already answered. She nods. Right. So you "
             "know.",
             +9, 0),
            ("Step back into the house",
             "The hallway is longer than it should be. Your coat "
             "is on the hook. The woman is still in the doorway, "
             "holding the other coat, the one you don't remember "
             "owning.",
             "The door closes behind you. The hallway is the "
             "length it should be. The woman is gone. The coat on "
             "the hook is yours. The coat in her hand is not.",
             +5, -3),
            ("Take the coat from her",
             "The coat is lighter than yours. It fits. When you put "
             "it on, the hallway is a hallway, and the woman is a "
             "woman at a door, and you are the one who lives here "
             "now.",
             "The coat is heavier than yours. It fits. When you put "
             "it on, the hallway is a hallway, and the woman is "
             "smiling, and you are the one who was traded.",
             +8, -1),
        ],
    },
    {
        "text": "The road ends at a door in the hillside. The door "
                "has your name on it, in a hand you almost "
                "recognize.",
        "choices": [
            ("Knock",
             "A hand opens the door from the inside. It is your "
             "hand. The person inside is older, and they are "
             "holding a letter with your name on it, in a hand "
             "you recognize now.",
             "The door opens. The person inside is not you. The "
             "letter has your name on it. They are already reading "
             "it.",
             +9, -2),
            ("Read the name on the door more carefully",
             "It is your name, but the last letter is a different "
             "letter. A letter you've never used. A letter that "
             "means 'the one who came back.'",
             "It is your name. It is exactly your name. You stand "
             "there reading it for a long time, and the door does "
             "not open.",
             +8, 0),
            ("Walk past it",
             "The road continues. The door is behind you. You do "
             "not look back, and the road is a road, and the "
             "letter, when you find it later, is in your coat "
             "pocket, and you did not put it there.",
             "The road continues. The door is behind you. You look "
             "back. The door is still there. It is waiting.",
             +6, 0),
        ],
    },
   {
        "text": "The ferryman across the salt flats will only take "
                "one passenger at a time, and he has already "
                "declined three of you.",
        "choices": [
            ("Offer him something he has never seen",
             "He turns the object over twice and lets you board "
             "first. The crossing takes ten minutes instead of an "
             "hour, and he does not speak.",
             "He keeps the object and still declines you. You walk "
             "the flats alone, and the salt is not as forgiving as "
             "the ferryman looked.",
             +7, -3),
            ("Cross in the middle of the night, when he sleeps",
             "You are across before dawn. The ferry is moored to a "
             "rock, and the rock is warm, which should not be "
             "possible on the flats.",
             "The water is exactly where the water was the night "
             "before, and the flats are very bad at keeping "
             "secrets. You lose the night and part of your "
             "shoes.",
             +9, -6),
            ("Ask him what he actually wants",
             "He names a coin minted in a city that stopped "
             "minting coins in his grandfather's time. You do not "
             "have it. He lets you talk for an hour anyway, which "
             "is more than you expected.",
             "He says the same sentence nine times. The ninth "
             "time it is almost a warning, and you are not sure "
             "which of you it is aimed at.",
             +5, -1),
        ],
    },
    {
        "text": "A market in the high pass sells maps of the "
                "valley below. Every map shows a different valley.",
        "choices": [
            ("Buy all three and compare them",
             "Where the three maps disagree, you find the real "
             "river: the only line all three draw in the same "
             "place. The rest is negotiation.",
             "The three maps agree on nothing, and the seller "
             "watches you fold them back up with an expression "
             "you will spend the next week decoding.",
             +8, -2),
            ("Draw your own from memory and trade it in",
             "The seller pins yours to the wall between the "
             "others. A stranger stops, points at it, and says "
             "the word for home in a language you almost know.",
             "The seller declines the trade without looking at "
             "it. Your memory, it turns out, drew a valley that "
             "isn't there.",
             +6, -4),
            ("Ask the locals instead of buying anything",
             "Three of them tell you the way down. Two agree. The "
             "third asks you to carry a letter to someone in the "
             "valley, and does not say who.",
             "The locals smile and give you the map that costs "
             "nothing, which means: the road, the only road, the "
             "road that is not on any of the maps.",
             +4, 0),
        ],
    },
    {
        "text": "The bell tower has been silent for a year, and "
                "tonight, with no one on the stairs, it rings "
                "once.",
        "choices": [
            ("Climb the tower in the dark",
             "The rope is coiled exactly as it was a year ago, "
             "and the bell is clean, and the clapper is hanging "
             "at an angle that no wind could hold. Someone is "
             "still up there.",
             "You take the stairs two at a time and the tower "
             "takes your rhythm from you. Halfway up you hear "
             "your own name, spoken softly, from below.",
             +8, -5),
            ("Wait by the tower with a lantern",
             "At the second bell — you did not know there would "
             "be one — a hand opens the door, and you are let "
             "in like a guest you forgot you were invited as.",
             "Nothing comes out of the tower. You stand there "
             "until the lantern goes, and the town does not ask "
             "why you came back pale.",
             +5, -2),
            ("Mark the tower and tell the whole town at dawn",
             "They do not thank you. They go up in a line, "
             "together, and the tower is quiet for the rest of "
             "the night, which is more than it has been in a "
             "year.",
             "Half the town laughs and half of them locks their "
             "doors. The bell rings a third time after you sleep, "
             "and you do not hear it, and you do not dream.",
             +3, -3),
        ],
    },
    {
        "text": "You are offered a seat at a table where a card "
                "game is mid-hand. Nobody says the rules, and "
                "nobody is losing.",
        "choices": [
            ("Sit down and play along",
             "The game is simple and you get it in three hands, "
             "and by the fourth the table is laughing at "
             "something you said that you don't remember "
             "saying.",
             "You sit down and the game is not simple, and the "
             "cards are not cards, and the table is longer than "
             "the room. You leave before you lose, which is "
             "counting.",
             +6, -4),
            ("Watch a full hand from the doorway",
             "You learn the rhythm: the game is not about the "
             "cards, it is about the pauses. You know when to "
             "speak. The table, eventually, knows you too.",
             "You watch one hand and the players watch you the "
             "whole time. When you finally leave, no one looks "
             "up, and the game continues exactly where it was.",
             +4, 0),
            ("Ask to be taught, properly, some other day",
             "They agree. You are given a card — one, face "
             "down — and told to keep it dry. The lesson starts "
             "the next time you find a table.",
             "They agree too, which is the trap. The card they "
             "hand you is a debt, and the next table will "
             "collect interest.",
             +5, -6),
        ],
    },
    {
        "text": "The road forks at a bridge with only one lane, "
                "and the second traveler is already on it, "
                "walking the wrong way.",
        "choices": [
            ("Step aside and let them pass first",
             "They nod, cross, and at the far end they stop and "
             "point back at you — a direction, not a word. You "
             "follow the finger and it works.",
             "They nod and keep walking, and the bridge is "
             "longer than the river is wide. You end up "
             "backtracking half a day with your pride folded "
             "small.",
             +4, 0),
            ("Refuse to move and hold the lane",
             "The standoff holds for a full minute. Then they "
             "laugh — a real laugh — and give you the lane "
             "without a word, and something about that laugh "
             "will help you later, you just don't know what yet.",
             "The standoff holds, and then their hand moves to "
             "their belt, and your 'first' costs more than the "
             "bridge is wide. You let them through fast.",
             +2, -7),
            ("Look for a second way across",
             "The ford is where they said it would be, and the "
             "water is cold and the other traveler arrives at "
             "the far side at exactly the same minute, as if "
             "they had taken the long way on purpose.",
             "There is no ford. There is no second way. You "
             "cross on the bridge behind them, and they do not "
             "say anything at all, which is worse.",
             +7, -3),
        ],
    },
    {
        "text": "A child in the camp offers you half a loaf and "
                "asks you to carry her story to the next town.",
        "choices": [
            ("Carry it, and mean it",
             "In the next town, someone listens to the whole "
             "story and pays for bread for the camp for a week. "
             "The child's face when the news comes back is "
             "worth every mile.",
             "You carry it, but the next town is two days "
             "worse than the child said, and the story gets "
             "lighter by a mile with every mile. What you "
             "deliver is a rumour, and the child can tell.",
             +8, -4),
            ("Give her the half loaf and keep walking",
             "She takes it without argument. You walk faster "
             "than you should, and the bread you would have "
             "saved for yourself feels heavier than hers ever "
             "was.",
             "She takes it, and then you walk, and by night "
             "you have eaten half of your own loaf and told "
             "yourself a story about it. The story does not "
             "taste like bread.",
             +3, -2),
            ("Ask what the story is worth, first",
             "She looks at you for a long time, takes back the "
             "loaf, and gives you the other half instead. You "
             "are not sure which half is the story anymore, "
             "and you are carrying the wrong one.",
             "She answers with a question. You answer with a "
             "number. She laughs, keeps the loaf, and the "
             "story goes to someone else's pack.",
             +2, -5),
        ],
    },
    {
        "text": "The last room of the old house has a window "
                "that looks out onto a street that does not "
                "exist anywhere else. It is raining there.",
        "choices": [
            ("Open the window and listen",
             "The rain is on a street with people, and one of "
             "them is looking up, and has been looking up for "
             "a while. When you finally wave, the looking-up "
             "stops. You are not sure you wanted that.",
             "The rain is there and the street is there, but "
             "when you lean out you hear nothing at all, which "
             "means the rain has no sound and the street has "
             "no air, and you close the window and take your "
             "lantern with you.",
             +6, -3),
            ("Copy the street exactly as you see it, in ink",
             "You draw it for an hour. When you hold the page "
             "up to the window, the people on the street are "
             "standing in your ink. One of them is waving.",
             "You draw it all wrong, in the wrong hand, and "
             "the street looks back at you with an expression "
             "that is very close to disappointment. You keep "
             "the page anyway.",
             +9, -5),
            ("Leave the window shut and leave the house",
             "The street stays wet and the window stays shut. "
             "At the door you understand that you were never "
             "meant to go through it, and that is almost a "
             "relief.",
             "You leave the window shut and the house leaves "
             "you at the door — one step further out than the "
             "front step, on a road you did not walk up.",
             +4, -1),
        ],
    },
    {
        "text": "The campaign's last mile is a door. The door "
                "has no handle on this side, and the lock is "
                "a keyhole shaped like the one on your own "
                "house's door.",
        "choices": [
            ("Speak your own name, exactly, once",
             "The door does not open. It unlocks. There is a "
             "difference, and you will spend the rest of your "
             "life knowing it. What is on the other side is "
             "yours to find out in the next story.",
             "The door does not unlock. But the keyhole "
             "glows, faintly, the way a pilot light does, and "
             "you understand: it was waiting for you to be "
             "worth the trying.",
             +10, 0),
            ("Knock, the way you knock at home",
             "The knocking comes back at you from the other "
             "side, three knocks to your four. You are not "
             "alone in the story anymore, and the door knows "
             "it too.",
             "The knocking comes back wrong — four to your "
             "three — and you stop, and the silence on the "
             "other side is the loudest thing in the "
             "campaign.",
             +6, -4),
            ("Turn around and take the long way home",
             "You walk the last mile backwards, and the "
             "campaign follows you all the way: every face, "
             "every door, every bell. You arrive home with "
             "the whole story behind your eyes.",
             "You turn around, and the long way home is "
             "shorter than the door was, and the story ends "
             "where the road does, which is to say: "
             "incompletely, on purpose, for the next time.",
             +3, -2),
        ],
    },
)

SHOP_GOODS: tuple[tuple[str, int], ...] = (
    ("a crate of oranges", 60), ("a roll of copper wire", 90),
    ("a jar of beeswax", 45), ("a bundle of dried herbs", 70),
    ("a hand-blown glass bottle", 55), ("a spool of silk thread", 110),
    ("a bundle of firewood", 40), ("a sack of river salt", 65),
)


def _word_of(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower().strip(".!?"))


# ── 9. mafia / werewolf ──────────────────────────────────────────────────────

class MafiaGame(MultiGame):
    name = "mafia"
    description = "the house plays the mafia — survive five nights with the table"
    min_players = 3
    max_players = 10
    ai_seats = 2
    needs_group = True
    move_timeout = 120
    rules = ("The house is the mafia. You are the town. Each night the "
             "mafia kills one of you (you'll see who). By day: talk "
             "(your turn = say anything), then vote — most votes "
             "exposes someone. The town wins by surviving five nights "
             "or outliving the mafia's credibility; the mafia wins "
             "when the table stops trusting itself. It's a social game "
             "— argue, misdirect, survive.")

    def new_state(self, rng: random.Random) -> dict[str, Any]:
        ai = [p for p in [] ]  # filled in setup
        return {"night": 0, "nights": 5, "phase": "setup",
                "roles": {}, "alive": [], "kills": 0,
                "votes": {}, "spoke": [], "suspicion": {}}

    def setup(self, room, mind):
        s = room.state
        s["alive"] = [p.key for p in room.players]
        ai_keys = [p.key for p in room.players if p.is_ai]
        # the mafia is always the house: pick 1 (or 2 in big games)
        mafia = mind.rng.sample(ai_keys, min(2 if len(room.players) >= 6
                                             else 1, len(ai_keys)))
        for p in room.players:
            s["roles"][p.key] = "mafia" if p.key in mafia else "town"
            s["suspicion"][p.key] = 0.0
        s["mafia"] = mafia
        s["phase"] = "night"
        return ("mafia — the house is in the wall. you're the town. "
                "night falls…")

    def _alive(self, room: Room) -> list[Player]:
        return [room.player(k) for k in room.state["alive"]
                if room.player(k) is not None]

    def _kill(self, room: Room, key: str) -> list[str]:
        s = room.state
        if key not in s["alive"]:
            return []
        s["alive"].remove(key)
        victim = room.player(key)
        role = s["roles"].get(key, "town")
        s["kills"] += 1
        out = [f"morning. {victim.name if victim else 'someone'} is "
               f"found. they were {role}."]
        if not any(k in s["alive"] and not k.startswith("ai:")
                   for k in s["alive"]):
            s["done"] = True
            out.append("🏁 the table is empty. the mafia owns the "
                       "streets.")
        elif s["kills"] >= s["nights"]:
            s["done"] = True
            out.append("🏁 five nights and the table is still standing. "
                       "the town survives.")
        return out

    def on_move(self, room, player, text, mind):
        s = room.state
        if s["phase"] == "talk":
            s["spoke"].append(player.key)
            s["suspicion"][player.key] = \
                min(1.0, s["suspicion"].get(player.key, 0.0) + 0.02)
            out = [f"{player.name}: {text[:300]}"]
            if len(s["spoke"]) >= len(self._alive(room)):
                s["phase"] = "vote"
                s["votes"] = {}
                s["spoke"] = []
                out.append("the talk is done. voting — name who you "
                           "think is the mafia (or 'skip').")
            return out
        if s["phase"] == "vote":
            t = text.strip().lower()
            if t in {"skip", "abstain", "-"}:
                s["votes"][player.key] = None
                out = [f"{player.name} abstains."]
            else:
                target = None
                for k in s["alive"]:
                    p = room.player(k)
                    if p is not None and p.key != player.key and \
                            (p.name.lower() in t or t in p.name.lower()):
                        target = k
                        break
                if target is None:
                    names = ", ".join(room.player(k).name
                                      for k in s["alive"]
                                      if k != player.key)
                    return [f"name one of the living: {names} (or 'skip')."]
                s["votes"][player.key] = target
                out = [f"{player.name} votes {room.player(target).name}."]
            if all(p.key in s["votes"] for p in self._alive(room)
                   if p.key != player.key or s["votes"].get(player.key) is not None):
                out.extend(self._tally(room, mind))
            return out
        return []

    def _tally(self, room: Room, mind: GameMind) -> list[str]:
        s = room.state
        counts: dict[str, int] = {}
        for target in s["votes"].values():
            if target:
                counts[target] = counts.get(target, 0) + 1
        if not counts:
            out = ["nobody voted. the town is quiet — the mafia "
                   "loves a quiet town."]
        else:
            top = max(counts.values())
            leaders = [k for k, v in counts.items() if v == top]
            if len(leaders) > 1:
                out = [f"split vote — {', '.join(room.player(k).name for k in leaders)}. nobody falls."]
                for k in leaders:
                    s["suspicion"][k] = min(1.0, s["suspicion"].get(k, 0) + 0.15)
            else:
                out = self._kill(room, leaders[0])
        if s.get("done"):
            return out
        s["phase"] = "night"
        s["night"] = s["kills"]
        out.append(f"night {s['kills'] + 1} falls…")
        return out

    def ai_turn(self, room, mind):
        s = room.state
        me = room.current
        if s["phase"] == "talk":
            s["spoke"].append(me.key)
            out = [f"{me.name} (the house, publicly): {mind.choice(('the well dried up the same night the map vanished, interesting', 'i trust the table, but i check the locks', 'who was last at the vault? we should say it out loud'), 'mafia day talk')}"]
            if len(s["spoke"]) >= len(self._alive(room)):
                s["phase"] = "vote"
                s["votes"] = {}
                s["spoke"] = []
                out.append("the talk is done. voting — name who you "
                           "think is the mafia (or 'skip').")
            return out
        if s["phase"] == "vote":
            if me.key in s["votes"]:
                return []
            targets = [k for k in s["alive"] if k != me.key]
            susp = {k: s["suspicion"].get(k, 0.3) + mind.rng.random() * 0.2
                    for k in targets}
            pick = mind.vote([room.player(k).name for k in targets], susp,
                             "you are a citizen in a mafia game; the "
                             "mafia is the house; vote a human name")
            target = None
            for k in targets:
                if room.player(k).name.lower() in pick.lower():
                    target = k
                    break
            s["votes"][me.key] = target
            out = [f"{me.name} votes {room.player(target).name if target else '— (abstain)'}."]
            if all(p.key in s["votes"] for p in self._alive(room)):
                out.extend(self._tally(room, mind))
            return out
        if s["phase"] == "night":
            mafia = s.get("mafia", [])
            if me.key in mafia and s["kills"] < s["nights"]:
                victims = [k for k in s["alive"]
                           if not k.startswith("ai:")]
                if not victims:
                    return []
                pick = mind.vote([room.player(k).name for k in victims],
                                 {k: 1.0 for k in victims},
                                 "you are the mafia; pick tonight's "
                                 "victim among the town")
                target = victims[0]
                for k in victims:
                    if room.player(k).name.lower() in pick.lower():
                        target = k
                        break
                out = self._kill(room, target)
                if not s.get("done"):
                    if s["kills"] >= s["nights"]:
                        s["done"] = True
                        out.append("🏁 five nights. the table survives "
                                   "— the town wins.")
                    else:
                        s["phase"] = "talk"
                        s["spoke"] = []
                        out.append("the table wakes. talk, then vote. "
                                   f"{self._alive(room)[0].name}, "
                                   "start the talk.")
                return out
            if not any(k in s["mafia"] for k in [me.key]):
                return []
        return []

    def is_over(self, room):
        return room.state.get("done", False)

    def winner(self, room):
        if not room.state.get("done"):
            return None
        humans_alive = any(k in room.state["alive"]
                           for k in room.state["alive"]
                           if not k.startswith("ai:"))
        if humans_alive:
            return "draw"  # the table survived together
        return Player(key="ai:mafia", platform="ai", name="The House",
                      is_ai=True)

    def describe_state(self, room):
        s = room.state
        alive = [room.player(k).name for k in s["alive"]]
        return (f"night {min(s['kills'] + 1, s['nights'])}/{s['nights']} · "
                f"phase: {s['phase']} · alive: {', '.join(alive)}")


# ── 10. king of the hill ─────────────────────────────────────────────────────

class KingOfHillGame(MultiGame):
    name = "king"
    description = "15 challenges — the table races for the hill"
    min_players = 1
    max_players = 8
    ai_seats = 2
    move_timeout = 90
    rules = ("15 quick challenges, everyone in turn each round. "
             "Correct: +1 (streaks pay +2). The house answers at 80% "
             "and keeps its own score — the hill goes to whoever holds "
             "the most at the end.")

    def new_state(self, rng: random.Random) -> dict[str, Any]:
        return {"challenges": rng.sample(list(CHALLENGES), 15),
                "round": 0, "points": {}, "streak": {},
                "king": None, "total": 15}

    def setup(self, room, mind):
        c = room.state["challenges"][0]
        return (f"king of the hill — {len(room.players)} climb, "
                f"{self.total_rounds(room)} rounds.\n"
                f"R1: {c[0]}\n({c[2]})")

    @staticmethod
    def total_rounds(room: Room) -> int:
        return room.state["total"]

    def _next(self, room: Room) -> str:
        s = room.state
        s["round"] += 1
        if s["round"] >= s["total"]:
            s["done"] = True
            return "the hill is settled."
        c = s["challenges"][s["round"]]
        return f"R{s['round']+1}: {c[0]}\n({c[2]})"

    def _check_king(self, room: Room) -> None:
        s = room.state
        top = max(s["points"].values()) if s["points"] else 0
        leaders = [k for k, v in s["points"].items() if v == top and top > 0]
        if len(leaders) == 1:
            s["king"] = leaders[0]

    def on_move(self, room, player, text, mind):
        s = room.state
        c = s["challenges"][s["round"]]
        guess = _word_of(text)
        right = c[1] in guess or guess in c[1]
        out: list[str] = []
        if right:
            s["streak"][player.key] = int(s["streak"].get(player.key, 0)) + 1
            pts = 2 if s["streak"][player.key] >= 2 else 1
            s["points"][player.key] = int(s["points"].get(player.key, 0)) + pts
            out.append(f"{player.name}: +{pts}")
        else:
            s["streak"][player.key] = 0
            out.append(f"{player.name}: off the trail (it was {c[1]}).")
        self._check_king(room)
        line = self._next(room)
        if s.get("done"):
            self._crown(room, out)
        else:
            out.append(line)
        return out

    def ai_turn(self, room, mind):
        s = room.state
        c = s["challenges"][s["round"]]
        correct = mind.rng.random() < 0.8
        out: list[str] = []
        if correct:
            s["streak"][room.current.key] = \
                int(s["streak"].get(room.current.key, 0)) + 1
            pts = 2 if s["streak"][room.current.key] >= 2 else 1
            s["points"][room.current.key] = \
                int(s["points"].get(room.current.key, 0)) + pts
            out.append(f"house: {c[1]} — +{pts}")
        else:
            s["streak"][room.current.key] = 0
            out.append("house slips on the scree.")
        self._check_king(room)
        line = self._next(room)
        if s.get("done"):
            self._crown(room, out)
        else:
            out.append(line)
        return out

    def _crown(self, room: Room, out: list[str]) -> None:
        s = room.state
        best = max(s["points"].items(), key=lambda kv: kv[1],
                   default=("", 0))
        w = room.player(best[0]) if best[0] else None
        out.append("🏁 the hill goes to " +
                   (f"{w.name} — {best[1]} pts." if w and best[1]
                    else "nobody. the hill keeps itself."))

    def is_over(self, room):
        return room.state.get("done", False)

    def winner(self, room):
        best = max(room.state.get("points", {}).items(),
                   key=lambda kv: kv[1], default=("", 0))
        w = room.player(best[0]) if best[0] else None
        return w if (w and best[1] > 0) else "draw"

    def score(self, room, player):
        return int(room.state.get("points", {}).get(player.key, 0))

    def describe_state(self, room):
        s = room.state
        king = room.player(s["king"]) if s.get("king") else None
        return (f"R{s['round']+1}/{s['total']} · hill: "
                f"{king.name if king else 'unclaimed'} · "
                f"scores: {s['points'] or '—'}")


# ── 11. story chain ──────────────────────────────────────────────────────────

class StoryChainGame(MultiGame):
    name = "story"
    description = "the table weaves one story, sentence by sentence"
    min_players = 1
    max_players = 8
    ai_seats = 1
    move_timeout = 120
    rules = ("I open the story. You each add one sentence, in turn — "
             "it must continue from the last line. Two loops around the "
             "table and I close it. The longest, most-woven line gets "
             "the weaver's bonus.")

    def new_state(self, rng: random.Random) -> dict[str, Any]:
        return {"lines": [], "loops": 2, "line_count": 0,
                "lengths": {}, "done": False}

    def setup(self, room, mind):
        opener = mind.choice(list(STORY_OPENERS), "story opener")
        room.state["lines"] = [opener]
        room.state["opening"] = opener
        return (f"story chain — {len(room.players)} writers.\n"
                f"“{opener}”\n"
                f"{room.players[0].name}, one sentence, continue it.")

    def on_move(self, room, player, text, mind):
        s = room.state
        t = text.strip()
        if len(t) < 8:
            return ["a real sentence — at least a breath of one."]
        if len(t) > 400:
            return ["one sentence. you're writing a novel, not a line."]
        s["lines"].append(t)
        s["lengths"][player.key] = int(s["lengths"].get(player.key, 0)) + \
            len(t)
        s["line_count"] += 1
        out = [f"{player.name}: “{t}”"]
        if s["line_count"] >= (len(room.players) * s["loops"] - 1):
            s["done"] = True
            out.extend(self._close(room, mind))
        return out

    def ai_turn(self, room, mind):
        s = room.state
        last = s["lines"][-1] if s["lines"] else ""
        line = mind.story_sentence(last)
        s["lines"].append(line)
        s["lengths"][room.current.key] = \
            int(s["lengths"].get(room.current.key, 0)) + len(line)
        s["line_count"] += 1
        out = [f"house: “{line}”"]
        if s["line_count"] >= (len(room.players) * s["loops"] - 1):
            s["done"] = True
            out.extend(self._close(room, mind))
        return out

    def _close(self, room: Room, mind: GameMind) -> list[str]:
        s = room.state
        closer = mind.choice(list(STORY_CLOSERS), "story closing")
        out = [f"the table writes the ending: “{closer}”"]
        best = max(s["lengths"].items(), key=lambda kv: kv[1],
                   default=("", 0))
        w = room.player(best[0]) if best[0] else None
        if w is not None and not w.is_ai and best[1] > 0:
            s["weaver"] = w.key
            out.append(f"the weaver's bonus: {w.name} "
                       f"({best[1]} letters of the story).")
        out.append("🏁 the story is done. it's in the table now.")
        return out

    def is_over(self, room):
        return room.state.get("done", False)

    def winner(self, room):
        w = room.player(room.state.get("weaver", "")) if \
            room.state.get("weaver") else None
        return w if (w and not w.is_ai) else "draw"

    def score(self, room, player):
        return int(room.state.get("lengths", {}).get(player.key, 0))


# ── 12. rpg adventure (the AI as game master) ────────────────────────────────

class RpgAdventureGame(MultiGame):
    name = "rpg"
    description = "a 16-scene campaign — the house is your game master"
    min_players = 1
    max_players = 4
    ai_seats = 0
    move_timeout = 180
    rules = ("I run a short campaign. Each scene, I set it and give you "
             "three ways in (plus 'd' for the reckless route). The "
             "house rolls the dice and tells you what happens. You "
             "start at 30 HP with 1 potion. Survive 16 scenes, earn "
             "XP, level up. At 0 HP you're out of the story.")

    def new_state(self, rng: random.Random) -> dict[str, Any]:
        return {"scene": 0, "scenes": len(RPG_SCENES), "sheets": {}, "d": {}}

    def setup(self, room, mind):
        for p in room.players:
            room.state["sheets"][p.key] = {"hp": 30, "max_hp": 30,
                                           "xp": 0, "level": 1,
                                           "potions": 1, "out": False}
        return self._scene_text(room)

    def _scene_text(self, room: Room) -> str:
        s = room.state
        idx = s["scene"] % len(RPG_SCENES)
        sc = RPG_SCENES[idx]
        lines = [f"— scene {s['scene']+1}/{s['scenes']} —", sc["text"],
                 "1. " + sc["choices"][0][0],
                 "2. " + sc["choices"][1][0],
                 "3. " + sc["choices"][2][0],
                 "d. something reckless (50/50, big swing)"]
        for p in room.players:
            sh = s["sheets"][p.key]
            if not sh["out"]:
                lines.append(f"({p.name}: {sh['hp']}/{sh['max_hp']} HP, "
                             f"lvl {sh['level']}, {sh['potions']} potions)")
        return "\n".join(lines)

    def _roll(self, room: Room, dc: int) -> int:
        return room.rng().randint(1, 20) + dc

    def _apply(self, room: Room, p: Player, success: bool,
               idx: int) -> list[str]:
        s = room.state
        sh = s["sheets"][p.key]
        if sh["out"]:
            return [f"{p.name} is out of the story and watches from "
                    "the tree line."]
        sc = RPG_SCENES[s["scene"] % len(RPG_SCENES)]
        if idx == 3:  # the reckless route
            roll = room.rng().randint(1, 20)
            success = roll >= 10
            landed = "it lands." if success else "it doesn't."
            text = (f"you gamble it. the roll: **{roll}** — "
                    f"{landed}")
            dxp, dhp = (10, -8) if success else (2, -6)
        else:
            text = sc["choices"][idx][1 if success else 2]
            dxp, dhp = sc["choices"][idx][3], sc["choices"][idx][4]
        sh["xp"] += dxp
        sh["hp"] = max(0, min(sh["max_hp"], sh["hp"] + dhp))
        if sh["hp"] >= sh["max_hp"] and sh["xp"] >= sh["level"] * 12:
            sh["level"] += 1
            sh["max_hp"] += 5
            sh["hp"] = sh["max_hp"]
            text += f" **level up — you're level {sh['level']} now.**"
        out = [f"{p.name}: {text}"]
        if dhp:
            out.append(f"  (HP {sh['hp']}/{sh['max_hp']}, XP {sh['xp']})")
        if sh["hp"] <= 0:
            sh["out"] = True
            out.append(f"{p.name} is out of the story.")
        return out

    def _advance(self, room: Room) -> list[str]:
        s = room.state
        s["scene"] += 1
        if s["scene"] >= s["scenes"]:
            s["done"] = True
            out = ["— the campaign ends —"]
            for p in room.players:
                sh = s["sheets"][p.key]
                out.append(f"  {p.name}: level {sh['level']}, "
                           f"{'alive' if not sh['out'] else 'out'}, "
                           f"{sh['xp']} XP")
            best = max((p for p in room.players if not
                        s["sheets"][p.key]["out"]),
                       key=lambda p: s["sheets"][p.key]["xp"],
                       default=None)
            if best is not None:
                out.append(f"🏁 the story belongs to {best.name}.")
            return out
        return [self._scene_text(room)]

    def on_move(self, room, player, text, mind):
        s = room.state
        sh = s["sheets"].get(player.key)
        if sh is None or sh["out"]:
            return []
        t = text.strip().lower()
        if t == "potion" and sh["potions"] > 0 and sh["hp"] < sh["max_hp"]:
            sh["potions"] -= 1
            sh["hp"] = min(sh["max_hp"], sh["hp"] + 15)
            return [f"{player.name} drinks a potion. "
                    f"(HP {sh['hp']}/{sh['max_hp']}) — now choose your "
                    "way in."]
        idx = None
        if t in {"1", "one", "a"}:
            idx = 0
        elif t in {"2", "two", "b"}:
            idx = 1
        elif t in {"3", "three", "c"}:
            idx = 2
        elif t in {"d", "reckless", "4", "four"}:
            idx = 3
        if idx is None:
            return ["1, 2, 3, or d — or 'potion' if you're hurt."]
        roll = self._roll(room, 2)
        out = self._apply(room, player, roll >= 10, idx)
        out.extend(self._advance(room))
        return out

    def ai_turn(self, room, mind):
        return []  # the house is the GM, not a player

    def is_over(self, room):
        return room.state.get("done", False)

    def winner(self, room):
        s = room.state
        alive = [p for p in room.players
                 if not s["sheets"].get(p.key, {}).get("out")]
        if not alive:
            return "draw"
        best = max(alive, key=lambda p: s["sheets"][p.key]["xp"])
        return best

    def score(self, room, player):
        return int(room.state.get("sheets", {}).get(player.key, {})
                   .get("xp", 0))

    def describe_state(self, room):
        s = room.state
        return (f"scene {s['scene']+1}/{s['scenes']} · " +
                " · ".join(f"{p.name} {s['sheets'][p.key]['hp']}HP"
                           for p in room.players
                           if not s["sheets"][p.key]["out"]))


# ── 13. shop / economy game ──────────────────────────────────────────────────

class ShopGame(MultiGame):
    name = "shop"
    description = "10 rounds of buying low, selling high — beat the house's ledger"
    min_players = 1
    max_players = 4
    ai_seats = 1
    move_timeout = 120
    rules = ("You start with 100 coins. Each round: 3 things for sale, "
             "3 buyers with offers. 'buy 1–3', 'sell 1–3' (from your "
             "goods), or 'wait'. Prices drift every round. After 10 "
             "rounds the richest ledger wins — and the house is "
             "trading too, and it's good at it.")

    def new_state(self, rng: random.Random) -> dict[str, Any]:
        return {"round": 1, "rounds": 10, "wallets": {}, "goods": {},
                "market": [], "buyers": [], "done": False}

    def setup(self, room, mind):
        for p in room.players:
            room.state["wallets"][p.key] = 100
            room.state["goods"][p.key] = []
        self._market(room, mind.rng)
        return (f"shop — 10 rounds, 100 coins each. "
                f"{self._market_text(room)}\n"
                f"{room.players[0].name}: 'buy N', 'sell N', or 'wait'.")

    def _market(self, room: Room, rng: random.Random) -> None:
        s = room.state
        goods = rng.sample(list(SHOP_GOODS), 3)
        s["market"] = [(n, int(f * rng.uniform(0.6, 1.4)))
                       for n, f in goods]
        buyers = rng.sample([g for g in SHOP_GOODS
                             if g[0] not in [m[0] for m in s["market"]]], 3)
        s["buyers"] = [(n, int(f * rng.uniform(0.7, 1.5)))
                       for n, f in buyers]

    def _market_text(self, room: Room) -> str:
        s = room.state
        lines = [f"round {s['round']}/{s['rounds']} — for sale:"]
        for i, (n, p) in enumerate(s["market"], 1):
            lines.append(f"  buy {i}: {n} — {p}c")
        lines.append("buyers (sell goods you hold):")
        for i, (n, p) in enumerate(s["buyers"], 1):
            lines.append(f"  sell {i}: {n} — {p}c")
        return "\n".join(lines)

    def _holdings(self, room: Room, key: str) -> str:
        goods = room.state["goods"].get(key, [])
        return ", ".join(goods) if goods else "empty"

    def _trade(self, room: Room, p: Player, text: str,
               rng: random.Random) -> list[str]:
        s = room.state
        t = text.strip().lower()
        wallet = s["wallets"][p.key]
        goods = s["goods"][p.key]
        m = re.match(r"buy\s*([1-3])$", t)
        if m:
            n = int(m.group(1)) - 1
            name, price = s["market"][n]
            if price > wallet:
                return [f"{name} is {price}c — you have {wallet}c. "
                        "sell something or wait."]
            wallet -= price
            goods.append(name)
            s["market"][n] = (name, int(price * rng.uniform(0.9, 1.3)))
            return [f"bought {name} for {price}c. "
                    f"(wallet {wallet}c, goods: {self._holdings(room, p.key)})"]
        m = re.match(r"sell\s*([1-3])$", t)
        if m:
            n = int(m.group(1)) - 1
            name, offer = s["buyers"][n]
            if name not in goods:
                return [f"you don't hold {name}. goods: "
                        f"{self._holdings(room, p.key)}"]
            goods.remove(name)
            wallet += offer
            s["buyers"][n] = (name, int(offer * rng.uniform(0.7, 1.1)))
            return [f"sold {name} for {offer}c. "
                    f"(wallet {wallet}c, goods: {self._holdings(room, p.key)})"]
        if t in {"wait", "hold", "pass"}:
            return [f"{p.name} waits. (wallet {wallet}c, goods: "
                    f"{self._holdings(room, p.key)})"]
        return ["'buy 1–3', 'sell 1–3', or 'wait'."]

    def on_move(self, room, player, text, mind):
        s = room.state
        out = self._trade(room, player, text, mind.rng)
        # the house trades every round too — a real opponent ledger
        house = [p for p in room.players if p.is_ai]
        if house:
            h = house[0]
            s = room.state
            wallet = s["wallets"][h.key]
            goods = s["goods"][h.key]
            action = None
            for i, (name, price) in enumerate(s["market"]):
                fair = dict(SHOP_GOODS).get(name, price)
                if price < fair * 0.75 and price <= wallet:
                    action = f"buy {i+1}"
                    break
            if action is None:
                for i, (name, offer) in enumerate(s["buyers"]):
                    fair = dict(SHOP_GOODS).get(name, offer)
                    if offer > fair * 1.25 and name in goods:
                        action = f"sell {i+1}"
                        break
            if action is not None:
                out.extend(self._trade(room, h, action, mind.rng))
        self._end_round(room, mind.rng)
        s = room.state
        if s["done"]:
            best = max(s["wallets"].items(), key=lambda kv: kv[1])
            w = room.player(best[0])
            out.append("🏁 final ledgers — " +
                       (f"{w.name} closes richest at {best[1]}c."
                        if w and not w.is_ai
                        else "the house's ledger holds."
                        if w and w.is_ai else "a flat market."))
        else:
            out.append(self._market_text(room))
            nxt = room.current if room.current else room.players[0]
            if nxt is not None:
                out.append(f"{nxt.name}, your trade?")
        return out

    def _end_round(self, room: Room, rng: random.Random) -> None:
        s = room.state
        s["round"] += 1
        if s["round"] > s["rounds"]:
            s["done"] = True
            return
        self._market(room, rng)

    def ai_turn(self, room, mind):
        return []  # the house trades inside every move, not on its turn

    def is_over(self, room):
        return room.state.get("done", False)

    def winner(self, room):
        best = max(room.state.get("wallets", {}).items(),
                   key=lambda kv: kv[1], default=("", 0))
        w = room.player(best[0]) if best[0] else None
        return w if (w and not w.is_ai and best[1] > 100) else \
            (w if (w and w.is_ai) else "draw")

    def score(self, room, player):
        return int(room.state.get("wallets", {}).get(player.key, 0)) - 100

    def describe_state(self, room):
        s = room.state
        return (f"round {min(s['round'], s['rounds'])}/{s['rounds']} · "
                + " · ".join(f"{room.player(k).name if room.player(k) else k}: "
                             f"{v}c" for k, v in s["wallets"].items()))


# ── 14. quiz duel ────────────────────────────────────────────────────────────

class QuizDuelGame(MultiGame):
    name = "duel"
    description = "rapid-fire 1v1 — first to 5 with streaks"
    min_players = 1
    max_players = 1
    ai_seats = 1
    move_timeout = 15
    rules = ("You vs the house, rapid fire. 15 seconds a question. "
             "Correct: +1 (streak 2+: +2). First to 5 takes the duel. "
             "The house answers at 85% and doesn't blink.")

    def new_state(self, rng: random.Random) -> dict[str, Any]:
        return {"questions": rng.sample(list(TRIVIA), 20), "idx": 0,
                "points": {}, "streak": {}, "target": 5, "done": False}

    def setup(self, room, mind):
        q, _ = room.state["questions"][0]
        return (f"quiz duel — first to 5, 15s a shot.\n"
                f"Q: {q}")

    def _next(self, room: Room) -> str | None:
        s = room.state
        s["idx"] += 1
        if s["idx"] >= len(s["questions"]) or s.get("done"):
            s["done"] = True
            return None
        q, _ = s["questions"][s["idx"]]
        return f"Q: {q}"

    def _check_win(self, room: Room) -> str | None:
        s = room.state
        for k, v in s["points"].items():
            if v >= s["target"]:
                s["done"] = True
                p = room.player(k)
                who = p.name if p else k
                return (f"🏁 {who} reaches {s['target']} — the duel is "
                        f"over. final: " +
                        " vs ".join(f"{room.player(kk).name if room.player(kk) else kk} {vv}"
                                    for kk, vv in s["points"].items()))
        return None

    def on_move(self, room, player, text, mind):
        s = room.state
        q, a = s["questions"][s["idx"]]
        right = a.lower() in text.lower() or text.lower().strip() in a.lower()
        out: list[str] = []
        if right:
            s["streak"][player.key] = int(s["streak"].get(player.key, 0)) + 1
            pts = 2 if s["streak"][player.key] >= 2 else 1
            s["points"][player.key] = int(s["points"].get(player.key, 0)) + pts
            out.append(f"correct +{pts}.")
        else:
            s["streak"][player.key] = 0
            out.append(f"miss — it was {a}.")
        win = self._check_win(room)
        if win:
            out.append(win)
            return out
        line = self._next(room)
        if line is None:
            s["done"] = True
            best = max(s["points"].items(), key=lambda kv: kv[1])
            out.append(f"the questions ran out — {best[1]} to "
                       f"{s['points'].get('ai:duel:0', 0)}.")
        else:
            out.append(line)
        return out

    def ai_turn(self, room, mind):
        s = room.state
        q, a = s["questions"][s["idx"]]
        correct = mind.rng.random() < 0.85
        out: list[str] = []
        if correct:
            s["streak"][room.current.key] = \
                int(s["streak"].get(room.current.key, 0)) + 1
            pts = 2 if s["streak"][room.current.key] >= 2 else 1
            s["points"][room.current.key] = \
                int(s["points"].get(room.current.key, 0)) + pts
            out.append(f"house: {a} +{pts}.")
        else:
            s["streak"][room.current.key] = 0
            out.append("house misses it.")
        win = self._check_win(room)
        if win:
            out.append(win)
            return out
        line = self._next(room)
        if line is None:
            s["done"] = True
            out.append("the questions ran out — final score on the "
                       "board.")
        else:
            out.append(line)
        return out

    def on_timeout(self, room, player, mind):
        out = [f"⏰ the clock ate {player.name}'s turn."]
        s = room.state
        s["streak"][player.key] = 0
        win = self._check_win(room)
        line = self._next(room)
        if win:
            out.append(win)
        elif line:
            out.append(line)
        return out

    def is_over(self, room):
        return room.state.get("done", False)

    def winner(self, room):
        best = max(room.state.get("points", {}).items(),
                   key=lambda kv: kv[1], default=("", 0))
        w = room.player(best[0]) if best[0] else None
        return w if (w and best[1] > 0) else "draw"

    def score(self, room, player):
        return int(room.state.get("points", {}).get(player.key, 0))


# ── 15. crime / investigation ────────────────────────────────────────────────

class InvestigationGame(MultiGame):
    name = "case"
    description = ("a fresh case every time — evidence, interviews, name "
                   "the culprit. 52 hand-written cases across "
                   "easy/medium/hard/expert, skill-adaptive clues, hints, "
                   "timed countdowns, and anti-repeat until the pool runs "
                   "dry. the house investigates too — and someone is lying.")
    min_players = 1
    max_players = 6
    ai_seats = 1
    move_timeout = 180
    rules = ("I open a case with four suspects — 52 hand-written cases "
             "across easy/medium/hard/expert plus fresh generated ones, "
             "graded by tier (harder pays more). 'clue' reveals the next "
             "piece of evidence, sharpest last. 'ask <name>' interviews a "
             "suspect — ask everyone, one of them is lying. 'hint' buys an "
             "escalating nudge (−2 pts each, max 3). 'accuse <name>' names "
             "the culprit — right and you close the case; wrong and the "
             "suspect walks (3 strikes and the culprit vanishes). Cases "
             "never repeat until you've seen the whole tier pool — then "
             "it reshuffles. Struggling players get a background-check "
             "clue; sharp ones get extra red herrings. Start with "
             "'/game case timed' for a countdown with a speed bonus. "
             "Solve streaks multiply your score. The house investigates "
             "alongside you — sometimes it accuses first.")

    def new_state(self, rng, **kw):
        """Deal a case for this player.

        ``kw`` carries ``history`` (the player's case history dict, or
        None → fresh) and ``timed`` (bool → countdown mode). The dealt
        case is already skill-adapted by ``deal_case``.
        """
        history = kw.get("history") or blank_history()
        timed = bool(kw.get("timed"))
        case, tier, reshuffled = deal_case(rng, history)
        state = {
            "case": case,
            "tier": tier,
            "clues_shown": 0,
            "strikes": 0,
            "max_strikes": 3,
            "asked": [],
            "done": False,
            "solved": False,
            "solver": None,
            "hints_used": 0,
            "timed": timed,
            "time_bonus": 0,
            "streak_in": int(history.get("streak") or 0),
            "reshuffled": reshuffled,
            "history": history,
        }
        if timed:
            limit = TIME_LIMITS.get(tier, 240)
            state["time_limit"] = limit
            state["deadline"] = time.time() + limit
        return state

    # ── per-player history (persisted via the engine) ──────────────────
    def load_history(self, store, player):
        """Load this player's case history from their profile."""
        try:
            prof = store.get(player.key)
            hist = (prof.per_game or {}).get("case")
            if isinstance(hist, dict) and hist.get("v") == HISTORY_VERSION:
                return hist
        except Exception:
            pass
        return blank_history()

    def save_history(self, store, player, state):
        """Record the finished game in the player's case history."""
        history = state.get("history") or blank_history()
        case = state.get("case") or {}
        tier = state.get("tier") or case.get("tier") or "medium"
        solved = state.get("solver") == player.key
        record_played(history, case.get("id", ""), tier, solved,
                      reshuffled=bool(state.get("reshuffled")))
        history["hints_taken"] = (int(history.get("hints_taken") or 0)
                                  + int(state.get("hints_used") or 0))
        if solved and state.get("timed"):
            history["timed_solved"] = int(history.get("timed_solved") or 0) + 1
        try:
            prof = store.get(player.key, name=player.name,
                             platform=player.platform)
            per_game = prof.per_game or {}
            per_game["case"] = history
            prof.per_game = per_game
            store._upsert(prof)  # same package; no public per-game setter
        except Exception:
            pass
        return history

    def setup(self, room, mind):
        c = room.state["case"]
        s = room.state
        tier = s.get("tier") or c.get("tier", "medium")
        fresh = (" — fresh from the generator, never seen before"
                 if c.get("generated") else "")
        timed = " ⏱ timed" if s.get("timed") else ""
        streak_bit = (f" · 🔥 streak {s['streak_in']}"
                      if s.get("streak_in") else "")
        lines = [
            f"the case [{tier}]{timed}{fresh} — {c['story']}",
            f"suspects: {', '.join(c['suspects'])}.",
            (f"'clue' for evidence · 'ask <name>' to interview · "
             f"'hint' (−{HINT_COST} pts) · 'accuse <name>' when you're sure."),
            f"3 wrong names and the culprit walks.{streak_bit}",
        ]
        if s.get("reshuffled"):
            lines.append("🔀 the case pool reshuffled — fresh faces again.")
        if s.get("timed"):
            lines.append(
                f"⏱ {int(s['time_limit'])}s on the clock — faster solves "
                f"earn up to +{TIME_BONUS_MAX.get(tier, 6)} bonus.")
        if c.get("assisted"):
            lines.append("📋 the file includes a background check — "
                         "you've earned the help.")
        if c.get("sharpened"):
            lines.append("🌶 the file is thicker than usual — someone's "
                         "laying false trails.")
        lines.append("the house is looking too — and one of them is lying.")
        return "\n".join(lines)

    def _reveal_clue(self, room):
        s = room.state
        c = s["case"]
        if s["clues_shown"] >= len(c["clues"]):
            return ["that's the whole file — all "
                    f"{len(c['clues'])} pieces. time to 'accuse <name>'."]
        s["clues_shown"] += 1
        return [f"📎 clue {s['clues_shown']}/{len(c['clues'])}: "
                f"{c['clues'][s['clues_shown'] - 1]}"]

    def _interview(self, room, name, asked_by):
        s = room.state
        c = s["case"]
        target = next((x for x in c["suspects"]
                       if name and x.lower() in name.lower()), None)
        if not target:
            return ["who? " + " ".join(c["suspects"])]
        line = c["statements"][target]
        tag = "🎙 " if target not in s["asked"] else "🎙 (again) "
        if target not in s["asked"]:
            s["asked"].append(target)
        return [f"{tag}{target}: \"{line}\""]

    def _accuse(self, room, name, player):
        s = room.state
        c = s["case"]
        accuser = player.name
        suspect = next((x for x in c["suspects"]
                        if name and x.lower() in name.lower()), None)
        if not suspect:
            return ["name a suspect: " + " ".join(c["suspects"])]
        if suspect == c["culprit"]:
            s["done"] = True
            s["solved"] = True
            s["solver"] = player.key
            bonus = 0
            if s.get("timed") and s.get("deadline"):
                left = s["deadline"] - time.time()
                if left > 0:
                    tier = s.get("tier") or c.get("tier", "medium")
                    frac = left / max(1, s.get("time_limit", 1))
                    bonus = int(round(TIME_BONUS_MAX.get(tier, 6) * frac))
            s["time_bonus"] = bonus
            msg = (f"🏁 {accuser} closes the case — it was **{suspect}**.")
            if bonus:
                msg += f" ⏱ +{bonus} time bonus."
            return [msg + " the file is signed."]
        s["strikes"] += 1
        if s["strikes"] >= s["max_strikes"]:
            s["done"] = True
            return [f"❌ {accuser} blunders — {suspect} walks, and the "
                    f"culprit slips away. it was {c['culprit']}."]
        return [f"❌ {accuser}: not {suspect}. "
                f"strike {s['strikes']}/{s['max_strikes']}."]

    def _hint(self, room):
        s = room.state
        c = s["case"]
        used = int(s.get("hints_used") or 0)
        if used >= 3:
            return ["the file's dry — no more hints on this case."]
        s["hints_used"] = used + 1
        if used == 0:
            return [f"🕵️ hint (−{HINT_COST} pts): two of the four have "
                    "airtight alibis — find them in the evidence, then "
                    "look at who's left."]
        if used == 1:
            hs = c.get("herring_suspect") or "someone"
            return [f"🕵️ hint (−{HINT_COST} pts): stop chasing {hs} — "
                    "that trail goes cold. re-read the evidence."]
        return [f"🕵️ hint (−{HINT_COST} pts): interview everyone — exactly "
                "one statement contradicts the evidence. that's your liar."]

    def on_move(self, room, player, text, mind):
        t = (text or "").strip().lower()
        if t in ("clue", "evidence", "look"):
            return self._reveal_clue(room)
        if t.startswith("ask"):
            return self._interview(room, t[3:], player.name)
        if t == "hint":
            return self._hint(room)
        if t.startswith("accuse"):
            return self._accuse(room, t[6:], player)
        return ["say 'clue', 'ask <name>', 'hint', or 'accuse <name>'."]

    def ai_turn(self, room, mind):
        c = room.state["case"]
        s = room.state
        known = s["clues_shown"]
        asked = set(s["asked"])
        all_sus = list(c["suspects"])
        if known < len(c["clues"]):
            msgs = self._reveal_clue(room)
        else:
            todo = [x for x in all_sus if x not in asked]
            msgs = self._interview(room, todo[0], "house") if todo else []
        # If the smoking gun is on the table and we've interviewed
        # everyone, the house puts it together.
        if (s["clues_shown"] == len(c["clues"])
                and len(s["asked"]) == len(all_sus)):
            msgs = msgs + self._accuse(room, c["culprit"], room.current)
        return msgs

    def is_over(self, room):
        return bool(room.state["done"])

    def winner(self, room):
        s = room.state
        if s.get("solver"):
            for p in room.players:
                if p.key == s["solver"]:
                    return p
        return None

    def final_message(self, room, mind):
        s = room.state
        c = s["case"]
        if s["done"] and s["strikes"] < s["max_strikes"]:
            return ("🏁 case closed — the detective who solved it gets the "
                    "glory, the evidence, and the points.")
        return super().final_message(room, mind)

    def score(self, room, player):
        s = room.state
        c = s.get("case") or {}
        tier = s.get("tier") or c.get("tier") or c.get("difficulty", "medium")
        if not s.get("solved"):
            # show up: 1 if they interviewed, else 0
            return 1 if s.get("asked") else 0
        return score_solve(tier, s.get("strikes", 0), s.get("hints_used", 0),
                           s.get("streak_in", 0), s.get("time_bonus", 0))

    def describe_state(self, room):
        s = room.state
        c = s.get("case") or {}
        asked = s.get("asked", [])
        bits = [f"evidence: {s.get('clues_shown', 0)}/"
                f"{len(c.get('clues', ()))}",
                f"interviewed: {len(asked)}/{len(c.get('suspects', ()))}",
                f"strikes: {s.get('strikes', 0)}/{s.get('max_strikes', 3)}"]
        if s.get("hints_used"):
            bits.append(f"hints: {s['hints_used']}")
        if s.get("timed") and s.get("deadline"):
            bits.append(f"⏱ {max(0, int(s['deadline'] - time.time()))}s left")
        if s.get("streak_in"):
            bits.append(f"🔥 streak {s['streak_in']}")
        return " · ".join(bits)


MEDIUM_GAMES: tuple[MultiGame, ...] = (
    MafiaGame(), KingOfHillGame(), StoryChainGame(), RpgAdventureGame(),
    ShopGame(), QuizDuelGame(), InvestigationGame(),
)
