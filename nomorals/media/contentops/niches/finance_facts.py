"""Niche: finance_facts — contrarian money facts with a "why it matters" payoff."""

from __future__ import annotations

from .base import NichePlugin, VisualPlan, VoiceSpec
from .registry import register


class FinanceFactsNiche(NichePlugin):
    name = "finance_facts"
    thesis = "One counterintuitive money fact, then exactly why it matters to you."
    cadence = 1.0
    target_seconds = (25, 50)

    def script_prompt(self, topic: str) -> str:
        lo, hi = self.word_budget()
        return f"""You are the writer for a finance channel that turns boring money facts into shorts people send to their group chats. Your superpower: making a number feel like a plot twist.

TOPIC: {topic}

Write a spoken-word script, {lo}–{hi} words, for a vertical short.

VOICE: sharp, a little conspiratorial — like a friend who works at a bank and is telling you what they actually think. Numbers are characters, not decoration: say them slowly, let them land. No jargon without an instant plain-English translation.

RETENTION STRUCTURE (mandatory):
1. HOOK (first line): the fact as a paradox — "X sounds impossible. The math says otherwise." Lead with the number, not the setup.
2. THE STORY OF THE FACT (40%): where it comes from, in one tight anecdote or mechanism. One re-hook halfway: "But here's the part nobody mentions—"
3. THE "SO WHAT" (70%): translate it into the viewer's wallet — what changes for someone earning an average salary. This is the save/share moment.
4. PAYOFF: one sentence that reframes how they see money, looping back to the hook number.

RULES:
- Plain spoken text only. No stage directions, no [brackets], no emojis, no "hey guys".
- Never invent statistics. If the topic implies a number you can't verify, phrase it as an illustration ("imagine", "say") rather than a fact.
- No financial advice ("you should invest in X"). Facts and frameworks only; end with "do your own research" energy, not a disclaimer paragraph.
- One idea per script.

Return ONLY the script text, paragraphs separated by blank lines."""

    _STYLE = (
        "premium editorial 3D render, dark navy and gold palette, floating "
        "coins and banknotes frozen mid-air, dramatic studio lighting, "
        "shallow depth of field, ultra-detailed"
    )

    _SCENES = (
        "giant glowing number made of gold coins rising out of a dark stock chart, cinematic",
        "close-up of a hand holding a phone showing a banking app, holographic graphs rising from the screen",
        "split scene: modest apartment on the left, luxury penthouse on the right, connected by a golden thread of coins",
        "vault door opening with light pouring out, stacks of cash and gold bars inside, epic",
    )

    def visual_strategy(self, script: str) -> VisualPlan:
        return self._beats_to_scenes(
            script,
            self._SCENES,
            style_lock=self._STYLE,
            motions=("zoom-in-slow", "pan-right-slow", "zoom-out-slow", "zoom-in-slow"),
            notes="Navy/gold money aesthetic throughout. Numbers are the hero of every frame.",
        )

    title_template = "{topic} — The Money Fact Nobody Talks About"
    description_template = (
        "{topic}\n\n"
        "The number sounds fake. The math doesn't care. Save this one.\n\n"
        "{script}"
    )
    hashtags = {
        "tiktok": ["#moneytok", "#finance", "#moneyfacts", "#investing", "#wealth"],
        "youtube": ["#finance", "#money", "#investing", "#shorts"],
        "instagram": ["#financetips", "#moneymindset", "#wealthbuilding", "#reels"],
        "default": ["#finance", "#moneyfacts"],
    }
    voice_spec = VoiceSpec(
        preferred_voice="devon-analyst",
        mood="confident",
        intensity=3,
        pace_wpm=155,
        pause_style="natural",
    )

    ypp_rationale = (
        "Original commentary on public facts: AI-written script with the "
        "niche's own framing, AI-generated 3D visuals, Devon's own voice. "
        "No repurposed clips, no reading someone else's article aloud. "
        "Educational and transformative — YPP-safe."
    )


register(FinanceFactsNiche())
