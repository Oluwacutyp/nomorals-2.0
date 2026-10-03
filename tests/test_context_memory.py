"""Test ContextEngine memory integration."""
import unittest
from types import SimpleNamespace

from nomorals.context.engine import ContextEngine


class FakeMemory:
    """Minimal MemoryManager stand-in."""
    def __init__(self, text=""):
        self._text = text

    def build_context(self, query):
        return self._text


class MemorySectionTest(unittest.TestCase):
    def test_memory_section_included(self):
        engine = ContextEngine()
        mem = FakeMemory("Relevant memory:\n- [fact] user likes coffee")
        built = engine.build(memory=mem, memory_query="coffee")
        names = [s.name for s in built.sections]
        self.assertIn("memory", names)

    def test_memory_section_empty_when_no_memory(self):
        engine = ContextEngine()
        built = engine.build()
        names = [s.name for s in built.sections]
        self.assertNotIn("memory", names)

    def test_memory_section_empty_text(self):
        engine = ContextEngine()
        mem = FakeMemory("")
        built = engine.build(memory=mem)
        # Section exists but content is empty
        mem_section = built.section("memory")
        self.assertIsNotNone(mem_section)
        self.assertEqual(mem_section.content, "")

    def test_memory_section_uses_build_context(self):
        """Verifies we call MemoryManager.build_context (unified path)."""
        engine = ContextEngine()
        mem = FakeMemory("memory text here")
        built = engine.build(memory=mem, memory_query="test query")
        mem_section = built.section("memory")
        self.assertIn("memory text here", mem_section.content)


if __name__ == "__main__":
    unittest.main()
