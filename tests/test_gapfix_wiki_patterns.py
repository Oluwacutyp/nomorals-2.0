"""Wiki patterns (lane 126, census §3B.3): the craft-wiki patterns the
first port left out — subcategory + cover frontmatter, parent
validation, the nested sidebar tree, prev/next pagination, the
canonical five, and the sitemap.

craft-wiki carries NO licence, so this is original Devon code
implementing mined patterns, never a copy of its text or files. The
tool count stays 774 — the secprobe wire-up added 25 tools; the new 775 after this lane (B.19) added scene_spawn. 776 after this lane (B.20) added cloudcam. 777 after this lane (B.22) added show. 778 after this lane (B.26) added scene_project. 779 after this lane (B.23) added facemcap. 780 after this lane (B.25) added facemesh. 781 after this lane (B.27) added vtuber. 782 after this lane (C.2) registered tonal_g2p; 783 after this lane (C.1) added smarthome; 784 after this lane (C.4) added camwall; 785 after this lane (C.8) added robot; 786 after this lane (audit N4) added build_swarm; 787 after gapfix74 added museum_library.
actions ride the existing action-dispatched ``wiki`` tool (774 =
749 + 25 secprobe tools; 749 = 748 at the lane's base + 1 mocap
tool, lane 125).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.wiki.nav import (
    CANONICAL_PAGES,
    canonical_missing,
    dangling_parents,
    flatten_tree,
    neighbors,
    sidebar_tree,
    sitemap,
)
from nomorals.wiki.pages import (
    parse_frontmatter,
    validate_cover,
)
from nomorals.wiki.render import render_chat, render_html
from nomorals.wiki.store import WikiStore, wiki_path
from nomorals.wiki.surface import (
    render_board,
    render_canonical,
    render_sitemap,
)
from nomorals.wiki.templates import canonical_body


def _store(tmp: str) -> WikiStore:
    return WikiStore(wiki_path(SimpleNamespace(home_path=Path(tmp))))


def _seed_nav(store: WikiStore) -> dict[str, str]:
    """A small tree: overview + two guides + a child group."""
    slugs = {}
    slugs["overview"] = store.create(
        "Guide Overview", "The landing page.\n", type="overview",
        category="Guides", order=0).slug
    slugs["start"] = store.create(
        "Getting Started", "Start here.\n", type="guide",
        category="Guides", order=1).slug
    slugs["deep"] = store.create(
        "Deep Dive", "Details.\n", type="guide", category="Guides",
        subcategory="Advanced", order=2).slug
    slugs["deeper"] = store.create(
        "Deeper Dive", "More details.\n", type="guide",
        category="Guides", subcategory="Advanced", order=3).slug
    slugs["plain"] = store.create(
        "Plain Note", "No subcategory.\n", type="guide",
        category="Guides", order=4).slug
    slugs["fix"] = store.create(
        "Fix It", "Problem | Symptom | Fix\n", type="troubleshooting",
        category="Guides", order=5).slug
    return slugs


class TestFrontmatterFields(unittest.TestCase):
    def test_subcategory_and_cover_round_trip(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-test-wiki-") as tmp:
            store = _store(tmp)
            page = store.create(
                "Covered", "Body.\n", type="guide", category="Docs",
                subcategory="Setup",
                cover="/assets/uploads/docs/hero.webp")
            self.assertEqual(page.subcategory, "Setup")
            self.assertEqual(page.cover, "/assets/uploads/docs/hero.webp")
            md = store.export_markdown(page.slug)
            self.assertIn("subcategory: Setup", md)
            self.assertIn("cover: /assets/uploads/docs/hero.webp", md)
            fields, _ = parse_frontmatter(md)
            self.assertEqual(fields["subcategory"], "Setup")
            self.assertEqual(fields["cover"],
                             "/assets/uploads/docs/hero.webp")

    def test_bad_cover_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            validate_cover("/assets/other/x.png")
        with self.assertRaises(ValueError):
            validate_cover("/assets/uploads/docs/hero.gif")
        with self.assertRaises(ValueError):
            validate_cover("https://example.com/hero.png")
        self.assertEqual(validate_cover(""), "")
        self.assertEqual(validate_cover("/assets/uploads/a/b.jpg"),
                         "/assets/uploads/a/b.jpg")

    def test_create_rejects_bad_cover(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-test-wiki-") as tmp:
            store = _store(tmp)
            with self.assertRaises(ValueError):
                store.create("Bad", "Body.\n", cover="hero.png")

    def test_import_rejects_bad_cover(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-test-wiki-") as tmp:
            store = _store(tmp)
            with self.assertRaises(ValueError):
                store.import_markdown(
                    "---\ntitle: Bad\ncover: hero.png\n---\nBody.\n")

    def test_prev_next_round_trip(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-test-wiki-") as tmp:
            store = _store(tmp)
            page = store.create("A", "Body.\n", prev="x", next="y")
            self.assertEqual(page.prev, "x")
            self.assertEqual(page.next, "y")
            md = store.export_markdown(page.slug)
            self.assertIn("prev: x", md)
            self.assertIn("next: y", md)


class TestParentValidation(unittest.TestCase):
    def test_parent_must_exist(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-test-wiki-") as tmp:
            store = _store(tmp)
            with self.assertRaises(ValueError):
                store.create("Child", "Body.\n", parent="no-such-page")

    def test_parent_must_be_top_level(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-test-wiki-") as tmp:
            store = _store(tmp)
            top = store.create("Top", "Body.\n", type="overview")
            child = store.create("Child", "Body.\n", parent=top.slug)
            with self.assertRaises(ValueError):
                store.create("Grandchild", "Body.\n", parent=child.slug)

    def test_valid_parent_and_reparent(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-test-wiki-") as tmp:
            store = _store(tmp)
            top = store.create("Top", "Body.\n", type="overview")
            other = store.create("Other", "Body.\n", type="overview")
            child = store.create("Child", "Body.\n", parent=top.slug)
            self.assertEqual(child.parent, top.slug)
            self.assertTrue(store.reparent(child.slug, other.slug))
            self.assertEqual(store.get(child.slug).parent, other.slug)
            self.assertTrue(store.reparent(child.slug, ""))
            self.assertEqual(store.get(child.slug).parent, "")

    def test_reparent_to_missing_parent_raises(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-test-wiki-") as tmp:
            store = _store(tmp)
            page = store.create("Solo", "Body.\n")
            with self.assertRaises(ValueError):
                store.reparent(page.slug, "ghost")
            self.assertFalse(store.reparent("ghost", ""))


class TestSidebarTree(unittest.TestCase):
    def test_landing_first_and_troubleshooting_last(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-test-wiki-") as tmp:
            store = _store(tmp)
            slugs = _seed_nav(store)
            pages = store.list_pages("Guides")
            self.assertEqual(pages[0].slug, slugs["overview"])
            self.assertEqual(pages[-1].slug, slugs["fix"])

    def test_grouping_plain_then_subcategory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-test-wiki-") as tmp:
            store = _store(tmp)
            slugs = _seed_nav(store)
            tree = sidebar_tree(store)
            # All six are top-level (no parents); the tree holds them
            # in sidebar order.
            flat = [p.slug for p in flatten_tree(tree)]
            self.assertEqual(
                flat,
                [slugs["overview"], slugs["start"], slugs["deep"],
                 slugs["deeper"], slugs["plain"], slugs["fix"]])

    def test_children_group_by_category_and_subcategory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-test-wiki-") as tmp:
            store = _store(tmp)
            top = store.create("Hub", "Hub.\n", type="overview")
            store.create("Setup One", "One.\n", parent=top.slug,
                         category="Setup", order=2)
            store.create("Setup Two", "Two.\n", parent=top.slug,
                         category="Setup", subcategory="Deep", order=1)
            store.create("Setup Three", "Three.\n", parent=top.slug,
                         category="Setup", subcategory="Deep", order=3)
            store.create("Loose", "Loose.\n", parent=top.slug, order=0)
            tree = sidebar_tree(store)
            self.assertEqual(len(tree), 1)
            items = tree[0]["items"]
            # Loose (no category) first, then the Setup group.
            self.assertIn("page", items[0])
            self.assertEqual(items[0]["page"].title, "Loose")
            self.assertEqual(items[1]["label"], "Setup")
            inner = items[1]["items"]
            # Plain Setup page first, then the Deep subcategory group.
            self.assertIn("page", inner[0])
            self.assertEqual(inner[0]["page"].title, "Setup One")
            self.assertEqual(inner[1]["label"], "Deep")
            self.assertEqual(
                [p.title for p in
                 [i["page"] for i in inner[1]["items"]]],
                ["Setup Two", "Setup Three"])

    def test_single_subcategory_collapses(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-test-wiki-") as tmp:
            store = _store(tmp)
            top = store.create("Hub", "Hub.\n", type="overview")
            store.create("Only Deep", "Deep.\n", parent=top.slug,
                         category="Setup", subcategory="Deep", order=1)
            store.create("Also Deep", "Deep.\n", parent=top.slug,
                         category="Setup", subcategory="Deep", order=2)
            items = sidebar_tree(store)[0]["items"]
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["label"], "Setup")
            inner = items[0]["items"]
            # No subcategory group heading — the pages list directly.
            self.assertTrue(all("page" in i for i in inner))
            self.assertEqual([i["page"].title for i in inner],
                             ["Only Deep", "Also Deep"])

    def test_landing_page_leads_its_group(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-test-wiki-") as tmp:
            store = _store(tmp)
            top = store.create("Hub", "Hub.\n", type="overview")
            store.create("Guide A", "A.\n", parent=top.slug,
                         category="Setup", order=2)
            store.create("Setup Overview", "Landing.\n", parent=top.slug,
                         type="overview", category="Setup", order=1)
            items = sidebar_tree(store)[0]["items"]
            self.assertEqual(items[0]["label"], "Setup")
            self.assertEqual(items[0]["landing"].title, "Setup Overview")
            flat = [p.slug for p in flatten_tree(sidebar_tree(store))]
            self.assertEqual(flat[1], "setup-overview")

    def test_plain_pages_come_before_subcategory_groups(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-test-wiki-") as tmp:
            store = _store(tmp)
            top = store.create("Hub", "Hub.\n", type="overview")
            # A troubleshooting page with no subcategory is a plain page
            # of its category, so it lists before the subcategory group
            # — craft-wiki's groupChildren puts plain pages first and
            # sorts troubleshooting last only inside one list.
            store.create("Fix It", "Fix.\n", parent=top.slug,
                         type="troubleshooting", category="Setup",
                         order=3)
            store.create("Guide A", "A.\n", parent=top.slug,
                         category="Setup", subcategory="Deep", order=1)
            flat = [p.slug for p in flatten_tree(sidebar_tree(store))]
            self.assertEqual(flat, ["hub", "fix-it", "guide-a"])

    def test_dangling_parent_lists_at_top(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-test-wiki-") as tmp:
            store = _store(tmp)
            parent = store.create("Parent", "P.\n", type="overview")
            child = store.create("Child", "C.\n", parent=parent.slug)
            store.delete(parent.slug)
            self.assertEqual(
                dangling_parents(store),
                [{"from": child.slug, "parent": parent.slug}])
            flat = [p.slug for p in flatten_tree(sidebar_tree(store))]
            self.assertIn(child.slug, flat)


class TestNeighbors(unittest.TestCase):
    def test_first_middle_last(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-test-wiki-") as tmp:
            store = _store(tmp)
            slugs = _seed_nav(store)
            prev, nxt = neighbors(store, slugs["overview"])
            self.assertIsNone(prev)
            self.assertEqual(nxt.slug, slugs["start"])
            prev, nxt = neighbors(store, slugs["deep"])
            self.assertEqual(prev.slug, slugs["start"])
            self.assertEqual(nxt.slug, slugs["deeper"])
            prev, nxt = neighbors(store, slugs["fix"])
            self.assertEqual(prev.slug, slugs["plain"])
            self.assertIsNone(nxt)

    def test_explicit_override_wins_when_it_resolves(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-test-wiki-") as tmp:
            store = _store(tmp)
            slugs = _seed_nav(store)
            page = store.get(slugs["start"])
            page.next = slugs["fix"]
            prev, nxt = neighbors(store, slugs["start"])
            self.assertEqual(nxt.slug, slugs["fix"])
            self.assertEqual(prev.slug, slugs["overview"])

    def test_dangling_override_falls_back_to_sidebar(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-test-wiki-") as tmp:
            store = _store(tmp)
            slugs = _seed_nav(store)
            page = store.get(slugs["start"])
            page.next = "ghost"
            _, nxt = neighbors(store, slugs["start"])
            self.assertEqual(nxt.slug, slugs["deep"])

    def test_missing_page_has_no_neighbors(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-test-wiki-") as tmp:
            store = _store(tmp)
            self.assertEqual(neighbors(store, "ghost"), (None, None))


class TestCanonicalAndSitemap(unittest.TestCase):
    def test_canonical_pages_are_five_in_order(self) -> None:
        self.assertEqual(
            CANONICAL_PAGES,
            ("overview", "getting-started", "install-update",
             "workflows", "troubleshooting"))

    def test_canonical_body_for_each(self) -> None:
        for slug in CANONICAL_PAGES:
            body = canonical_body(slug)
            self.assertTrue(body.strip())
        self.assertIn("Install", canonical_body("install-update"))
        self.assertIn("Update", canonical_body("install-update"))
        self.assertIn("workflows", canonical_body("workflows").lower())
        with self.assertRaises(ValueError):
            canonical_body("not-a-page")

    def test_canonical_missing(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-test-wiki-") as tmp:
            store = _store(tmp)
            self.assertEqual(canonical_missing(store),
                             list(CANONICAL_PAGES))
            store.create("Overview", "O.\n", type="overview",
                         category="Docs")
            store.create("Getting Started", "G.\n", type="guide",
                         category="Docs")
            self.assertEqual(
                canonical_missing(store, "Docs"),
                ["install-update", "workflows", "troubleshooting"])
            self.assertIn("overview",
                          canonical_missing(store, "Other"))

    def test_sitemap_lists_every_page_in_sidebar_order(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-test-wiki-") as tmp:
            store = _store(tmp)
            slugs = _seed_nav(store)
            urls = sitemap(store)
            self.assertEqual(
                urls,
                [f"/wiki/{slugs[s]}" for s in
                 ("overview", "start", "deep", "deeper", "plain", "fix")])
            self.assertIn("/wiki/guide-overview", render_sitemap(store))


class TestRenders(unittest.TestCase):
    def test_chat_render_shows_cover_and_nav(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-test-wiki-") as tmp:
            store = _store(tmp)
            slugs = _seed_nav(store)
            page = store.get(slugs["start"])
            page.cover = "/assets/uploads/guides/hero.png"
            out = render_chat(store, page)
            self.assertIn("Cover: /assets/uploads/guides/hero.png", out)
            self.assertIn("Previous: Guide Overview", out)
            self.assertIn("Next: Deep Dive", out)

    def test_html_render_shows_cover_img_and_nav_links(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-test-wiki-") as tmp:
            store = _store(tmp)
            slugs = _seed_nav(store)
            page = store.get(slugs["start"])
            page.cover = "/assets/uploads/guides/hero.png"
            out = render_html(store, page)
            self.assertIn('<img src="/assets/uploads/guides/hero.png"',
                          out)
            self.assertIn(f"/wiki/{slugs['overview']}", out)
            self.assertIn(f"/wiki/{slugs['deep']}", out)

    def test_board_groups_children(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-test-wiki-") as tmp:
            store = _store(tmp)
            top = store.create("Hub", "Hub.\n", type="overview")
            store.create("Setup One", "One.\n", parent=top.slug,
                         category="Setup", order=2)
            store.create("Setup Two", "Two.\n", parent=top.slug,
                         category="Setup", subcategory="Deep", order=1)
            out = render_board(store)
            self.assertIn("Hub", out)
            self.assertIn("[Setup]", out)
            self.assertIn("[Deep]", out)
            self.assertIn("Setup One", out)

    def test_canonical_render_names_missing(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-test-wiki-") as tmp:
            store = _store(tmp)
            out = render_canonical(store, "Docs")
            self.assertIn("install-update", out)
            self.assertIn("troubleshooting", out)
            store.create("Overview", "O.\n", type="overview",
                         category="Docs")
            self.assertNotIn(
                "overview", render_canonical(store, "Docs"))


class TestToolAndVerb(unittest.TestCase):
    def _reg(self, tmp):
        from nomorals.tools import wiki as wiki_tool
        from nomorals.tools.registry import ToolRegistry

        reg = ToolRegistry(SimpleNamespace(
            settings=SimpleNamespace(home_path=Path(tmp)),
            db=None, router=None))
        wiki_tool.register(reg)
        return reg

    def _call(self, reg, **kw):
        return reg._tools["wiki"].fn(**kw)

    def test_tool_count_stays_778(self) -> None:
        from nomorals.tools.registry import ToolRegistry

        registry = ToolRegistry(SimpleNamespace())
        registry.register_builtins()
        # Lane 126 adds no tool: nav/sitemap/canonical ride the
        # existing action-dispatched "wiki" tool. 774 = 749 + 25 secprobe
        # tools; 749 = 748 at the lane's base + 1 mocap tool
        # (lane 125), which landed between the lane's base and
        # the rebase. 778 after B.26 added scene_project. 779 after this lane (B.23) added facemcap. 780 after this lane (B.25) added facemesh. 781 after this lane (B.27) added vtuber. 782 after this lane (C.2) registered tonal_g2p; 783 after this lane (C.1) added smarthome; 784 after this lane (C.4) added camwall; 785 after this lane (C.8) added robot; 786 after this lane (audit N4) added build_swarm; 787 after gapfix74 added museum_library.
        self.assertEqual(len(registry._tools), 787)
        self.assertIn("wiki", registry._tools)

    def test_tool_nav_sitemap_canonical(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-test-wiki-") as tmp:
            reg = self._reg(tmp)
            store = _store(tmp)
            slugs = _seed_nav(store)
            nav = self._call(reg, action="nav", slug=slugs["start"])
            self.assertTrue(nav["ok"])
            self.assertEqual(nav["prev"], slugs["overview"])
            self.assertEqual(nav["next"], slugs["deep"])
            self.assertIn("Previous:", nav["render"])
            smap = self._call(reg, action="sitemap")
            self.assertTrue(smap["ok"])
            self.assertEqual(smap["urls"][0], f"/wiki/{slugs['overview']}")
            canon = self._call(reg, action="canonical", category="Guides")
            self.assertTrue(canon["ok"])
            self.assertEqual(
                canon["missing"],
                ["overview", "install-update", "workflows",
                 "troubleshooting"])
            self.assertEqual(canon["dangling_parents"], [])

    def test_tool_canonical_new_and_reparent(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-test-wiki-") as tmp:
            reg = self._reg(tmp)
            out = self._call(reg, action="canonical-new",
                             slug="install-update", category="Docs")
            self.assertTrue(out["ok"])
            self.assertEqual(out["slug"], "install-update")
            self.assertIn("Install", out["page"]["body"])
            again = self._call(reg, action="canonical-new",
                               slug="install-update", category="Docs")
            self.assertFalse(again["ok"])
            bad = self._call(reg, action="canonical-new", slug="nope")
            self.assertFalse(bad["ok"])
            created = self._call(reg, action="create", title="Kid",
                                 body="Kid.\n")
            top = self._call(reg, action="create", title="Top",
                             body="Top.\n", type="overview")
            moved = self._call(reg, action="reparent",
                               slug=created["slug"], parent=top["slug"])
            self.assertTrue(moved["ok"])
            self.assertEqual(moved["page"]["parent"], top["slug"])
            bad_parent = self._call(reg, action="reparent",
                                    slug=created["slug"], parent="ghost")
            self.assertFalse(bad_parent["ok"])

    def test_tool_create_with_subcategory_and_cover(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nm-test-wiki-") as tmp:
            reg = self._reg(tmp)
            out = self._call(reg, action="create", title="Covered",
                             body="Body.\n", subcategory="Setup",
                             cover="/assets/uploads/docs/hero.webp")
            self.assertTrue(out["ok"])
            self.assertEqual(out["page"]["subcategory"], "Setup")
            self.assertEqual(out["page"]["cover"],
                             "/assets/uploads/docs/hero.webp")
            bad = self._call(reg, action="create", title="Bad",
                             body="Body.\n", cover="hero.png")
            self.assertFalse(bad["ok"])
            self.assertIn("cover", bad["error"])

    def test_verb_nav_sitemap_canonical(self) -> None:
        from nomorals.agents.partner.runtime_system import (
            RuntimeSystemMixin,
        )

        with tempfile.TemporaryDirectory(prefix="nm-test-wiki-") as tmp:
            store = _store(tmp)
            slugs = _seed_nav(store)
            runtime = object.__new__(RuntimeSystemMixin)
            runtime._wiki_store = lambda: store
            nav = runtime._control_wiki(f"nav {slugs['start']}",
                                        chat_key="telegram:1")
            self.assertIn("Previous: Guide Overview", nav)
            self.assertIn("Next: Deep Dive", nav)
            smap = runtime._control_wiki("sitemap", chat_key="telegram:1")
            self.assertIn("/wiki/guide-overview", smap)
            canon = runtime._control_wiki("canonical Guides",
                                          chat_key="telegram:1")
            self.assertIn("install-update", canon)
            self.assertIn("workflows", canon)
            new = runtime._control_wiki(
                "canonical-new install-update Guides",
                chat_key="telegram:1")
            self.assertIn("install-update", new)
            self.assertIn("started from the install-update skeleton", new)
            moved = runtime._control_wiki(
                f"reparent {slugs['plain']} {slugs['overview']}",
                chat_key="telegram:1")
            self.assertIn("now sits under", moved)


if __name__ == "__main__":
    unittest.main()
