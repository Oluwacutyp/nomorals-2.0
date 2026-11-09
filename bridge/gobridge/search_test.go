package main

// Bridge-level pins for the search verb (parity bundle
// #11).  The filter is pure, so it is pinned directly:
// case-insensitive substring over the message text, the
// most recent matches kept in remembered order, and the
// total reported uncapped — a swapped cap or a total that
// quietly equals the page size would misreport coverage
// to the owner, the one thing this verb exists to state
// honestly.

import (
	"strings"
	"testing"
)

func searchFixture() []*remembered {
	texts := []string{
		"Morning standup moved to 10",
		"the INVOICE for October is ready",
		"lunch?",
		"Invoice paid, thank you",
		"see the attached invoice copy",
		"unrelated entirely",
	}
	arr := make([]*remembered, 0, len(texts))
	for i, text := range texts {
		arr = append(arr, &remembered{
			text: text, tsMs: int64(1000 * (i + 1))})
	}
	return arr
}

func TestSearchMatchesCaseInsensitively(t *testing.T) {
	matches, total := searchRemembered(searchFixture(),
		"invoice", 20)
	if total != 3 || len(matches) != 3 {
		t.Fatalf("want 3 invoice matches, got total=%d len=%d",
			total, len(matches))
	}
	// remembered order preserved (oldest first)
	if !strings.Contains(matches[0].text, "INVOICE") ||
		!strings.Contains(matches[2].text, "attached") {
		t.Fatalf("match order wrong: %q … %q",
			matches[0].text, matches[2].text)
	}
}

func TestSearchCapKeepsMostRecentAndReportsTotal(t *testing.T) {
	matches, total := searchRemembered(searchFixture(),
		"invoice", 2)
	if total != 3 {
		t.Fatalf("total must stay 3 under a cap, got %d", total)
	}
	if len(matches) != 2 ||
		!strings.Contains(matches[0].text, "paid") ||
		!strings.Contains(matches[1].text, "attached") {
		t.Fatalf("cap must keep the most recent matches: %v",
			matches)
	}
}

func TestSearchEmptyQueryAndNoMatch(t *testing.T) {
	for _, q := range []string{"", "   "} {
		matches, total := searchRemembered(searchFixture(),
			q, 20)
		if matches != nil || total != 0 {
			t.Fatalf("empty query %q must match nothing", q)
		}
	}
	matches, total := searchRemembered(searchFixture(),
		"zzz-no-hit", 20)
	if len(matches) != 0 || total != 0 {
		t.Fatalf("no-hit query: got len=%d total=%d",
			len(matches), total)
	}
	matches, total = searchRemembered(nil, "invoice", 20)
	if len(matches) != 0 || total != 0 {
		t.Fatalf("empty memory: got len=%d total=%d",
			len(matches), total)
	}
}

func TestSearchEntryShape(t *testing.T) {
	entry := rememberedEntry(&remembered{
		text: "hello", sender: "Ada", tsMs: 1234})
	for _, key := range []string{
		"id", "incoming", "text", "sender", "media",
		"reply_to", "ts"} {
		if _, ok := entry[key]; !ok {
			t.Fatalf("search/history row missing %q", key)
		}
	}
	if entry["text"] != "hello" || entry["sender"] != "Ada" ||
		entry["ts"] != int64(1234) {
		t.Fatalf("entry fields wrong: %v", entry)
	}
}
