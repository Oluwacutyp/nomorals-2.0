package main

// Bridge-level pins for the status-depth verbs (parity #13):
// the audience list (status_audience / status_audience_set)
// and the viewer memory (status_viewers).  The audience
// logic runs over a fake client seam, like markStatusViewed;
// the node builder, the verdict normalization and the shape
// rule are pure.  The write is proven by read-back — a fake
// whose read disagrees with the write must surface the
// READ values, never the requested ones.

import (
	"context"
	"errors"
	"strconv"
	"testing"

	waBinary "go.mau.fi/whatsmeow/binary"
	"go.mau.fi/whatsmeow/types"
)

var (
	audAda  = types.NewJID("2348011111111", types.DefaultUserServer)
	audBola = types.NewJID("2348022222222", types.DefaultUserServer)
)

type fakeStatusAudienceClient struct {
	lists    []types.StatusPrivacy
	readErr  error
	writeErr error
	gotNode  *waBinary.Node
	// afterWrite, when set, replaces lists once a write
	// lands — the server's own read-back, which is what
	// the verb must report.
	afterWrite []types.StatusPrivacy
}

func (f *fakeStatusAudienceClient) GetStatusPrivacy(
	_ context.Context) ([]types.StatusPrivacy, error) {
	return f.lists, f.readErr
}

func (f *fakeStatusAudienceClient) SetStatusPrivacyList(
	_ context.Context, list waBinary.Node) error {
	f.gotNode = &list
	if f.writeErr == nil && f.afterWrite != nil {
		f.lists = f.afterWrite
	}
	return f.writeErr
}

func TestStatusAudienceVerdictPicksTheDefaultList(t *testing.T) {
	lists := []types.StatusPrivacy{
		{Type: types.StatusPrivacyTypeWhitelist,
			List: []types.JID{audAda}},
		{Type: types.StatusPrivacyTypeBlacklist,
			List:      []types.JID{audBola},
			IsDefault: true},
	}
	v := statusAudienceVerdict(lists)
	if v["ok"] != true || v["mode"] != "except" {
		t.Fatalf("verdict = %v", v)
	}
	if v["list_known"] != true {
		t.Fatalf("list_known = %v", v["list_known"])
	}
	got := v["list"].([]string)
	if len(got) != 1 || got[0] != "2348022222222@s.whatsapp.net" {
		t.Fatalf("list = %v", got)
	}
}

func TestStatusAudienceVerdictNoListsIsHonestlyUnknown(t *testing.T) {
	v := statusAudienceVerdict(nil)
	if v["mode"] != "contacts" || v["list_known"] != false {
		t.Fatalf("verdict = %v", v)
	}
	if len(v["list"].([]string)) != 0 {
		t.Fatalf("an unknown list must not be invented: %v", v)
	}
}

func TestStatusAudienceShapeRule(t *testing.T) {
	if got := statusAudienceShapeError("everyone", nil); got == "" {
		t.Fatal("an unknown mode must be refused")
	}
	if got := statusAudienceShapeError("contacts",
		[]types.JID{audAda}); got == "" {
		t.Fatal("contacts with a list must be refused")
	}
	if got := statusAudienceShapeError("only", nil); got == "" {
		t.Fatal("only with no people must be refused")
	}
	if got := statusAudienceShapeError("except",
		[]types.JID{audAda}); got != "" {
		t.Fatalf("a valid set was refused: %q", got)
	}
	if got := statusAudienceShapeError("contacts", nil); got != "" {
		t.Fatalf("contacts with no list was refused: %q", got)
	}
}

func TestStatusPrivacyListNodeShape(t *testing.T) {
	node := statusPrivacyListNode("only",
		[]types.JID{audAda, audBola})
	if node.Tag != "list" {
		t.Fatalf("tag = %q", node.Tag)
	}
	if node.Attrs["type"] != "whitelist" ||
		node.Attrs["default"] != "true" {
		t.Fatalf("attrs = %v", node.Attrs)
	}
	users, ok := node.Content.([]waBinary.Node)
	if !ok || len(users) != 2 {
		t.Fatalf("content = %#v", node.Content)
	}
	if users[0].Tag != "user" || users[0].Attrs["jid"] != audAda {
		t.Fatalf("user node = %#v", users[0])
	}
	// contacts carries an empty (but present) people list
	empty := statusPrivacyListNode("contacts", nil)
	if empty.Attrs["type"] != "contacts" {
		t.Fatalf("contacts attrs = %v", empty.Attrs)
	}
	if kids, ok := empty.Content.([]waBinary.Node); !ok ||
		len(kids) != 0 {
		t.Fatalf("contacts content = %#v", empty.Content)
	}
}

func TestWriteStatusAudienceReportsTheReadBack(t *testing.T) {
	fake := &fakeStatusAudienceClient{
		lists: []types.StatusPrivacy{
			{Type: types.StatusPrivacyTypeContacts,
				IsDefault: true},
		},
		// The server applied a DIFFERENT list than asked
		// (a contact was dropped): the verb must report
		// the server's truth.
		afterWrite: []types.StatusPrivacy{
			{Type: types.StatusPrivacyTypeWhitelist,
				List:      []types.JID{audAda},
				IsDefault: true},
		},
	}
	v := writeStatusAudience(context.Background(), fake,
		"only", []types.JID{audAda, audBola})
	if v["ok"] != true || v["written"] != true {
		t.Fatalf("verdict = %v", v)
	}
	if v["mode"] != "only" {
		t.Fatalf("mode = %v", v["mode"])
	}
	got := v["list"].([]string)
	if len(got) != 1 ||
		got[0] != "2348011111111@s.whatsapp.net" {
		t.Fatalf("the read-back list must win: %v", got)
	}
	if fake.gotNode == nil {
		t.Fatal("the write never reached the client")
	}
}

func TestWriteStatusAudienceFailuresStayHonest(t *testing.T) {
	boom := errors.New("boom")
	fake := &fakeStatusAudienceClient{writeErr: boom}
	v := writeStatusAudience(context.Background(), fake,
		"except", []types.JID{audAda})
	if v["ok"] != false || v["error"] != "boom" {
		t.Fatalf("write failure = %v", v)
	}
	// A shape refusal never reaches the wire.
	fake2 := &fakeStatusAudienceClient{}
	v = writeStatusAudience(context.Background(), fake2,
		"only", nil)
	if v["ok"] != false || fake2.gotNode != nil {
		t.Fatalf("shape refusal = %v (node %v)", v,
			fake2.gotNode)
	}
	// A read failure on the plain read verb is a refusal.
	fake3 := &fakeStatusAudienceClient{readErr: boom}
	v = readStatusAudience(context.Background(), fake3)
	if v["ok"] != false || v["error"] != "boom" {
		t.Fatalf("read failure = %v", v)
	}
}

func TestStatusViewersMemory(t *testing.T) {
	b := &bridge{}
	b.noteStatusSent("post-1")
	b.noteStatusView("post-1",
		"2348011111111@s.whatsapp.net", 3000)
	b.noteStatusView("post-1",
		"2348022222222@s.whatsapp.net", 1000)
	// a repeat view keeps the FIRST time seen
	b.noteStatusView("post-1",
		"2348011111111@s.whatsapp.net", 9000)
	// empty shapes record nothing
	b.noteStatusView("", "2348011111111@s.whatsapp.net", 1)
	b.noteStatusView("post-1", "", 1)

	v := b.statusViewers("post-1")
	if v["ok"] != true || v["count"] != 2 {
		t.Fatalf("verdict = %v", v)
	}
	if v["posted_here"] != true {
		t.Fatalf("posted_here = %v", v["posted_here"])
	}
	rows := v["viewers"].([]map[string]any)
	// first-seen order: Bola (1000) before Ada (3000),
	// and Ada's repeat at 9000 did not move her
	if rows[0]["jid"] != "2348022222222@s.whatsapp.net" ||
		rows[1]["jid"] != "2348011111111@s.whatsapp.net" ||
		rows[1]["ts"] != int64(3000) {
		t.Fatalf("rows = %v", rows)
	}
	// a post this bridge neither sent nor saw views for
	other := b.statusViewers("post-unknown")
	if other["count"] != 0 || other["posted_here"] != false {
		t.Fatalf("unknown post = %v", other)
	}
}

func TestStatusViewersMemoryIsBounded(t *testing.T) {
	b := &bridge{}
	for i := 0; i < 130; i++ {
		b.noteStatusSent("post-" + strconv.Itoa(i))
	}
	// the sent memory holds at most the last 100 posts
	b.statusMu.Lock()
	n := len(b.statusOrder)
	b.statusMu.Unlock()
	if n != 100 {
		t.Fatalf("sent memory holds %d posts, want 100", n)
	}
}
