package main

// Poll-results truth pass (parity #16, census bundle #16).
// Bundle #6 built poll SENDING and the bridge-memory tally,
// but the read path shipped UNPROVEN: the Python suite's fake
// bridge re-implemented the tally by option NAME, so the real
// machinery — creations teaching option names, votes landing
// as option HASHES (whatsmeow's own identity), re-votes
// replacing, retractions dropping the voter — was never
// exercised anywhere.  These tests drive the real bridge
// state functions and the real cmdPollResults wire handler
// with protocol-true shapes: creations as waE2E messages,
// votes as whatsmeow.HashPollOptions output (exactly what
// DecryptPollVote hands recordPollVote).

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/json"
	"net"
	"testing"

	"go.mau.fi/whatsmeow"
	waE2E "go.mau.fi/whatsmeow/proto/waE2E"
	"go.mau.fi/whatsmeow/types"
	"google.golang.org/protobuf/proto"
)

var pollChat = types.JID{User: "120363001234567890",
	Server: types.GroupServer}

const pollChatStr = "120363001234567890@g.us"

func pollCreationMsg(question string, options ...string,
) *waE2E.PollCreationMessage {
	opts := make([]*waE2E.PollCreationMessage_Option,
		len(options))
	for i, o := range options {
		opts[i] = &waE2E.PollCreationMessage_Option{
			OptionName: proto.String(o)}
	}
	return &waE2E.PollCreationMessage{
		Name: proto.String(question), Options: opts,
		SelectableOptionsCount: proto.Uint32(1)}
}

// selection returns the vote payload for the named options,
// hashed the way the protocol itself hashes them.
func selection(names ...string) [][]byte {
	return whatsmeow.HashPollOptions(names)
}

func notedBridge() *bridge {
	b := &bridge{}
	b.notePollCreation(pollChat, "POLL-1",
		pollCreationMsg("Dinner?", "Pizza", "Sushi"), 1000)
	return b
}

func tallyOf(t *testing.T, b *bridge, chat types.JID,
	id string) (map[string]int, int, map[string]any) {
	t.Helper()
	out, ok := b.pollResults(chat, id)
	if !ok {
		t.Fatalf("pollResults(%s) not in memory", id)
	}
	counts := map[string]int{}
	for _, row := range out["options"].([]map[string]any) {
		counts[row["text"].(string)] = row["voters"].(int)
	}
	return counts, out["total_voters"].(int), out
}

func TestPollResultsTallyRealShapes(t *testing.T) {
	b := notedBridge()
	b.recordPollVote(pollChat, "POLL-1",
		"2348011111111@s.whatsapp.net", selection("Pizza"))
	b.recordPollVote(pollChat, "POLL-1",
		"2348022222222@s.whatsapp.net", selection("Pizza"))
	b.recordPollVote(pollChat, "POLL-1",
		"2348033333333@s.whatsapp.net", selection("Sushi"))
	counts, total, out := tallyOf(t, b, pollChat, "POLL-1")
	if counts["Pizza"] != 2 || counts["Sushi"] != 1 {
		t.Fatalf("counts = %v, want Pizza 2 / Sushi 1", counts)
	}
	if total != 3 {
		t.Fatalf("total_voters = %d, want 3", total)
	}
	if out["question"] != "Dinner?" {
		t.Fatalf("question = %v", out["question"])
	}
	if _, rides := out["unreadable_votes"]; rides {
		t.Fatal("unreadable_votes must stay off the verdict " +
			"when every vote decrypted")
	}
}

func TestPollResultsRevoteReplaces(t *testing.T) {
	b := notedBridge()
	voter := "2348033333333@s.whatsapp.net"
	b.recordPollVote(pollChat, "POLL-1", voter,
		selection("Sushi"))
	b.recordPollVote(pollChat, "POLL-1", voter,
		selection("Pizza")) // changed mind — replaces, not adds
	counts, total, _ := tallyOf(t, b, pollChat, "POLL-1")
	if counts["Pizza"] != 1 || counts["Sushi"] != 0 {
		t.Fatalf("counts = %v, want Pizza 1 / Sushi 0", counts)
	}
	if total != 1 {
		t.Fatalf("total_voters = %d, want 1", total)
	}
}

func TestPollResultsRetractionDropsVoter(t *testing.T) {
	b := notedBridge()
	voter := "2348033333333@s.whatsapp.net"
	b.recordPollVote(pollChat, "POLL-1", voter,
		selection("Sushi"))
	// An empty selection is a retraction: the phone's tally
	// drops the voter entirely (parity #16 fix — both bridges
	// used to keep the voter in total_voters with no vote).
	if !b.recordPollVote(pollChat, "POLL-1", voter, nil) {
		t.Fatal("a retraction on a known poll must land")
	}
	counts, total, _ := tallyOf(t, b, pollChat, "POLL-1")
	if counts["Sushi"] != 0 || total != 0 {
		t.Fatalf("counts = %v total = %d, want 0 / 0",
			counts, total)
	}
	// The poll itself is still KNOWN — an all-retracted poll
	// reads as zero votes, never as the not-in-memory refusal.
	if _, ok := b.pollResults(pollChat, "POLL-1"); !ok {
		t.Fatal("an all-retracted poll must stay readable")
	}
}

func TestPollResultsMultiSelect(t *testing.T) {
	b := notedBridge()
	b.recordPollVote(pollChat, "POLL-1",
		"2348011111111@s.whatsapp.net",
		selection("Pizza", "Sushi"))
	counts, total, _ := tallyOf(t, b, pollChat, "POLL-1")
	if counts["Pizza"] != 1 || counts["Sushi"] != 1 {
		t.Fatalf("counts = %v, want 1 / 1", counts)
	}
	if total != 1 {
		t.Fatalf("total_voters = %d, want 1 distinct voter",
			total)
	}
}

func TestPollResultsUnknownHashNotCounted(t *testing.T) {
	b := notedBridge()
	sum := sha256.Sum256([]byte("NotAnOption"))
	b.recordPollVote(pollChat, "POLL-1",
		"2348011111111@s.whatsapp.net", [][]byte{sum[:]})
	counts, _, _ := tallyOf(t, b, pollChat, "POLL-1")
	if counts["Pizza"] != 0 || counts["Sushi"] != 0 {
		t.Fatalf("an unmapped hash counted: %v", counts)
	}
}

func TestPollResultsUnreadableSurfaced(t *testing.T) {
	b := notedBridge()
	b.notePollVoteUnreadable(pollChat, "POLL-1")
	b.notePollVoteUnreadable(pollChat, "POLL-1")
	_, _, out := tallyOf(t, b, pollChat, "POLL-1")
	if out["unreadable_votes"] != 2 {
		t.Fatalf("unreadable_votes = %v, want 2",
			out["unreadable_votes"])
	}
	// Unreadable votes on a poll the bridge never saw have
	// nowhere to land and must not conjure the poll.
	b.notePollVoteUnreadable(pollChat, "POLL-GHOST")
	if _, ok := b.pollResults(pollChat, "POLL-GHOST"); ok {
		t.Fatal("an unreadable vote conjured a ghost poll")
	}
	if b.recordPollVote(pollChat, "POLL-GHOST",
		"2348011111111@s.whatsapp.net", selection("Pizza")) {
		t.Fatal("a vote on an unknown poll reported tallied")
	}
}

func TestPollResultsKeyedByChatAndID(t *testing.T) {
	b := notedBridge()
	other := types.JID{User: "120363009999999999",
		Server: types.GroupServer}
	if _, ok := b.pollResults(other, "POLL-1"); ok {
		t.Fatal("the same id in another chat must not resolve")
	}
	if _, ok := b.pollResults(pollChat, "POLL-OTHER"); ok {
		t.Fatal("another id in the same chat must not resolve")
	}
}

func TestPollCreationV3Unwrap(t *testing.T) {
	pc := pollCreationMsg("V3 poll?", "A", "B")
	msg := &waE2E.Message{PollCreationMessageV3: pc}
	if pollCreationOf(msg) != pc {
		t.Fatal("the V3 creation form must unwrap to the " +
			"common shape notePollEvent feeds memory")
	}
	b := &bridge{}
	b.notePollCreation(pollChat, "POLL-V3",
		pollCreationOf(msg), 1000)
	b.recordPollVote(pollChat, "POLL-V3",
		"2348011111111@s.whatsapp.net", selection("B"))
	counts, _, _ := tallyOf(t, b, pollChat, "POLL-V3")
	if counts["B"] != 1 {
		t.Fatalf("counts = %v, want B 1", counts)
	}
}

// cmdPollResultsOverPipe runs the real wire handler once and
// returns its decoded reply line.
func cmdPollResultsOverPipe(t *testing.T, b *bridge,
	req request) map[string]any {
	t.Helper()
	server, client := net.Pipe()
	defer server.Close()
	defer client.Close()
	cc := &clientConn{conn: server}
	done := make(chan struct{})
	go func() {
		defer close(done)
		b.cmdPollResults(context.Background(), cc, "req-1",
			req)
	}()
	line, err := bufio.NewReader(client).ReadBytes('\n')
	if err != nil {
		t.Fatalf("reading the reply line: %v", err)
	}
	<-done
	var out map[string]any
	if err := json.Unmarshal(line, &out); err != nil {
		t.Fatalf("the reply is not one JSON line: %v", err)
	}
	return out
}

func TestCmdPollResultsWireShapes(t *testing.T) {
	b := notedBridge()
	b.recordPollVote(pollChat, "POLL-1",
		"2348011111111@s.whatsapp.net", selection("Pizza"))
	b.notePollVoteUnreadable(pollChat, "POLL-1")

	out := cmdPollResultsOverPipe(t, b, request{
		"chat": pollChatStr, "message_id": "POLL-1"})
	if out["ok"] != true || out["id"] != "req-1" {
		t.Fatalf("reply envelope = %v", out)
	}
	if out["question"] != "Dinner?" ||
		out["total_voters"] != float64(1) {
		t.Fatalf("reply body = %v", out)
	}
	opts, isList := out["options"].([]any)
	if !isList || len(opts) != 2 {
		t.Fatalf("options = %v", out["options"])
	}
	first := opts[0].(map[string]any)
	if first["text"] != "Pizza" ||
		first["voters"] != float64(1) {
		t.Fatalf("first option row = %v", first)
	}
	if out["unreadable_votes"] != float64(1) {
		t.Fatalf("unreadable_votes = %v, want 1",
			out["unreadable_votes"])
	}

	out = cmdPollResultsOverPipe(t, b, request{
		"chat": pollChatStr, "message_id": "POLL-NEVER"})
	if out["ok"] != false {
		t.Fatalf("an unknown poll claimed ok: %v", out)
	}
	want := "that poll is not in bridge memory — results " +
		"live on polls this bridge has seen since it started"
	if out["error"] != want {
		t.Fatalf("refusal = %v, want %q", out["error"], want)
	}

	out = cmdPollResultsOverPipe(t, b, request{
		"chat": pollChatStr})
	if out["ok"] != false ||
		out["error"] != "poll_results needs a message_id" {
		t.Fatalf("missing-id reply = %v", out)
	}
}

func TestSendThenResultsRoundTrip(t *testing.T) {
	// The G3 send path, exactly as cmdSendPoll drives it:
	// sendPoll builds + sends, the built creation teaches
	// memory, and cmdPollResults reads the tally back.
	cli := &fakePollClient{}
	id, built, err := sendPoll(context.Background(), cli,
		pollChat, "Dinner?", []string{"Pizza", "Sushi"}, 1)
	if err != nil {
		t.Fatalf("sendPoll: %v", err)
	}
	b := &bridge{}
	if pc := pollCreationOf(built); pc != nil {
		b.notePollCreation(pollChat, string(id), pc, 1000)
	}
	b.recordPollVote(pollChat, string(id),
		"2348022222222@s.whatsapp.net", selection("Sushi"))
	out := cmdPollResultsOverPipe(t, b, request{
		"chat": pollChatStr, "message_id": string(id)})
	if out["ok"] != true || out["total_voters"] != float64(1) {
		t.Fatalf("round-trip reply = %v", out)
	}
	opts := out["options"].([]any)
	second := opts[1].(map[string]any)
	if second["text"] != "Sushi" ||
		second["voters"] != float64(1) {
		t.Fatalf("second option row = %v", second)
	}
}
