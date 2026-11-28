package main

// Bridge-level pins for the status verbs (status fix,
// 2026-11-07).  The truth pass proved the Python seams against
// a RECORDING FAKE bridge — a fake that accepts any well-shaped
// call, so the bridge binaries' own call shapes were never
// exercised by any test, and that unexercised layer is exactly
// where the mined divergences lived (status mining REPORT
// §1.3-1.5).  These tests pin the SHAPES: the receipt's
// destination/time/mode and the reaction's destination/key.

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"go.mau.fi/whatsmeow"
	waCommon "go.mau.fi/whatsmeow/proto/waCommon"
	waE2E "go.mau.fi/whatsmeow/proto/waE2E"
	"go.mau.fi/whatsmeow/types"
	"google.golang.org/protobuf/proto"
)

// ── status view ─────────────────────────────────────────────

type fakeStatusViewClient struct {
	settings     types.PrivacySettings
	markErr      error
	gotIDs       []types.MessageID
	gotTimestamp time.Time
	gotChat      types.JID
	gotSender    types.JID
}

func (f *fakeStatusViewClient) MarkRead(_ context.Context,
	ids []types.MessageID, timestamp time.Time, chat,
	sender types.JID, _ ...types.ReceiptType) error {
	f.gotIDs = ids
	f.gotTimestamp = timestamp
	f.gotChat = chat
	f.gotSender = sender
	return f.markErr
}

func (f *fakeStatusViewClient) GetPrivacySettings(
	context.Context) types.PrivacySettings {
	return f.settings
}

var pnPoster = types.JID{User: "2348012345678",
	Server: types.DefaultUserServer}
var lidPoster = types.JID{User: "123456789012345",
	Server: types.HiddenUserServer}

func TestMarkStatusViewedShapeReadTimeAndMode(t *testing.T) {
	cli := &fakeStatusViewClient{settings: types.PrivacySettings{
		ReadReceipts: types.PrivacySettingAll}}
	before := time.Now()
	mode, err := markStatusViewed(context.Background(), cli,
		pnPoster, types.MessageID("POST1"))
	after := time.Now()
	if err != nil {
		t.Fatalf("markStatusViewed: %v", err)
	}
	if mode != "read" {
		t.Fatalf("receipt mode = %q, want read", mode)
	}
	if cli.gotChat != types.StatusBroadcastJID {
		t.Fatalf("receipt chat = %v, want status@broadcast",
			cli.gotChat)
	}
	if cli.gotSender != pnPoster {
		t.Fatalf("receipt sender = %v, want the poster %v",
			cli.gotSender, pnPoster)
	}
	if len(cli.gotIDs) != 1 || cli.gotIDs[0] != "POST1" {
		t.Fatalf("receipt ids = %v, want [POST1]", cli.gotIDs)
	}
	// The receipt is stamped at READ time (the phone's shape),
	// never with the post's own timestamp.
	if cli.gotTimestamp.Before(before) ||
		cli.gotTimestamp.After(after) {
		t.Fatalf("receipt timestamp %v outside the read window "+
			"[%v, %v]", cli.gotTimestamp, before, after)
	}
}

func TestMarkStatusViewedReceiptModeFollowsPrivacy(t *testing.T) {
	// Read receipts OFF on the account: whatsmeow rewrites the
	// receipt to read-self (the poster is never notified) — the
	// mode must say so, because ok alone cannot.
	cli := &fakeStatusViewClient{settings: types.PrivacySettings{
		ReadReceipts: types.PrivacySettingNone}}
	mode, err := markStatusViewed(context.Background(), cli,
		lidPoster, types.MessageID("POST2"))
	if err != nil {
		t.Fatalf("markStatusViewed: %v", err)
	}
	if mode != "read-self" {
		t.Fatalf("receipt mode = %q, want read-self", mode)
	}
	// A setting that could not be read (empty struct) is
	// "unknown" — never guessed into either real mode.
	cli2 := &fakeStatusViewClient{}
	mode, err = markStatusViewed(context.Background(), cli2,
		pnPoster, types.MessageID("POST3"))
	if err != nil {
		t.Fatalf("markStatusViewed: %v", err)
	}
	if mode != "unknown" {
		t.Fatalf("receipt mode = %q, want unknown", mode)
	}
}

func TestMarkStatusViewedErrorPropagates(t *testing.T) {
	cli := &fakeStatusViewClient{markErr: errors.New("boom")}
	mode, err := markStatusViewed(context.Background(), cli,
		pnPoster, types.MessageID("POST4"))
	if err == nil || mode != "" {
		t.Fatalf("mode=%q err=%v, want the MarkRead error", mode, err)
	}
}

// ── status react ────────────────────────────────────────────

type fakeStatusReactClient struct {
	buildChat   types.JID
	buildSender types.JID
	buildID     types.MessageID
	sendTo      types.JID
	sendMsg     *waE2E.Message
	sendErr     error
}

func (f *fakeStatusReactClient) BuildReaction(chat,
	sender types.JID, id types.MessageID,
	reaction string) *waE2E.Message {
	f.buildChat = chat
	f.buildSender = sender
	f.buildID = id
	// The payload shape the real BuildReaction produces: a
	// ReactionMessage whose key is the status post's key.
	return &waE2E.Message{ReactionMessage: &waE2E.ReactionMessage{
		Key: &waCommon.MessageKey{
			RemoteJID:   proto.String(chat.String()),
			Participant: proto.String(sender.String()),
			ID:          proto.String(string(id)),
			FromMe:      proto.Bool(false),
		},
		Text: proto.String(reaction),
	}}
}

func (f *fakeStatusReactClient) SendMessage(_ context.Context,
	to types.JID, message *waE2E.Message,
	_ ...whatsmeow.SendRequestExtra) (whatsmeow.SendResponse,
	error) {
	f.sendTo = to
	f.sendMsg = message
	return whatsmeow.SendResponse{}, f.sendErr
}

func TestSendStatusReactionGoesToThePosterNotTheBroadcast(t *testing.T) {
	for _, poster := range []types.JID{pnPoster, lidPoster} {
		cli := &fakeStatusReactClient{}
		err := sendStatusReaction(context.Background(), cli,
			poster, types.MessageID("POST9"), "❤️")
		if err != nil {
			t.Fatalf("sendStatusReaction(%v): %v", poster, err)
		}
		// The KEY is built over the status post…
		if cli.buildChat != types.StatusBroadcastJID {
			t.Fatalf("reaction key chat = %v, want "+
				"status@broadcast", cli.buildChat)
		}
		if cli.buildSender != poster || cli.buildID != "POST9" {
			t.Fatalf("reaction key = (%v, %v), want (%v, POST9)",
				cli.buildSender, cli.buildID, poster)
		}
		// …but the SEND destination is the POSTER, 1:1.  A send
		// addressed to status@broadcast fans out to the
		// account's own status audience — the mined death of
		// this verb on the Go engine.
		if cli.sendTo != poster {
			t.Fatalf("reaction destination = %v, want the "+
				"poster %v", cli.sendTo, poster)
		}
		if cli.sendTo == types.StatusBroadcastJID {
			t.Fatal("reaction was addressed to status@broadcast")
		}
		key := cli.sendMsg.GetReactionMessage().GetKey()
		if key.GetRemoteJID() != "status@broadcast" ||
			key.GetParticipant() != poster.String() ||
			key.GetID() != "POST9" {
			t.Fatalf("sent key = %v, want the status post's key",
				key)
		}
		if cli.sendMsg.GetReactionMessage().GetText() != "❤️" {
			t.Fatal("sent reaction lost its emoji")
		}
	}
}

func TestSendStatusReactionErrorPropagates(t *testing.T) {
	cli := &fakeStatusReactClient{sendErr: errors.New("boom")}
	if err := sendStatusReaction(context.Background(), cli,
		pnPoster, types.MessageID("POST10"), "❤️"); err == nil {
		t.Fatal("the send error vanished")
	}
}

// ── gapfix166: the poster's alternate form + supplied keys ──

func TestPosterJIDPrefersFromAlt(t *testing.T) {
	// Both forms sent: the receipt/react must carry the
	// alternate form WhatsApp keyed the status under.
	jid, err := request{"from": lidPoster.String(),
		"from_alt": pnPoster.String()}.posterJID()
	if err != nil || jid != pnPoster {
		t.Fatalf("posterJID = %v, %v, want the alt %v",
			jid, err, pnPoster)
	}
	// No alt: the primary form, exactly as before.
	jid, err = request{"from": lidPoster.String()}.posterJID()
	if err != nil || jid != lidPoster {
		t.Fatalf("posterJID = %v, %v, want %v", jid, err,
			lidPoster)
	}
	// A bad alt refuses instead of falling back to the wrong
	// form silently.  ("@" parses to an empty JID —
	// whatsmeow's ParseJID never errors, it returns empty
	// for garbage — so emptiness is the refusal signal.)
	if _, err := (request{"from": lidPoster.String(),
		"from_alt": "@"}).posterJID(); err == nil {
		t.Fatal("an alt that parses to an empty JID was accepted")
	}
}

func TestSuppliedReactKeyShapes(t *testing.T) {
	self := types.JID{User: "2348000000000",
		Server: types.DefaultUserServer}
	group := types.JID{User: "120363025000000000",
		Server: types.GroupServer}
	// Group key: the participant is the sender.
	chat, sender, id, ok := suppliedReactKey(map[string]any{
		"remoteJid": group.String(), "id": "m1",
		"participant": pnPoster.String(), "fromMe": false,
	}, self)
	if !ok || chat != group || sender != pnPoster || id != "m1" {
		t.Fatalf("group key = (%v, %v, %v, %v), want (%v, %v, m1, true)",
			chat, sender, id, ok, group, pnPoster)
	}
	// DM key from the other party: no participant, so the
	// sender is the chat itself.
	chat, sender, id, ok = suppliedReactKey(map[string]any{
		"remoteJid": pnPoster.String(), "id": "m2",
		"fromMe": false,
	}, self)
	if !ok || chat != pnPoster || sender != pnPoster || id != "m2" {
		t.Fatalf("DM key = (%v, %v, %v, %v), want (%v, %v, m2, true)",
			chat, sender, id, ok, pnPoster, pnPoster)
	}
	// The account's own DM message: no participant, fromMe —
	// the sender is the account, or the reaction key would
	// claim the other party sent it.
	chat, sender, id, ok = suppliedReactKey(map[string]any{
		"remoteJid": pnPoster.String(), "id": "m3",
		"fromMe": true,
	}, self)
	if !ok || sender != self {
		t.Fatalf("own DM key sender = %v, want self %v",
			sender, self)
	}
	// Malformed keys refuse: no remoteJid, no id, a remoteJid
	// that parses to empty, a bad participant — never a
	// half-built reaction.
	for _, key := range []map[string]any{
		{"id": "m4"},
		{"remoteJid": pnPoster.String()},
		{"remoteJid": "@", "id": "m4"},
		{"remoteJid": group.String(), "id": "m4",
			"participant": "@"},
	} {
		if _, _, _, ok := suppliedReactKey(key, self); ok {
			t.Fatalf("malformed key %v was accepted", key)
		}
	}
}

// ── authored keyboards (buttons completion, 2026-11-08) ─────

func TestParseButtonsFlattensRowsInOrder(t *testing.T) {
	req := request{"buttons": []any{
		[]any{[]any{"A", "a"}, []any{"B", "b"}},
		[]any{[]any{"C", "c"}},
		[]any{"junk"},
	}}
	flat := parseButtons(req)
	if len(flat) != 3 || flat[0].cmd != "a" || flat[1].cmd != "b" ||
		flat[2].cmd != "c" {
		t.Fatalf("parseButtons = %+v, want a,b,c in order", flat)
	}
	if got := parseButtons(request{}); len(got) != 0 {
		t.Fatalf("parseButtons(no buttons) = %+v, want empty", got)
	}
}

func TestSplitNativeCapsAtThreeAndLinesTheOverflow(t *testing.T) {
	flat := []waButton{{"1", "a"}, {"2", "b"}, {"3", "c"}, {"4", "d"}}
	native, body := splitNative(flat, "pick")
	if len(native) != 3 {
		t.Fatalf("native = %d, want 3", len(native))
	}
	if body != "pick\n• 4 — /d" {
		t.Fatalf("body = %q, want the overflow as a text line", body)
	}
	native, body = splitNative(flat[:2], "pick")
	if len(native) != 2 || body != "pick" {
		t.Fatalf("small keyboard changed: %v %q", native, body)
	}
}

func TestInteractiveMessageShape(t *testing.T) {
	long := "abcdefghijklmnopqrstuvwxyz0123456789" // 36 runes
	msg := interactiveMessage("body", []waButton{
		{long, "cmd1"}, {"B", "cmd2"}}, nil, nil)
	im := msg.GetInteractiveMessage()
	if im == nil || im.GetBody().GetText() != "body" {
		t.Fatalf("interactive body wrong: %+v", im)
	}
	nf := im.GetNativeFlowMessage()
	if nf == nil || nf.GetMessageVersion() != 3 ||
		len(nf.GetButtons()) != 2 {
		t.Fatalf("native flow wrong: %+v", nf)
	}
	first := nf.GetButtons()[0]
	if first.GetName() != "quick_reply" {
		t.Fatalf("button name = %q, want quick_reply", first.GetName())
	}
	var params map[string]string
	if err := json.Unmarshal([]byte(first.GetButtonParamsJSON()),
		&params); err != nil {
		t.Fatalf("button params not JSON: %v", err)
	}
	if params["id"] != "cmd1" ||
		len([]rune(params["display_text"])) != 25 {
		t.Fatalf("params = %+v, want id cmd1 + label cut at 25 runes",
			params)
	}
}

func TestInteractiveMessageMediaHeader(t *testing.T) {
	img := &waE2E.ImageMessage{URL: proto.String("https://x")}
	header := &waE2E.InteractiveMessage_Header{
		HasMediaAttachment: proto.Bool(true),
		Media: &waE2E.InteractiveMessage_Header_ImageMessage{
			ImageMessage: img}}
	msg := interactiveMessage("cap", []waButton{{"A", "a"}},
		nil, header)
	h := msg.GetInteractiveMessage().GetHeader()
	if h == nil || !h.GetHasMediaAttachment() ||
		h.GetImageMessage() != img {
		t.Fatalf("media header wrong: %+v", h)
	}
}

func TestBuildListMessageShape(t *testing.T) {
	var flat []waButton
	for i := 0; i < 12; i++ {
		flat = append(flat, waButton{"L", "cmd"})
	}
	msg := buildListMessage("pick one", flat, nil)
	lm := msg.GetListMessage()
	if lm == nil || lm.GetListType() !=
		waE2E.ListMessage_SINGLE_SELECT {
		t.Fatalf("list type wrong: %+v", lm)
	}
	if lm.GetDescription() != "pick one" ||
		lm.GetButtonText() != "Choose" {
		t.Fatalf("list body wrong: %q / %q",
			lm.GetDescription(), lm.GetButtonText())
	}
	if len(lm.GetSections()) != 2 ||
		len(lm.GetSections()[0].GetRows()) != 10 ||
		len(lm.GetSections()[1].GetRows()) != 2 {
		t.Fatalf("sections = %+v, want 10 + 2 rows", lm.GetSections())
	}
	if lm.GetSections()[0].GetRows()[0].GetRowID() != "cmd" {
		t.Fatalf("row id must carry the command")
	}
}

func TestButtonsFormName(t *testing.T) {
	flat := []waButton{{"A", "a"}}
	if buttonsFormName("", flat) != "interactive" ||
		buttonsFormName("list", flat) != "list" ||
		buttonsFormName("list", nil) != "" {
		t.Fatal("form names wrong")
	}
}
