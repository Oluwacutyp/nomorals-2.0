package main

// Bridge-level pins for the identity + poll verbs (parity G3,
// census bundles #5/#6).  Same standard as the status-fix
// tests: the Python side's recording fakes accept any shape,
// so the shapes that fail SILENTLY live — a privacy value
// outside its category vocabulary, a poll built with the
// wrong selectable count — are pinned HERE, against fakes
// that record exactly what the whatsmeow client was asked.

import (
	"context"
	"crypto/sha256"
	"errors"
	"strings"
	"testing"

	"go.mau.fi/whatsmeow"
	"go.mau.fi/whatsmeow/appstate"
	waE2E "go.mau.fi/whatsmeow/proto/waE2E"
	"go.mau.fi/whatsmeow/types"
	"google.golang.org/protobuf/proto"
)

type fakeProfileClient struct {
	patches    []appstate.PatchInfo
	statusText *string
	photoJID   types.JID
	photoBytes []byte
	photoSet   bool
	info       map[types.JID]types.UserInfo
	settings   types.PrivacySettings
	setName    types.PrivacySettingType
	setValue   types.PrivacySetting
	setErr     error
}

func (f *fakeProfileClient) SendAppState(_ context.Context,
	patch appstate.PatchInfo) error {
	f.patches = append(f.patches, patch)
	return nil
}

func (f *fakeProfileClient) SetStatusMessage(_ context.Context,
	status types.SetStatusInput) error {
	f.statusText = status.Text
	return nil
}

func (f *fakeProfileClient) SetGroupPhoto(_ context.Context,
	jid types.JID, avatar []byte) (string, error) {
	f.photoJID = jid
	f.photoBytes = avatar
	f.photoSet = true
	if avatar == nil {
		return "remove", nil
	}
	return "pic-1", nil
}

func (f *fakeProfileClient) GetUserInfo(_ context.Context,
	_ []types.JID) (map[types.JID]types.UserInfo, error) {
	return f.info, nil
}

func (f *fakeProfileClient) TryFetchPrivacySettings(
	_ context.Context, _ bool) (*types.PrivacySettings, error) {
	s := f.settings
	return &s, nil
}

func (f *fakeProfileClient) SetPrivacySetting(_ context.Context,
	name types.PrivacySettingType, value types.PrivacySetting,
) (types.PrivacySettings, error) {
	f.setName = name
	f.setValue = value
	return f.settings, f.setErr
}

func TestApplyProfileNamePatchThenStore(t *testing.T) {
	cli := &fakeProfileClient{}
	stored := ""
	err := applyProfileName(context.Background(), cli,
		"Devon Owner", func(n string) { stored = n })
	if err != nil {
		t.Fatalf("applyProfileName: %v", err)
	}
	if len(cli.patches) != 1 {
		t.Fatalf("patches = %d, want exactly 1 app-state patch",
			len(cli.patches))
	}
	if stored != "Devon Owner" {
		t.Fatalf("store = %q, want the new name after the patch",
			stored)
	}
}

func TestSetPrivacyValidatesBeforeTraffic(t *testing.T) {
	cli := &fakeProfileClient{}
	// readreceipts only takes all|none — "contacts" must
	// refuse in the bridge, before any IQ is sent.
	if _, err := setPrivacy(context.Background(), cli,
		"readreceipts", "contacts"); err == nil {
		t.Fatal("readreceipts=contacts must refuse")
	}
	// online only takes all|match_last_seen.
	if _, err := setPrivacy(context.Background(), cli,
		"online", "none"); err == nil {
		t.Fatal("online=none must refuse")
	}
	// an unknown category refuses too.
	if _, err := setPrivacy(context.Background(), cli,
		"nonsense", "all"); err == nil {
		t.Fatal("unknown category must refuse")
	}
	if cli.setName != "" {
		t.Fatalf("a refused set reached the client: %v",
			cli.setName)
	}
}

func TestSetPrivacyMapsCategoryAndValue(t *testing.T) {
	cli := &fakeProfileClient{settings: types.PrivacySettings{
		LastSeen: types.PrivacySettingContacts}}
	out, err := setPrivacy(context.Background(), cli,
		"lastseen", "contacts")
	if err != nil {
		t.Fatalf("setPrivacy: %v", err)
	}
	if cli.setName != types.PrivacySettingTypeLastSeen {
		t.Fatalf("category mapped to %q, want %q",
			cli.setName, types.PrivacySettingTypeLastSeen)
	}
	if cli.setValue != types.PrivacySettingContacts {
		t.Fatalf("value = %q, want contacts", cli.setValue)
	}
	if out["lastseen"] != "contacts" {
		t.Fatalf("reply settings = %v, want lastseen=contacts",
			out)
	}
}

func TestSetPrivacyErrorPropagates(t *testing.T) {
	cli := &fakeProfileClient{setErr: errors.New("iq refused")}
	if _, err := setPrivacy(context.Background(), cli,
		"status", "all"); err == nil {
		t.Fatal("a client refusal must surface, not succeed")
	}
}

func TestPrivacySettingsMapOmitsUnread(t *testing.T) {
	out := privacySettingsMap(types.PrivacySettings{
		GroupAdd: types.PrivacySettingAll,
		Online:   types.PrivacySettingMatchLastSeen})
	if out["groupadd"] != "all" || out["online"] !=
		"match_last_seen" {
		t.Fatalf("map = %v", out)
	}
	// messages/defense/stickers have no struct field: absent,
	// never a fabricated "".
	for _, k := range []string{"messages", "defense",
		"stickers", "status"} {
		if _, present := out[k]; present {
			t.Fatalf("unread category %q present in %v",
				k, out)
		}
	}
}

// ── polls ────────────────────────────────────────────────────

type fakePollClient struct {
	builtName       string
	builtOptions    []string
	builtSelectable int
	sentTo          types.JID
	sentMsg         *waE2E.Message
	sendErr         error
}

func (f *fakePollClient) BuildPollCreation(name string,
	optionNames []string, selectableOptionCount int,
) *waE2E.Message {
	f.builtName = name
	f.builtOptions = optionNames
	f.builtSelectable = selectableOptionCount
	options := make([]*waE2E.PollCreationMessage_Option,
		len(optionNames))
	for i, o := range optionNames {
		options[i] = &waE2E.PollCreationMessage_Option{
			OptionName: proto.String(o)}
	}
	return &waE2E.Message{PollCreationMessage: &waE2E.PollCreationMessage{
		Name: proto.String(name), Options: options,
		SelectableOptionsCount: proto.Uint32(
			uint32(selectableOptionCount))}}
}

func (f *fakePollClient) SendMessage(_ context.Context,
	to types.JID, message *waE2E.Message,
	_ ...whatsmeow.SendRequestExtra,
) (whatsmeow.SendResponse, error) {
	f.sentTo = to
	f.sentMsg = message
	if f.sendErr != nil {
		return whatsmeow.SendResponse{}, f.sendErr
	}
	return whatsmeow.SendResponse{ID: "POLL-1"}, nil
}

func TestSendPollShape(t *testing.T) {
	cli := &fakePollClient{}
	chat := types.JID{User: "120363001234567890",
		Server: types.GroupServer}
	id, built, err := sendPoll(context.Background(), cli, chat,
		"Lunch?", []string{"Pizza", "Sushi"}, 1)
	if err != nil {
		t.Fatalf("sendPoll: %v", err)
	}
	if id != "POLL-1" {
		t.Fatalf("id = %q, want the send response's id", id)
	}
	if cli.sentTo != chat {
		t.Fatalf("poll sent to %v, want %v", cli.sentTo, chat)
	}
	if cli.builtName != "Lunch?" ||
		len(cli.builtOptions) != 2 ||
		cli.builtSelectable != 1 {
		t.Fatalf("built with name=%q options=%v selectable=%d",
			cli.builtName, cli.builtOptions,
			cli.builtSelectable)
	}
	if built.GetPollCreationMessage() == nil {
		t.Fatal("the sent message carries no poll creation")
	}
}

func TestPollShapeRules(t *testing.T) {
	good := []string{"a", "b"}
	if msg := pollShapeError("Q?", good, 1); msg != "" {
		t.Fatalf("a valid poll refused: %s", msg)
	}
	if msg := pollShapeError("", good, 1); msg == "" {
		t.Fatal("an empty question must refuse")
	}
	if msg := pollShapeError("Q?", []string{"only"}, 1); msg ==
		"" {
		t.Fatal("one option must refuse (2–12)")
	}
	many := make([]string, 13)
	for i := range many {
		many[i] = "x"
	}
	if msg := pollShapeError("Q?", many, 1); msg == "" {
		t.Fatal("thirteen options must refuse (2–12)")
	}
	if msg := pollShapeError("Q?", good, 3); msg == "" {
		t.Fatal("selectable beyond the option count refuses")
	}
	if msg := pollShapeError(strings.Repeat("q", 256), good,
		1); msg == "" {
		t.Fatal("a 256-character question must refuse")
	}
}

func TestPollTallyCountsAndRevotes(t *testing.T) {
	b := &bridge{}
	chat := types.JID{User: "120363001234567890",
		Server: types.GroupServer}
	pc := &waE2E.PollCreationMessage{
		Name: proto.String("Lunch?"),
		Options: []*waE2E.PollCreationMessage_Option{
			{OptionName: proto.String("Pizza")},
			{OptionName: proto.String("Sushi")},
		}}
	b.notePollCreation(chat, "POLL-1", pc, 1000)
	hPizza := sha256.Sum256([]byte("Pizza"))
	hSushi := sha256.Sum256([]byte("Sushi"))
	if !b.recordPollVote(chat, "POLL-1", "voter-a",
		[][]byte{hPizza[:]}) {
		t.Fatal("a vote on a remembered poll must land")
	}
	b.recordPollVote(chat, "POLL-1", "voter-b",
		[][]byte{hPizza[:]})
	// voter-a CHANGES their vote: one voter, one tally row —
	// the replacement is the phone's own semantics.
	b.recordPollVote(chat, "POLL-1", "voter-a",
		[][]byte{hSushi[:]})
	out, ok := b.pollResults(chat, "POLL-1")
	if !ok {
		t.Fatal("results for a remembered poll must exist")
	}
	if out["total_voters"] != 2 {
		t.Fatalf("total_voters = %v, want 2", out["total_voters"])
	}
	opts := out["options"].([]map[string]any)
	if opts[0]["voters"] != 1 || opts[1]["voters"] != 1 {
		t.Fatalf("tally = %v, want Pizza 1 / Sushi 1 after "+
			"the re-vote", opts)
	}
	if b.recordPollVote(chat, "POLL-UNKNOWN", "voter-a",
		[][]byte{hPizza[:]}) {
		t.Fatal("a vote on an unseen poll cannot be tallied")
	}
	if _, ok := b.pollResults(chat, "POLL-UNKNOWN"); ok {
		t.Fatal("results for an unseen poll must not exist")
	}
}
