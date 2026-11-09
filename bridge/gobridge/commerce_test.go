package main

// Bridge-level pins for the commerce verbs (parity bundle
// #17).  Same standard as the identity/poll pins: the
// Python side's recording fakes accept any shape, so the
// shapes that fail SILENTLY live — a label patch addressed
// at the wrong app-state index, a color outside WhatsApp's
// twenty — are pinned HERE, against a fake that records
// exactly what the whatsmeow client was asked to send.

import (
	"context"
	"strings"
	"testing"

	"go.mau.fi/whatsmeow/appstate"
	waSyncAction "go.mau.fi/whatsmeow/proto/waSyncAction"
	"go.mau.fi/whatsmeow/types"
	"go.mau.fi/whatsmeow/types/events"
)

type fakeCommerceClient struct {
	patches []appstate.PatchInfo
	err     error
}

func (f *fakeCommerceClient) SendAppState(_ context.Context,
	patch appstate.PatchInfo) error {
	f.patches = append(f.patches, patch)
	return f.err
}

func TestApplyLabelEditPatchShape(t *testing.T) {
	cli := &fakeCommerceClient{}
	err := applyLabelEdit(context.Background(), cli,
		"1717000000000", "School fees", 4, false)
	if err != nil {
		t.Fatalf("applyLabelEdit: %v", err)
	}
	if len(cli.patches) != 1 {
		t.Fatalf("patches = %d, want 1", len(cli.patches))
	}
	p := cli.patches[0]
	if p.Type != appstate.WAPatchRegular {
		t.Fatalf("patch type = %v, want WAPatchRegular",
			p.Type)
	}
	if len(p.Mutations) != 1 {
		t.Fatalf("mutations = %d, want 1", len(p.Mutations))
	}
	m := p.Mutations[0]
	if len(m.Index) != 2 || m.Index[0] != "label_edit" ||
		m.Index[1] != "1717000000000" {
		t.Fatalf("index = %v, want [label_edit <id>]",
			m.Index)
	}
	act := m.Value.GetLabelEditAction()
	if act == nil {
		t.Fatal("mutation carries no LabelEditAction")
	}
	if act.GetName() != "School fees" ||
		act.GetColor() != 4 || act.GetDeleted() {
		t.Fatalf("action = %+v, want name/color kept, "+
			"not deleted", act)
	}
}

func TestApplyLabelDeleteIsATombstone(t *testing.T) {
	cli := &fakeCommerceClient{}
	if err := applyLabelEdit(context.Background(), cli,
		"42", "", 0, true); err != nil {
		t.Fatalf("applyLabelEdit delete: %v", err)
	}
	act := cli.patches[0].Mutations[0].Value.
		GetLabelEditAction()
	if !act.GetDeleted() {
		t.Fatal("delete must ride the tombstone flag")
	}
}

func TestApplyLabelChatPatchShape(t *testing.T) {
	cli := &fakeCommerceClient{}
	chat := types.JID{User: "2348012345678",
		Server: types.DefaultUserServer}
	if err := applyLabelChat(context.Background(), cli,
		chat, "42", true); err != nil {
		t.Fatalf("applyLabelChat: %v", err)
	}
	m := cli.patches[0].Mutations[0]
	want := []string{"label_jid", "42", chat.String()}
	if strings.Join(m.Index, " ") !=
		strings.Join(want, " ") {
		t.Fatalf("index = %v, want %v", m.Index, want)
	}
	if !m.Value.GetLabelAssociationAction().GetLabeled() {
		t.Fatal("association must carry labeled=true")
	}
}

func TestApplyLabelMessagePatchShape(t *testing.T) {
	cli := &fakeCommerceClient{}
	chat := types.JID{User: "120363001234567890",
		Server: types.GroupServer}
	if err := applyLabelMessage(context.Background(), cli,
		chat, "42", "MSGID1", false); err != nil {
		t.Fatalf("applyLabelMessage: %v", err)
	}
	m := cli.patches[0].Mutations[0]
	want := []string{"label_message", "42", chat.String(),
		"MSGID1", "0", "0"}
	if strings.Join(m.Index, " ") !=
		strings.Join(want, " ") {
		t.Fatalf("index = %v, want %v", m.Index, want)
	}
	if m.Value.GetLabelAssociationAction().GetLabeled() {
		t.Fatal("unlabel must carry labeled=false")
	}
}

func TestLabelShapeErrors(t *testing.T) {
	// a nameless label refuses before any patch exists
	if msg := labelEditShapeError("", 0, false, false,
		""); msg == "" {
		t.Fatal("nameless create must refuse")
	}
	// deleting without an id refuses
	if msg := labelEditShapeError("", 0, false, true,
		""); msg == "" {
		t.Fatal("id-less delete must refuse")
	}
	// colors outside WhatsApp's twenty refuse
	if msg := labelEditShapeError("X", 20, true, false,
		""); msg == "" {
		t.Fatal("color 20 must refuse")
	}
	if msg := labelEditShapeError("X", -1, true, false,
		""); msg == "" {
		t.Fatal("color -1 must refuse")
	}
	// the valid shapes pass
	if msg := labelEditShapeError("X", 19, true, false,
		""); msg != "" {
		t.Fatalf("color 19 must pass, got %q", msg)
	}
	if msg := labelEditShapeError("", 0, false, true,
		"42"); msg != "" {
		t.Fatalf("id-carrying delete must pass, got %q",
			msg)
	}
	// association shapes
	if msg := labelAssocShapeError("", "42", "",
		false); msg == "" {
		t.Fatal("chat-less labeling must refuse")
	}
	if msg := labelAssocShapeError("c@g.us", "", "",
		false); msg == "" {
		t.Fatal("label-less labeling must refuse")
	}
	if msg := labelAssocShapeError("c@g.us", "42", "",
		true); msg == "" {
		t.Fatal("message labeling without an id " +
			"must refuse")
	}
}

func TestNoteAppStateRecordsLabels(t *testing.T) {
	b := &bridge{
		labelDefs:      map[string]*labelDef{},
		labelChatAssoc: map[string]map[string]bool{},
		labelMsgAssoc:  map[string]bool{},
	}
	name, color := "School fees", int32(4)
	b.noteAppState(&events.AppState{
		Index: []string{"label_edit", "42"},
		SyncActionValue: &waSyncAction.SyncActionValue{
			LabelEditAction: &waSyncAction.
				LabelEditAction{
				Name: &name, Color: &color},
		},
	})
	chat := "2348012345678@s.whatsapp.net"
	labeled := true
	b.noteAppState(&events.AppState{
		Index: []string{"label_jid", "42", chat},
		SyncActionValue: &waSyncAction.SyncActionValue{
			LabelAssociationAction: &waSyncAction.
				LabelAssociationAction{
				Labeled: &labeled},
		},
	})
	b.noteAppState(&events.AppState{
		Index: []string{"label_message", "42", chat,
			"MSGID1"},
		SyncActionValue: &waSyncAction.SyncActionValue{
			LabelAssociationAction: &waSyncAction.
				LabelAssociationAction{
				Labeled: &labeled},
		},
	})
	// an unrelated entry is ignored, not recorded
	b.noteAppState(&events.AppState{
		Index:           []string{"setting_pushName"},
		SyncActionValue: &waSyncAction.SyncActionValue{},
	})
	out := b.labelsPayload()
	defs, _ := out["labels"].([]any)
	if len(defs) != 1 {
		t.Fatalf("labels = %v, want the one definition",
			out["labels"])
	}
	row := defs[0].(map[string]any)
	if row["name"] != "School fees" || row["id"] != "42" {
		t.Fatalf("definition row = %v", row)
	}
	chats, _ := out["chat_labels"].(map[string]any)
	if got, _ := chats["42"].([]string); len(got) != 1 ||
		got[0] != chat {
		t.Fatalf("chat_labels = %v", out["chat_labels"])
	}
	msgs, _ := out["message_labels"].([]any)
	if len(msgs) != 1 {
		t.Fatalf("message_labels = %v",
			out["message_labels"])
	}
	if cov, _ := out["coverage"].(string); !strings.
		Contains(cov, "the sync is the read") {
		t.Fatalf("coverage = %q, want the honest reach",
			cov)
	}
}

func TestCatalogRefusalNamesTheShape(t *testing.T) {
	if !strings.Contains(catalogGoError, "whatsmeow") ||
		!strings.Contains(catalogGoError, "Baileys") {
		t.Fatalf("catalog refusal must name the missing "+
			"API and the engine that carries it: %q",
			catalogGoError)
	}
}
