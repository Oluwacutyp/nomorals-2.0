package main

// commerce.go — WhatsApp's commerce leftovers (parity
// bundle #17): labels, and the honest shape of the two
// rows this engine cannot carry.
//
// Labels are REAL here.  WhatsApp's label state lives in
// the "regular" app-state collection: definitions are
// LabelEditAction mutations and chat/message associations
// are LabelAssociationAction mutations, and whatsmeow
// ships the patch builders for all three
// (appstate.BuildLabelEdit / BuildLabelChat /
// BuildLabelMessage) plus SendAppState to land them.
// There is no label QUERY endpoint for a linked device —
// the read is the sync itself: FetchAppState replays the
// collection through events.AppState, and the bridge
// records every label mutation it sees (noteAppState)
// into the label memory the `labels` verb answers from.
// A write the server accepts is recorded the same way,
// so a read after a write shows the write.
//
// The catalog is NOT here, and the refusal says why in
// words: whatsmeow exposes no catalog API at all (its
// business surface is GetOrderDetails / GetBusinessProfile
// — no product list, no product write; the w:biz:catalog
// namespace is unimplemented).  Catalog verbs ride the
// Baileys engine, whose business socket carries them.
// Broadcast lists are not here either, at the library's
// own word: whatsmeow refuses non-status broadcast sends
// outright (ErrBroadcastListUnsupported, "sending to
// non-status broadcast lists is not yet supported"), so
// no verb is registered for them — the /broadcast
// command answers that truth on the Python side without
// a bridge round-trip that could only lie.

import (
	"context"
	"sort"
	"strconv"
	"strings"
	"time"

	"go.mau.fi/whatsmeow/appstate"
	waSyncAction "go.mau.fi/whatsmeow/proto/waSyncAction"
	"go.mau.fi/whatsmeow/types"
	"go.mau.fi/whatsmeow/types/events"
)

// commerceClient is the slice of *whatsmeow.Client the
// label WRITES ride — an interface so the patch SHAPES are
// pinnable in tests (the house standard: the Python side's
// recording fakes accept any shape; the bridge layer is
// where shapes must be pinned).  The read (cmdLabels) uses
// the concrete client: FetchAppState's full-sync replay
// only reaches noteAppState with the client's
// EmitAppStateEventsOnFullSync field set, and that field
// is not a method an interface can carry.
type commerceClient interface {
	SendAppState(ctx context.Context,
		patch appstate.PatchInfo) error
}

// ── shapes (words shared with the Baileys bridge and the
// Python adapter — one rule, refused before traffic) ─────

func labelEditShapeError(name string, color int32,
	hasColor bool, deleted bool, labelID string) string {
	if deleted && labelID == "" {
		return "deleting a label needs the label's id"
	}
	if !deleted && strings.TrimSpace(name) == "" {
		return "a label needs a name"
	}
	if hasColor && (color < 0 || color > 19) {
		return "label colors run 0 to 19 — WhatsApp's " +
			"twenty label colors"
	}
	return ""
}

func labelAssocShapeError(chat, labelID, messageID string,
	needMessage bool) string {
	if strings.TrimSpace(chat) == "" {
		return "labeling needs the chat"
	}
	if strings.TrimSpace(labelID) == "" {
		return "labeling needs the label's id"
	}
	if needMessage && strings.TrimSpace(messageID) == "" {
		return "labeling a message needs the message's id"
	}
	return ""
}

// ── the writes (app-state patches) ───────────────────────

func applyLabelEdit(ctx context.Context, cli commerceClient,
	labelID, name string, color int32, deleted bool) error {
	return cli.SendAppState(ctx,
		appstate.BuildLabelEdit(labelID, name, color,
			deleted))
}

func applyLabelChat(ctx context.Context, cli commerceClient,
	chat types.JID, labelID string, labeled bool) error {
	return cli.SendAppState(ctx,
		appstate.BuildLabelChat(chat, labelID, labeled))
}

func applyLabelMessage(ctx context.Context,
	cli commerceClient, chat types.JID, labelID,
	messageID string, labeled bool) error {
	return cli.SendAppState(ctx,
		appstate.BuildLabelMessage(chat, labelID,
			messageID, labeled))
}

// ── label memory (the sync is the read) ──────────────────

type labelDef struct {
	name       string
	color      int32
	hasColor   bool
	deleted    bool
	predefined int32
	hasPredef  bool
}

func (b *bridge) noteLabelEdit(labelID string,
	act *waSyncAction.LabelEditAction) {
	if act == nil || labelID == "" {
		return
	}
	b.labelsMu.Lock()
	defer b.labelsMu.Unlock()
	def := b.labelDefs[labelID]
	if def == nil {
		def = &labelDef{}
		b.labelDefs[labelID] = def
	}
	if act.Name != nil {
		def.name = *act.Name
	}
	if act.Color != nil {
		def.color = *act.Color
		def.hasColor = true
	}
	if act.PredefinedID != nil {
		def.predefined = *act.PredefinedID
		def.hasPredef = true
	}
	if act.Deleted != nil {
		def.deleted = *act.Deleted
	}
}

func (b *bridge) noteLabelAssoc(index []string,
	act *waSyncAction.LabelAssociationAction) {
	if act == nil || act.Labeled == nil ||
		len(index) < 3 {
		return
	}
	labelID, target := index[1], index[2]
	b.labelsMu.Lock()
	defer b.labelsMu.Unlock()
	switch index[0] {
	case appstate.IndexLabelAssociationChat:
		m := b.labelChatAssoc[labelID]
		if m == nil {
			m = map[string]bool{}
			b.labelChatAssoc[labelID] = m
		}
		m[target] = *act.Labeled
	case appstate.IndexLabelAssociationMessage:
		if len(index) < 4 {
			return
		}
		key := labelID + "|" + target + "|" + index[3]
		b.labelMsgAssoc[key] = *act.Labeled
	}
}

// noteAppState records the label slice of one app-state
// sync entry — definitions and associations — and ignores
// everything else the collection carries.
func (b *bridge) noteAppState(v *events.AppState) {
	if v == nil || len(v.Index) == 0 {
		return
	}
	switch v.Index[0] {
	case appstate.IndexLabelEdit:
		if len(v.Index) >= 2 {
			b.noteLabelEdit(v.Index[1],
				v.LabelEditAction)
		}
	case appstate.IndexLabelAssociationChat,
		appstate.IndexLabelAssociationMessage:
		b.noteLabelAssoc(v.Index,
			v.LabelAssociationAction)
	}
}

const labelsCoverage = "the account's label state as " +
	"its app-state sync carries it — fetched from " +
	"WhatsApp's sync for this read (a linked device gets " +
	"no label query endpoint; the sync is the read)"

// labelsPayload renders the label memory as the verb's
// reply body: definitions sorted by id for a stable read,
// chat associations grouped by label, message
// associations listed — tombstoned definitions ride with
// their deleted flag set, and the Python side decides
// what a tombstone means for display.
func (b *bridge) labelsPayload() map[string]any {
	b.labelsMu.Lock()
	defer b.labelsMu.Unlock()
	ids := make([]string, 0, len(b.labelDefs))
	for id := range b.labelDefs {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	defs := make([]any, 0, len(ids))
	for _, id := range ids {
		def := b.labelDefs[id]
		row := map[string]any{
			"id": id, "name": def.name,
			"deleted": def.deleted,
		}
		if def.hasColor {
			row["color"] = def.color
		}
		if def.hasPredef {
			row["predefined_id"] = def.predefined
		}
		defs = append(defs, row)
	}
	chatLabels := map[string]any{}
	for labelID, m := range b.labelChatAssoc {
		chats := make([]string, 0, len(m))
		for chat, labeled := range m {
			if labeled {
				chats = append(chats, chat)
			}
		}
		if len(chats) > 0 {
			sort.Strings(chats)
			chatLabels[labelID] = chats
		}
	}
	msgLabels := make([]any, 0)
	for key, labeled := range b.labelMsgAssoc {
		if !labeled {
			continue
		}
		parts := strings.SplitN(key, "|", 3)
		if len(parts) != 3 {
			continue
		}
		msgLabels = append(msgLabels, map[string]any{
			"label_id": parts[0], "chat": parts[1],
			"message_id": parts[2],
		})
	}
	return map[string]any{
		"ok": true, "labels": defs,
		"chat_labels":    chatLabels,
		"message_labels": msgLabels,
		"coverage":       labelsCoverage,
	}
}

// ── verbs ────────────────────────────────────────────────

// cmdLabels answers the account's labels.  The read is a
// fetch: a full sync the first time per process (the
// memory starts empty and only a full replay proves it
// complete — EmitAppStateEventsOnFullSync makes the
// replay visible to noteAppState), an incremental
// catch-up after that.  A failed fetch is a failed read:
// the error answers, no state is invented.
func (b *bridge) cmdLabels(ctx context.Context,
	cc *clientConn, id string, req request) {
	b.labelsMu.Lock()
	full := !b.labelsFullSync
	b.labelsMu.Unlock()
	b.cli.EmitAppStateEventsOnFullSync = true
	if err := b.cli.FetchAppState(ctx,
		appstate.WAPatchRegular, full, false); err != nil {
		b.fail(cc, id, err)
		return
	}
	if full {
		b.labelsMu.Lock()
		b.labelsFullSync = true
		b.labelsMu.Unlock()
	}
	b.reply(cc, id, b.labelsPayload())
}

// cmdLabelEdit lands ONE label-definition write.  The
// app-state record is whole-record — name, color and the
// deleted flag travel together in a single mutation — so
// a rename carries the label's current color with it;
// the Python side resolves the full record from a fresh
// labels read before asking for the write.
func (b *bridge) cmdLabelEdit(ctx context.Context,
	cc *clientConn, id string, req request) {
	// the label's id rides as label_id: the wire's bare
	// "id" is the request's correlation id.
	labelID := strings.TrimSpace(req.str("label_id"))
	name := strings.TrimSpace(req.str("name"))
	deleted := req.boolean("deleted")
	hasColor := req.has("color")
	color := int32(req.num("color", 0))
	if msg := labelEditShapeError(name, color, hasColor,
		deleted, labelID); msg != "" {
		b.reply(cc, id, map[string]any{"ok": false,
			"error": msg})
		return
	}
	if labelID == "" {
		// A new label's id is minted here — WhatsApp's
		// own clients mint numeric ids client-side, and
		// the creation is only real once the patch lands.
		labelID = strconv.FormatInt(
			time.Now().UnixMilli(), 10)
	}
	if err := applyLabelEdit(ctx, b.cli, labelID, name,
		color, deleted); err != nil {
		b.fail(cc, id, err)
		return
	}
	b.noteLabelEdit(labelID, &waSyncAction.LabelEditAction{
		Name: &name, Color: &color, Deleted: &deleted})
	b.reply(cc, id, map[string]any{
		"ok": true, "label_id": labelID, "name": name,
		"color": color, "deleted": deleted})
}

func (b *bridge) cmdLabelChat(ctx context.Context,
	cc *clientConn, id string, req request) {
	labelID := strings.TrimSpace(req.str("label_id"))
	labeled := true
	if req.has("labeled") {
		labeled = req.boolean("labeled")
	}
	if msg := labelAssocShapeError(req.str("chat"),
		labelID, "", false); msg != "" {
		b.reply(cc, id, map[string]any{"ok": false,
			"error": msg})
		return
	}
	chat, err := parseJID(req.str("chat"))
	if err != nil {
		b.fail(cc, id, err)
		return
	}
	if err := applyLabelChat(ctx, b.cli, chat, labelID,
		labeled); err != nil {
		b.fail(cc, id, err)
		return
	}
	b.noteLabelAssoc(
		[]string{appstate.IndexLabelAssociationChat,
			labelID, chat.String()},
		&waSyncAction.LabelAssociationAction{
			Labeled: &labeled})
	b.reply(cc, id, map[string]any{
		"ok": true, "chat": chat.String(),
		"label_id": labelID, "labeled": labeled})
}

func (b *bridge) cmdLabelMessage(ctx context.Context,
	cc *clientConn, id string, req request) {
	labelID := strings.TrimSpace(req.str("label_id"))
	messageID := strings.TrimSpace(req.str("message_id"))
	labeled := true
	if req.has("labeled") {
		labeled = req.boolean("labeled")
	}
	if msg := labelAssocShapeError(req.str("chat"),
		labelID, messageID, true); msg != "" {
		b.reply(cc, id, map[string]any{"ok": false,
			"error": msg})
		return
	}
	chat, err := parseJID(req.str("chat"))
	if err != nil {
		b.fail(cc, id, err)
		return
	}
	if err := applyLabelMessage(ctx, b.cli, chat, labelID,
		messageID, labeled); err != nil {
		b.fail(cc, id, err)
		return
	}
	b.noteLabelAssoc(
		[]string{appstate.IndexLabelAssociationMessage,
			labelID, chat.String(), messageID},
		&waSyncAction.LabelAssociationAction{
			Labeled: &labeled})
	b.reply(cc, id, map[string]any{
		"ok": true, "chat": chat.String(),
		"label_id": labelID, "message_id": messageID,
		"labeled": labeled})
}

// The catalog refusal, worded once: whatsmeow has no
// catalog API to ride, on either catalog verb.
const catalogGoError = "the catalog is not reachable " +
	"on the Go bridge — whatsmeow exposes no catalog " +
	"API (its business surface reads orders and business " +
	"profiles only; the w:biz:catalog namespace is not " +
	"implemented). Catalog verbs ride the Baileys engine."

func (b *bridge) cmdCatalog(cc *clientConn, id string,
	req request) {
	b.reply(cc, id, map[string]any{"ok": false,
		"error": catalogGoError})
}

func (b *bridge) cmdCatalogProduct(cc *clientConn,
	id string, req request) {
	b.reply(cc, id, map[string]any{"ok": false,
		"error": catalogGoError})
}
