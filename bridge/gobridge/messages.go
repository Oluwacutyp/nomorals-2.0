// messages.go — the adapter protocol's core: inbound message /
// receipt events and the send / send_media / typing / history /
// search / read / chats / status commands. Shapes are the old bridge's,
// field for field (see whatsapp-bridge.mjs's header, now deleted —
// the contract lives on here and in the Python adapter's docstring).
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"go.mau.fi/whatsmeow"
	"go.mau.fi/whatsmeow/appstate"
	waE2E "go.mau.fi/whatsmeow/proto/waE2E"
	"go.mau.fi/whatsmeow/types"
	"go.mau.fi/whatsmeow/types/events"
	"google.golang.org/protobuf/proto"
)

const maxKeepPerChat = 200

// ── request decoding helpers (the wire is dynamic JSON) ───────────

type request map[string]any

func (r request) str(key string) string {
	if v, ok := r[key].(string); ok {
		return v
	}
	return ""
}

func (r request) num(key string, def int) int {
	switch v := r[key].(type) {
	case float64:
		return int(v)
	case string:
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

// flt reads a JSON number as float64.  The second return is
// presence: coordinates are float-typed AND tri-state (an
// absent latitude is not 0.0 — 0,0 is a real place in the
// Gulf of Guinea, and a missing one is a shape error).
func (r request) flt(key string) (float64, bool) {
	switch v := r[key].(type) {
	case float64:
		return v, true
	case string:
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f, true
		}
	}
	return 0, false
}

func (r request) boolean(key string) bool {
	v, _ := r[key].(bool)
	return v
}

// has reports whether the key is present at all — chat_modify's
// fields are tri-state (absent = leave it, present = set it),
// which a zero-value read cannot express.
func (r request) has(key string) bool {
	_, ok := r[key]
	return ok
}

// ── TCP client serving + command dispatch ─────────────────────────

func (b *bridge) serveClient(cc *clientConn) {
	defer b.dropClient(cc)
	scanner := bufio.NewScanner(cc.conn)
	scanner.Buffer(make([]byte, 64*1024), 16*1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var req request
		if err := json.Unmarshal([]byte(line), &req); err != nil {
			continue
		}
		if cmd, _ := req["cmd"].(string); cmd != "" {
			go b.handleCommand(cc, req)
		}
	}
}

func (b *bridge) handleCommand(cc *clientConn, req request) {
	id := req.str("id")
	cmd := req.str("cmd")
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	switch cmd {
	case "send":
		b.cmdSend(ctx, cc, id, req)
	case "edit":
		b.cmdEdit(ctx, cc, id, req)
	case "send_media":
		b.cmdSendMedia(ctx, cc, id, req)
	case "typing":
		b.cmdTyping(ctx, cc, id, req)
	case "history":
		b.cmdHistory(cc, id, req)
	case "search":
		b.cmdSearch(cc, id, req)
	case "read":
		b.cmdRead(ctx, cc, id, req)
	case "presence":
		b.cmdPresence(ctx, cc, id, req)
	case "status_view":
		b.cmdStatusView(ctx, cc, id, req)
	case "status_send":
		b.cmdStatusSend(ctx, cc, id, req)
	case "status_react":
		b.cmdStatusReact(ctx, cc, id, req)
	case "react":
		b.cmdReact(ctx, cc, id, req)
	case "forward":
		b.cmdForward(ctx, cc, id, req)
	case "chat_modify":
		b.cmdChatModify(ctx, cc, id, req)
	case "message_star":
		b.cmdMessageStar(ctx, cc, id, req)
	case "reject_call":
		b.cmdRejectCall(ctx, cc, id, req)
	case "blocklist":
		b.cmdBlocklist(ctx, cc, id, req)
	case "chat_disappear":
		b.cmdChatDisappear(ctx, cc, id, req)
	case "profile_get":
		b.cmdProfileGet(ctx, cc, id, req)
	case "profile_set":
		b.cmdProfileSet(ctx, cc, id, req)
	case "privacy_get":
		b.cmdPrivacyGet(ctx, cc, id, req)
	case "privacy_set":
		b.cmdPrivacySet(ctx, cc, id, req)
	case "send_poll":
		b.cmdSendPoll(ctx, cc, id, req)
	case "poll_results":
		b.cmdPollResults(ctx, cc, id, req)
	case "send_location":
		b.cmdSendLocation(ctx, cc, id, req)
	case "send_contact":
		b.cmdSendContact(ctx, cc, id, req)
	case "check_number":
		b.cmdCheckNumber(ctx, cc, id, req)
	case "chats":
		b.cmdChats(cc, id, req)
	case "status":
		state := "closed"
		if b.cli.IsLoggedIn() {
			state = "open"
		}
		b.reply(cc, id, map[string]any{"ok": true, "program": bridgeVersion, "port": b.port, "user": b.meJID, "user_name": b.myPushName(), "lid": b.myLID(), "state": state})
	default:
		if b.handleGroupCommand(ctx, cc, id, req) {
			return
		}
		if b.handleGroupAdminCommand(ctx, cc, id, req) {
			return
		}
		b.reply(cc, id, map[string]any{"ok": false, "error": "unknown cmd: " + cmd})
	}
}

func (b *bridge) fail(cc *clientConn, id string, err error) {
	b.reply(cc, id, map[string]any{"ok": false, "error": err.Error()})
}

// ── text / media extraction from an E2E message ───────────────────

// textOf mirrors the old bridge: button/list taps re-enter as the
// command they carry ("/" + id) so a keyboard authored once works
// on every platform; otherwise conversation / extended text /
// media caption.
func textOf(m *waE2E.Message) string {
	if m == nil {
		return ""
	}
	btnID := ""
	switch {
	case m.GetButtonsResponseMessage() != nil:
		btnID = m.GetButtonsResponseMessage().GetSelectedButtonID()
	case m.GetTemplateButtonReplyMessage() != nil:
		btnID = m.GetTemplateButtonReplyMessage().GetSelectedID()
	case m.GetListResponseMessage() != nil:
		btnID = m.GetListResponseMessage().GetSingleSelectReply().GetSelectedRowID()
	}
	if btnID != "" {
		return "/" + btnID
	}
	if flow := m.GetInteractiveResponseMessage().GetNativeFlowResponseMessage().GetParamsJSON(); flow != "" {
		var parsed map[string]any
		if err := json.Unmarshal([]byte(flow), &parsed); err == nil {
			if fid, ok := parsed["id"].(string); ok && fid != "" {
				return "/" + fid
			}
		}
	}
	switch {
	case m.GetConversation() != "":
		return m.GetConversation()
	case m.GetExtendedTextMessage() != nil:
		return m.GetExtendedTextMessage().GetText()
	case m.GetImageMessage() != nil:
		return m.GetImageMessage().GetCaption()
	case m.GetVideoMessage() != nil:
		return m.GetVideoMessage().GetCaption()
	}
	return ""
}

// fromOf builds the wire "from" object for an inbound event:
// the sender's addressing id + push name, plus "alt" — the
// sender's OTHER addressing form (LID ↔ phone JID) when
// whatsmeow reports one. Identity resolution on the Python
// side folds both forms of one person into one identity;
// without alt, an LID-addressed sender and the same person
// number-addressed look like two different people.
func fromOf(v *events.Message) map[string]any {
	from := map[string]any{
		"id":   v.Info.Sender.ToNonAD().String(),
		"name": v.Info.PushName,
	}
	if alt := v.Info.SenderAlt.ToNonAD(); !alt.IsEmpty() {
		from["alt"] = alt.String()
	}
	return from
}

func contextInfoOf(m *waE2E.Message) *waE2E.ContextInfo {
	if m == nil {
		return nil
	}
	candidates := []*waE2E.ContextInfo{
		m.GetExtendedTextMessage().GetContextInfo(),
		m.GetImageMessage().GetContextInfo(),
		m.GetVideoMessage().GetContextInfo(),
		m.GetAudioMessage().GetContextInfo(),
		m.GetDocumentMessage().GetContextInfo(),
		m.GetStickerMessage().GetContextInfo(),
		// Button/list taps: the response's context stanza names
		// the message that carried the buttons — the live-session
		// presentation path edits THAT message in place, so a tap
		// must report it as reply_to like a quote does.
		m.GetButtonsResponseMessage().GetContextInfo(),
		m.GetTemplateButtonReplyMessage().GetContextInfo(),
		m.GetListResponseMessage().GetContextInfo(),
		m.GetInteractiveResponseMessage().GetContextInfo(),
	}
	for _, ci := range candidates {
		if ci != nil {
			return ci
		}
	}
	return nil
}

func mediaKindOf(m *waE2E.Message) string {
	switch {
	case m == nil:
		return ""
	case m.GetImageMessage() != nil:
		return "image"
	case m.GetVideoMessage() != nil:
		return "video"
	case m.GetAudioMessage() != nil:
		if m.GetAudioMessage().GetPTT() {
			return "voice"
		}
		return "audio"
	case m.GetDocumentMessage() != nil:
		return "file"
	case m.GetStickerMessage() != nil:
		return "sticker"
	}
	return ""
}

var unsafeIDChars = regexp.MustCompile(`[^a-zA-Z0-9_-]`)

// unwrapInbound resolves the message an inbound event actually
// carries, and whether it is view-once in ANY real shape.
//
// whatsmeow's dispatch runs events.Message.UnwrapRaw before
// handlers see the event, so on the standard path v.Message is
// already the inner message and v.IsViewOnce names the wrapper
// shapes (ViewOnceMessage / V2 / V2Extension, ephemeral-wrapped
// included).  Two real shapes survive that unwrap, and both
// were invisible to this bridge for the feature's whole life —
// the owner's journal held zero view-once rows of any kind
// while his capture toggle sat ON:
//
//   - the FLAG form: a bare ImageMessage / VideoMessage /
//     AudioMessage carrying ViewOnce=true, no wrapper at all.
//     UnwrapRaw never folds the media flag into IsViewOnce, so
//     the arrival stamped view_once:false and read as an
//     ordinary photo — no capture attempted, nothing journaled.
//   - an event that reaches this handler STILL wrapped (any
//     dispatch path that skipped UnwrapRaw): the media sits
//     inside the wrapper where mediaKindOf cannot see it, so
//     nothing downloads either.
//
// Classification therefore happens here, once, over every
// container and the flag, and the returned message — wrappers
// resolved — is what text/media/context extraction reads.
func unwrapInbound(v *events.Message) (*waE2E.Message, bool) {
	msg := v.Message
	viewOnce := v.IsViewOnce
	for depth := 0; depth < 4 && msg != nil; depth++ {
		inner := msg
		switch {
		case msg.GetDeviceSentMessage().GetMessage() != nil:
			inner = msg.GetDeviceSentMessage().GetMessage()
		case msg.GetEphemeralMessage().GetMessage() != nil:
			inner = msg.GetEphemeralMessage().GetMessage()
		case msg.GetViewOnceMessage().GetMessage() != nil:
			inner = msg.GetViewOnceMessage().GetMessage()
			viewOnce = true
		case msg.GetViewOnceMessageV2().GetMessage() != nil:
			inner = msg.GetViewOnceMessageV2().GetMessage()
			viewOnce = true
		case msg.GetViewOnceMessageV2Extension().GetMessage() != nil:
			inner = msg.GetViewOnceMessageV2Extension().GetMessage()
			viewOnce = true
		}
		if inner == msg {
			break
		}
		msg = inner
	}
	if msg != nil && (msg.GetImageMessage().GetViewOnce() ||
		msg.GetVideoMessage().GetViewOnce() ||
		msg.GetAudioMessage().GetViewOnce()) {
		viewOnce = true
	}
	return msg, viewOnce
}

// downloadInbound fetches a message's media into the media dir and
// returns its wire row {path, mime, kind}, or nil.
func (b *bridge) downloadInbound(ctx context.Context, msg *waE2E.Message, idStr string) map[string]any {
	return b.downloadMediaOf(ctx, msg, idStr)
}

// mediaPartsOf extracts the downloadable media one message proto
// carries: its kind, the whatsmeow download handle, the mime,
// and (documents only) the sender-given file name.
func mediaPartsOf(msg *waE2E.Message) (kind string,
	dl whatsmeow.DownloadableMessage, mime, fileName string) {
	kind = mediaKindOf(msg)
	switch kind {
	case "image":
		dl, mime = msg.GetImageMessage(), msg.GetImageMessage().GetMimetype()
	case "video":
		dl, mime = msg.GetVideoMessage(), msg.GetVideoMessage().GetMimetype()
	case "voice", "audio":
		dl, mime = msg.GetAudioMessage(), msg.GetAudioMessage().GetMimetype()
	case "file":
		dl = msg.GetDocumentMessage()
		mime = msg.GetDocumentMessage().GetMimetype()
		fileName = msg.GetDocumentMessage().GetFileName()
	case "sticker":
		dl, mime = msg.GetStickerMessage(), msg.GetStickerMessage().GetMimetype()
	}
	return kind, dl, mime, fileName
}

// mediaExtFor is the cache file extension for one media kind:
// a document's own file name wins, the kind map covers the rest.
func mediaExtFor(kind, fileName string) string {
	if idx := strings.LastIndex(fileName, "."); idx >= 0 {
		return strings.ToLower(fileName[idx:])
	}
	ext := map[string]string{
		"image": ".jpg", "video": ".mp4", "voice": ".ogg",
		"audio": ".ogg", "sticker": ".webp",
	}[kind]
	if ext == "" {
		ext = ".bin"
	}
	return ext
}

// mediaCacheName maps one media identity (a WhatsApp message
// id) + extension to the ONE cache filename that identity ever
// gets — the dedup key of the whole cache (media-cache
// discipline, 2026-11-08).  The name used to be
// "<download-time-millis>-<id fragment>", which made every
// download of the same media a new file by construction: the
// channel sweep re-served the newest posts every housekeeping
// cycle and one 76MB channel video accumulated ~145 copies in a
// morning, 11.07 GB in the directory, until the disk filled and
// /update died on it.  The identity is the FULL sanitized id
// (capped at 64 chars — real ids fit whole), never the old
// 12-char fragment, so two different messages cannot collide
// into one file.  "" when there is no identity to dedup by.
func mediaCacheName(idStr, ext string) string {
	short := unsafeIDChars.ReplaceAllString(idStr, "")
	if len(short) > 64 {
		short = short[:64]
	}
	if short == "" {
		return ""
	}
	return short + ext
}

// probeMedia is the cache's existence check: the wire row for
// a message whose media identity already has a complete file
// in the media dir, or false.  Both layers use it — the
// trigger sites (the channel_fetch loop, which re-serves the
// same posts every sweep) check before asking for a download,
// and downloadMediaOf checks again at the writer, so a
// duplicate event that slips past a trigger (a redelivery, a
// concurrent handler) still cannot produce a sibling copy.
func (b *bridge) probeMedia(msg *waE2E.Message, idStr string) (map[string]any, bool) {
	kind, _, mime, fileName := mediaPartsOf(msg)
	if kind == "" {
		return nil, false
	}
	name := mediaCacheName(idStr, mediaExtFor(kind, fileName))
	if name == "" {
		return nil, false
	}
	p := filepath.Join(b.mediaDir, name)
	st, err := os.Stat(p)
	if err != nil || st.IsDir() || st.Size() == 0 {
		return nil, false
	}
	if mime == "" {
		mime = "application/octet-stream"
	}
	return map[string]any{"path": p, "mime": mime, "kind": kind}, true
}

// mediaLock is the per-identity download lock: one download of
// an identity at a time per bridge process, so concurrent
// triggers of the same media serialize into ONE fetch + write
// and the rest serve the finished file through the probe.
func (b *bridge) mediaLock(name string) *sync.Mutex {
	m, _ := b.mediaLocks.LoadOrStore(name, &sync.Mutex{})
	return m.(*sync.Mutex)
}

// downloadMediaOf downloads the media carried by one message proto
// (a live message, or the quoted message inside a reply's
// contextInfo — the bytes are addressed the same way) into the
// media dir and returns its wire row {path, mime, kind}, or nil.
// A media identity that already has its file is SERVED, never
// re-downloaded (see mediaCacheName).
func (b *bridge) downloadMediaOf(ctx context.Context, msg *waE2E.Message, idStr string) map[string]any {
	kind, dl, mime, fileName := mediaPartsOf(msg)
	if kind == "" || dl == nil {
		return nil
	}
	if mime == "" {
		mime = "application/octet-stream"
	}
	if row, ok := b.probeMedia(msg, idStr); ok {
		return row
	}
	ext := mediaExtFor(kind, fileName)
	name := mediaCacheName(idStr, ext)
	if name == "" {
		// No identity to dedup by: the legacy timestamped name
		// keeps the bytes reachable.  Every real WhatsApp
		// message carries an id, so this is the defensive
		// path, not the operating one.
		name = strconv.FormatInt(time.Now().UnixMilli(), 10) +
			"-anon" + ext
	} else {
		lock := b.mediaLock(name)
		lock.Lock()
		defer lock.Unlock()
		if row, ok := b.probeMedia(msg, idStr); ok {
			return row
		}
	}
	data, err := b.cli.Download(ctx, dl)
	if err != nil {
		// The media ARRIVED but its bytes could not be fetched:
		// emit the entry with an empty path so the payload carries
		// the trace (media arrived, unseen) instead of looking like
		// no media at all. The Python side stamps pathless entries
		// media_unavailable.
		logf("media download failed: %v", err)
		return map[string]any{"path": "", "mime": mime, "kind": kind}
	}
	p := filepath.Join(b.mediaDir, name)
	// Write to a sibling temp and rename into place: a reader
	// (the Python side copying a capture into the vault) never
	// meets a half-written file, and a writer that loses a race
	// across processes serves the winner's complete file
	// instead of clobbering it.
	tmp := p + ".part"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		logf("media write failed: %v", err)
		return map[string]any{"path": "", "mime": mime, "kind": kind}
	}
	if err := os.Rename(tmp, p); err != nil {
		os.Remove(tmp)
		if _, statErr := os.Stat(p); statErr == nil {
			return map[string]any{"path": p, "mime": mime, "kind": kind}
		}
		logf("media write failed: %v", err)
		return map[string]any{"path": "", "mime": mime, "kind": kind}
	}
	return map[string]any{"path": p, "mime": mime, "kind": kind}
}

// quotedOf resolves a message's reply quote: the quoted message
// proto inside the contextInfo, or nil when the message quotes
// nothing. A reply on WhatsApp carries the quoted message IN the
// reply itself — the stanza id alone (all the event used to send)
// tells the Python side THAT a message was quoted, never WHAT.
func quotedOf(m *waE2E.Message) *waE2E.Message {
	ci := contextInfoOf(m)
	if ci == nil {
		return nil
	}
	return ci.GetQuotedMessage()
}

// stampQuoted adds the reply quote's payload to an inbound event:
// "quoted_media" (the quoted message's media, downloaded now —
// the only moment its bytes are reachable) and "quoted_text".
// Reply-media commands (/sticker on a replied photo, /convert,
// /steal) read these; without them they saw a bare stanza id and
// fell back to usage text while the photo sat in the quote.
func (b *bridge) stampQuoted(ctx context.Context, out map[string]any, m *waE2E.Message) {
	q := quotedOf(m)
	if q == nil {
		return
	}
	if text := textOf(q); text != "" {
		out["quoted_text"] = text
	}
	stanza := ""
	if ci := contextInfoOf(m); ci != nil {
		stanza = ci.GetStanzaID()
	}
	if row := b.downloadMediaOf(ctx, q, stanza); row != nil {
		out["quoted_media"] = []map[string]any{row}
	}
}

// ── inbound events ────────────────────────────────────────────────

func (b *bridge) onMessage(v *events.Message) {
	if v.Message == nil {
		return
	}
	chat := v.Info.Chat.ToNonAD()
	sender := v.Info.Sender.ToNonAD()
	tsMs := v.Info.Timestamp.UnixMilli()

	// Protocol messages are not conversation: REVOKE (0) is a
	// delete-for-everyone, MESSAGE_EDIT arrives unwrapped
	// (IsEdit) with its target in the raw wrapper. Broadcast as
	// the evidentiary feeds the Python adapter carries.
	if pm := v.Message.GetProtocolMessage(); pm != nil {
		target := pm.GetKey().GetID()
		if target != "" {
			switch pm.GetType() {
			case waE2E.ProtocolMessage_REVOKE:
				b.broadcast(map[string]any{
					"type": "revoke", "id": target,
					"chat": map[string]any{"id": chat.String(), "kind": kindOf(chat)},
					"from": map[string]any{"id": sender.String()},
					"ts":   tsMs,
				})
				return
			}
		}
	}
	if v.IsEdit {
		target := ""
		if pm := v.RawMessage.GetProtocolMessage(); pm != nil {
			target = pm.GetKey().GetID()
		}
		if target != "" {
			b.broadcast(map[string]any{
				"type": "edit", "id": target,
				"chat": map[string]any{"id": chat.String(), "kind": kindOf(chat)},
				"from": map[string]any{"id": sender.String(), "name": v.Info.PushName},
				"text": textOf(v.Message),
				"ts":   tsMs,
			})
			return
		}
		logf("edit event without a target id — dropped (message %s)", v.Info.ID)
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	// Classify once, here: the message this event actually
	// carries (wrappers resolved) and whether it is view-once
	// in any real shape.  Everything below — text, media
	// download, context, the view_once stamp — reads these,
	// never the raw event fields, so a shape the upstream
	// unwrap missed cannot slip past as an ordinary message.
	msg, isViewOnce := unwrapInbound(v)

	// Poll memory (parity G3): creations teach the bridge a
	// poll's option names; votes arrive encrypted and are
	// tallied as they decrypt.  Neither is conversation — the
	// capture is a side ledger on the same event, and the
	// message itself flows on untouched below.
	b.notePollEvent(ctx, v)

	if chat == types.StatusBroadcastJID {
		if v.Info.IsFromMe {
			return
		}
		var media []map[string]any
		if m := b.downloadInbound(ctx, msg, string(v.Info.ID)); m != nil {
			media = []map[string]any{m}
		}
		b.broadcast(map[string]any{
			"type": "status_post", "id": string(v.Info.ID),
			"from":      fromOf(v),
			"text":      textOf(msg),
			"media":     media,
			"view_once": isViewOnce,
			"ts":        tsMs,
		})
		return
	}

	kind := kindOf(chat)
	if v.Info.IsFromMe {
		ownReplyTo := ""
		if ci := contextInfoOf(msg); ci != nil {
			ownReplyTo = ci.GetStanzaID()
		}
		ownName := v.Info.PushName
		if ownName == "" {
			ownName = b.myPushName()
		}
		out := map[string]any{
			"type": "own_message", "id": string(v.Info.ID),
			"chat": map[string]any{
				"id": chat.String(), "kind": kind, "title": b.titleOf(chat),
			},
			// The sender IS the account itself. Name it: the
			// adapter resolves the owner's display identity from
			// this — without it, his own turns arrive nameless
			// and downstream identity resolution degrades to a
			// placeholder (register #29).
			"from":      map[string]any{"id": sender.String(), "name": ownName},
			"text":      textOf(msg),
			"reply_to":  ownReplyTo,
			"view_once": isViewOnce,
			"ts":        tsMs,
		}
		b.stampQuoted(ctx, out, msg)
		b.remember(chat, &remembered{
			incoming: false, text: out["text"].(string),
			id: v.Info.ID, chatJID: chat, senderJID: sender, tsMs: tsMs,
		})
		b.broadcast(out)
		return
	}

	title := b.titleFor(ctx, chat, kind, v.Info.PushName, sender)
	var media []map[string]any
	if m := b.downloadInbound(ctx, msg, string(v.Info.ID)); m != nil {
		media = []map[string]any{m}
	}
	mentioned := false
	mentionList := []string{}
	if ci := contextInfoOf(msg); ci != nil {
		for _, mj := range ci.GetMentionedJID() {
			mentionList = append(mentionList, mj)
			parsed, err := types.ParseJID(mj)
			if err != nil {
				continue
			}
			if b.isMe(parsed.ToNonAD()) {
				mentioned = true
			}
		}
	}
	replyTo := ""
	if ci := contextInfoOf(msg); ci != nil {
		replyTo = ci.GetStanzaID()
	}
	from := fromOf(v)
	out := map[string]any{
		"type": "message", "id": string(v.Info.ID),
		"chat":      map[string]any{"id": chat.String(), "kind": kind, "title": title},
		"from":      from,
		"text":      textOf(msg),
		"media":     media,
		"reply_to":  replyTo,
		"mentioned": mentioned,
		// The full mention payload (chat & privacy wave — the
		// antitag sensor counts it; the Python envelope already
		// reads a "mentions" list when a bridge sends one).
		"mentions":  mentionList,
		"view_once": isViewOnce,
		"ts":        tsMs,
	}
	b.stampQuoted(ctx, out, msg)
	if v.Info.ID != "" {
		b.cacheMessage(string(v.Info.ID), msg, sender)
	}
	b.remember(chat, &remembered{
		incoming: true, text: out["text"].(string), sender: v.Info.PushName,
		media: media, replyTo: replyTo,
		id: v.Info.ID, chatJID: chat, senderJID: sender, tsMs: tsMs,
	})
	b.broadcast(out)
}

func (b *bridge) isMe(jid types.JID) bool {
	if b.cli.Store.ID != nil && jid == b.cli.Store.ID.ToNonAD() {
		return true
	}
	if !b.cli.Store.LID.IsEmpty() && jid == b.cli.Store.LID.ToNonAD() {
		return true
	}
	return false
}

func (b *bridge) titleOf(chat types.JID) string {
	b.memMu.Lock()
	defer b.memMu.Unlock()
	return b.titles[chat.String()]
}

// titleFor resolves a chat title like the old bridge: groups from
// metadata (cached after first fetch), DMs from the push name.
func (b *bridge) titleFor(ctx context.Context, chat types.JID, kind, pushName string, sender types.JID) string {
	key := chat.String()
	b.memMu.Lock()
	title := b.titles[key]
	b.memMu.Unlock()
	if title == "" && kind == "group" {
		gctx, cancel := context.WithTimeout(ctx, 10*time.Second)
		if info, err := b.cli.GetGroupInfo(gctx, chat); err == nil && info.Name != "" {
			title = info.Name
		}
		cancel()
		if title != "" {
			b.memMu.Lock()
			b.titles[key] = title
			b.memMu.Unlock()
		}
	}
	if title == "" && kind == "channel" {
		// A channel post's sender IS the channel — there is no
		// push name to fall back to. The newsletter metadata
		// carries the channel's real name; cache it like a
		// group's, so the registry note names the channel
		// instead of filing it under its numeric JID.
		gctx, cancel := context.WithTimeout(ctx, 10*time.Second)
		if meta, err := b.cli.GetNewsletterInfo(gctx, chat); err == nil && meta.ThreadMeta.Name.Text != "" {
			title = meta.ThreadMeta.Name.Text
		}
		cancel()
		if title != "" {
			b.memMu.Lock()
			b.titles[key] = title
			b.memMu.Unlock()
		}
	}
	if title == "" {
		title = pushName
	}
	if title == "" {
		title = sender.User
	}
	if kind == "dm" && pushName != "" {
		b.memMu.Lock()
		b.titles[key] = pushName
		b.memMu.Unlock()
	}
	return title
}

func (b *bridge) remember(chat types.JID, rec *remembered) {
	key := chat.String()
	b.memMu.Lock()
	defer b.memMu.Unlock()
	arr := b.recent[key]
	arr = append(arr, rec)
	if len(arr) > maxKeepPerChat {
		arr = arr[len(arr)-maxKeepPerChat:]
	}
	b.recent[key] = arr
}

// cachedSender returns the author of a remembered message, or
// types.EmptyJID when the bridge never saw it. Message-moderation
// verbs (revoke, pin) need the author to build the message key
// for someone else's message.
func (b *bridge) cachedSender(id string) types.JID {
	b.memMu.Lock()
	defer b.memMu.Unlock()
	if cached, ok := b.msgCache[id]; ok && cached != nil {
		return cached.sender
	}
	return types.EmptyJID
}

func (b *bridge) cacheMessage(id string, msg *waE2E.Message, sender types.JID) {
	b.memMu.Lock()
	defer b.memMu.Unlock()
	if _, ok := b.msgCache[id]; !ok {
		b.cacheKeys = append(b.cacheKeys, id)
	}
	b.msgCache[id] = &cachedMsg{msg: msg, sender: sender}
	for len(b.msgCache) > 2000 {
		// evict oldest down to 1500, like the old bridge
		drop := len(b.msgCache) - 1500
		for _, k := range b.cacheKeys[:drop] {
			delete(b.msgCache, k)
		}
		b.cacheKeys = b.cacheKeys[drop:]
	}
}

// onReceipt maps whatsmeow receipt types onto the wire buckets the
// adapter tracks: "" (delivered), "read"/"played" → read.
func (b *bridge) onReceipt(v *events.Receipt) {
	var kind string
	switch v.Type {
	case types.ReceiptTypeDelivered:
		kind = "delivered"
	case types.ReceiptTypeRead, types.ReceiptTypePlayed:
		kind = "read"
	default:
		return
	}
	if len(v.MessageIDs) == 0 {
		return
	}
	ids := make([]string, 0, len(v.MessageIDs))
	for _, id := range v.MessageIDs {
		if id != "" {
			ids = append(ids, string(id))
		}
	}
	if len(ids) == 0 {
		return
	}
	b.broadcast(map[string]any{
		"type": "receipt", "chat": v.Chat.ToNonAD().String(),
		"ids": ids, "kind": kind,
	})
}

// ── outbound commands ─────────────────────────────────────────────

func (b *bridge) quoteContext(replyTo string) *waE2E.ContextInfo {
	if replyTo == "" {
		return nil
	}
	b.memMu.Lock()
	cached := b.msgCache[replyTo]
	b.memMu.Unlock()
	if cached == nil {
		return nil
	}
	return &waE2E.ContextInfo{
		StanzaID:      proto.String(replyTo),
		Participant:   proto.String(cached.sender.String()),
		QuotedMessage: cached.msg,
	}
}

// ── authored keyboards (buttons completion, 2026-11-08) ───────────
// One keyboard contract serves text sends, media sends and edits,
// on BOTH bridges (the Baileys bridge builds the same shapes):
//
//   * rows of (label, command) pairs flatten in authored order;
//   * the quick_reply form is an InteractiveMessage carrying a
//     NativeFlow button set — at most three native buttons, the
//     overflow rendered as "• label — /cmd" body lines, so no
//     authored action is ever silently dropped;
//   * the list form (buttons_style == "list") is a ListMessage —
//     sections of ten rows, row ids carrying the command, so taps
//     return through the same ListResponseMessage mapping as
//     everything else;
//   * a refused interactive send degrades the WHOLE keyboard to
//     text lines, and the reply's buttons_form names what actually
//     went out ("interactive" / "list" / "text") so the Python
//     side journals the degradation instead of passing in silence.

type waButton struct{ label, cmd string }

func parseButtons(req request) []waButton {
	var flat []waButton
	if rows, ok := req["buttons"].([]any); ok {
		for _, rowAny := range rows {
			row, _ := rowAny.([]any)
			for _, bAny := range row {
				pair, _ := bAny.([]any)
				if len(pair) >= 2 {
					label, _ := pair[0].(string)
					cmd, _ := pair[1].(string)
					flat = append(flat, waButton{label, cmd})
				}
			}
		}
	}
	return flat
}

func buttonLines(items []waButton) string {
	parts := make([]string, 0, len(items))
	for _, it := range items {
		parts = append(parts, "• "+it.label+" — /"+it.cmd)
	}
	return strings.Join(parts, "\n")
}

// splitNative caps the native button set at three (the quick_reply
// ceiling current clients render) and folds the overflow into the
// body as text lines.
func splitNative(flat []waButton, body string) ([]waButton, string) {
	if len(flat) <= 3 {
		return flat, body
	}
	if body != "" {
		body += "\n"
	}
	return flat[:3], body + buttonLines(flat[3:])
}

// interactiveMessage builds the NativeFlow quick_reply form. A
// header (media sends) attaches the uploaded media so the photo /
// video / document itself carries the buttons.
func interactiveMessage(body string, native []waButton,
	quote *waE2E.ContextInfo,
	header *waE2E.InteractiveMessage_Header) *waE2E.Message {
	flowButtons := make(
		[]*waE2E.InteractiveMessage_NativeFlowMessage_NativeFlowButton,
		0, len(native))
	for _, it := range native {
		label := it.label
		if runes := []rune(label); len(runes) > 25 {
			label = string(runes[:25])
		}
		params, _ := json.Marshal(map[string]string{
			"id": it.cmd, "display_text": label})
		flowButtons = append(flowButtons,
			&waE2E.InteractiveMessage_NativeFlowMessage_NativeFlowButton{
				Name:             proto.String("quick_reply"),
				ButtonParamsJSON: proto.String(string(params)),
			})
	}
	return &waE2E.Message{InteractiveMessage: &waE2E.InteractiveMessage{
		Header: header,
		Body: &waE2E.InteractiveMessage_Body{
			Text: proto.String(body)},
		InteractiveMessage: &waE2E.InteractiveMessage_NativeFlowMessage_{
			NativeFlowMessage: &waE2E.InteractiveMessage_NativeFlowMessage{
				Buttons:        flowButtons,
				MessageVersion: proto.Int32(3),
			},
		},
		ContextInfo: quote,
	}}
}

// buildListMessage builds the LIST form of a keyboard: a
// ListMessage whose rows carry the commands as row ids. Row
// titles truncate at 24 runes (the client's row title budget);
// the body text rides as the list description.
func buildListMessage(body string, flat []waButton,
	quote *waE2E.ContextInfo) *waE2E.Message {
	var sections []*waE2E.ListMessage_Section
	for start := 0; start < len(flat); start += 10 {
		end := start + 10
		if end > len(flat) {
			end = len(flat)
		}
		rows := make([]*waE2E.ListMessage_Row, 0, end-start)
		for _, it := range flat[start:end] {
			title := it.label
			if runes := []rune(title); len(runes) > 24 {
				title = string(runes[:24])
			}
			rows = append(rows, &waE2E.ListMessage_Row{
				RowID: proto.String(it.cmd),
				Title: proto.String(title),
			})
		}
		sections = append(sections, &waE2E.ListMessage_Section{
			Title: proto.String("Actions"),
			Rows:  rows,
		})
	}
	return &waE2E.Message{ListMessage: &waE2E.ListMessage{
		Title:       proto.String("Actions"),
		Description: proto.String(body),
		ButtonText:  proto.String("Choose"),
		ListType:    waE2E.ListMessage_SINGLE_SELECT.Enum(),
		Sections:    sections,
		ContextInfo: quote,
	}}
}

// buttonsFormName names the form a (style, keyboard) pair went out
// in, for reply payloads.
func buttonsFormName(style string, flat []waButton) string {
	if len(flat) == 0 {
		return ""
	}
	if style == "list" {
		return "list"
	}
	return "interactive"
}

func (b *bridge) cmdSend(ctx context.Context, cc *clientConn, id string, req request) {
	chat, err := parseJID(req.str("chat"))
	if err != nil {
		b.fail(cc, id, err)
		return
	}
	text := req.str("text")
	replyTo := req.str("reply_to")
	quote := b.quoteContext(replyTo)

	flat := parseButtons(req)
	send := func(msg *waE2E.Message) (types.MessageID, error) {
		resp, err := b.cli.SendMessage(ctx, chat, msg)
		if err != nil {
			return "", err
		}
		// Remember our own sends: the edit command (progressive
		// display) may only target messages this bridge sent,
		// and chat history should show them too. whatsmeow does
		// not echo a client's own SendMessage back as an event,
		// so without this the send is invisible to both.
		rec := &remembered{
			incoming: false, text: textOf(msg),
			id: resp.ID, chatJID: chat, tsMs: time.Now().UnixMilli(),
		}
		if b.cli.Store != nil && b.cli.Store.ID != nil {
			rec.senderJID = b.cli.Store.ID.ToNonAD()
		}
		b.remember(chat, rec)
		// Cache the payload too: forwarding (cmdForward) needs
		// the original message, and the owner's own sends are
		// exactly the messages he forwards most.
		b.cacheMessage(string(resp.ID), msg, rec.senderJID)
		return resp.ID, nil
	}

	if len(flat) > 0 {
		// The legacy ButtonsMessage is dead on current WhatsApp
		// clients for unofficial multi-device senders: WhatsApp
		// deprecated the form and clients simply do not render it,
		// so button-carrying messages landed as bare text with no
		// error anywhere (the owner's "WhatsApp has no buttons"
		// report, 2026-11-07).  The form clients render today is
		// an InteractiveMessage carrying a NativeFlow quick_reply
		// button set; a tap returns as an
		// InteractiveResponseMessage whose params JSON carries
		// the button id — textOf already maps it back to "/<id>",
		// so one authored keyboard works before and after.  The
		// adapter may ask for the LIST form instead
		// (buttons_style == "list") when this account's probe /
		// degradation evidence says quick_reply does not render
		// on its client — same commands, ListMessage shape.
		degrade := func() {
			// Every action degrades to a text line; none are
			// silently dropped, and the reply NAMES the degraded
			// form so the Python side journals it.
			full := text
			if full != "" {
				full += "\n"
			}
			full += buttonLines(flat)
			mid, err := send(b.textMessage(full, quote))
			if err != nil {
				b.fail(cc, id, err)
				return
			}
			b.reply(cc, id, map[string]any{
				"ok": true, "message_id": string(mid),
				"buttons_form": "text"})
		}
		if req.str("buttons_style") == "list" {
			if mid, err := send(buildListMessage(text, flat, quote)); err == nil {
				b.reply(cc, id, map[string]any{
					"ok": true, "message_id": string(mid),
					"buttons_form": "list"})
				return
			}
			degrade()
			return
		}
		native, body := splitNative(flat, text)
		if mid, err := send(interactiveMessage(body, native, quote, nil)); err == nil {
			b.reply(cc, id, map[string]any{
				"ok": true, "message_id": string(mid),
				"buttons_form": "interactive"})
			return
		}
		degrade()
		return
	}

	// Mentions (chat & privacy wave — /tagall): JID strings the
	// text must notify without necessarily naming.  They ride
	// the extended text's context info, merged over any quote.
	var mentionJIDs []string
	if raw, ok := req["mentions"].([]any); ok {
		for _, m := range raw {
			if s, ok := m.(string); ok && strings.TrimSpace(s) != "" {
				mentionJIDs = append(mentionJIDs, s)
			}
		}
	}
	var plain *waE2E.Message
	if len(mentionJIDs) > 0 {
		ci := &waE2E.ContextInfo{}
		if quote != nil {
			ci = quote
		}
		ci.MentionedJID = mentionJIDs
		plain = &waE2E.Message{ExtendedTextMessage: &waE2E.ExtendedTextMessage{
			Text: proto.String(text), ContextInfo: ci}}
	} else {
		plain = b.textMessage(text, quote)
	}
	mid, err := send(plain)
	if err != nil {
		b.fail(cc, id, err)
		return
	}
	b.reply(cc, id, map[string]any{"ok": true, "message_id": string(mid)})
}

func (b *bridge) textMessage(text string, quote *waE2E.ContextInfo) *waE2E.Message {
	if quote != nil {
		return &waE2E.Message{ExtendedTextMessage: &waE2E.ExtendedTextMessage{
			Text:        proto.String(text),
			ContextInfo: quote,
		}}
	}
	return &waE2E.Message{Conversation: proto.String(text)}
}

// cmdEdit edits one of this bridge's OWN already-sent text
// messages in place (progressive display, register #31):
// WhatsApp message edits are a protocol message wrapping the
// replacement body (whatsmeow BuildEdit wraps ANY message body —
// the wrap is content-agnostic, so since the buttons completion
// the replacement may itself be an interactive/list message, or a
// plain body when the keyboard is being CLEARED). The protocol
// only allows editing your own messages, and only within
// WhatsApp's edit window (~15 minutes) — both are enforced
// server-side and any refusal comes back as the send error,
// surfaced verbatim. The bridge additionally refuses a target it
// never sent (not in its own remembered history for the chat), so
// a stale or foreign id fails fast with a named reason instead of
// a protocol round-trip.
//
// The buttons field is PRESENCE-significant: absent means the
// original text-only edit; present (even as an empty list) means
// "the settled render is this text with this keyboard" — an empty
// list clears the keyboard the earlier render carried.
func (b *bridge) cmdEdit(ctx context.Context, cc *clientConn, id string, req request) {
	chat, err := parseJID(req.str("chat"))
	if err != nil {
		b.fail(cc, id, err)
		return
	}
	mid := strings.TrimSpace(req.str("message_id"))
	if mid == "" {
		b.reply(cc, id, map[string]any{"ok": false, "error": "edit needs a message_id"})
		return
	}
	text := req.str("text")
	key := chat.String()
	b.memMu.Lock()
	var target *remembered
	for _, rec := range b.recent[key] {
		if rec != nil && !rec.incoming && string(rec.id) == mid {
			target = rec
			break
		}
	}
	b.memMu.Unlock()
	if target == nil {
		b.reply(cc, id, map[string]any{
			"ok": false,
			"error": "edit target " + mid + " is not one of this bridge's own " +
				"messages in that chat (only own messages can be edited)",
		})
		return
	}
	if _, hasButtons := req["buttons"]; hasButtons {
		flat := parseButtons(req)
		style := req.str("buttons_style")
		var content *waE2E.Message
		switch {
		case len(flat) == 0:
			// Keyboard cleared: the settled render is plain text.
			content = &waE2E.Message{Conversation: proto.String(text)}
		case style == "list":
			content = buildListMessage(text, flat, nil)
		default:
			native, body := splitNative(flat, text)
			content = interactiveMessage(body, native, nil, nil)
		}
		edit := b.cli.BuildEdit(chat, types.MessageID(mid), content)
		if _, err := b.cli.SendMessage(ctx, chat, edit); err != nil {
			b.fail(cc, id, err)
			return
		}
		b.memMu.Lock()
		target.text = text
		b.memMu.Unlock()
		reply := map[string]any{"ok": true, "message_id": mid}
		if form := buttonsFormName(style, flat); form != "" {
			reply["buttons_form"] = form
		}
		b.reply(cc, id, reply)
		return
	}
	edit := b.cli.BuildEdit(chat, types.MessageID(mid),
		&waE2E.Message{Conversation: proto.String(text)})
	if _, err := b.cli.SendMessage(ctx, chat, edit); err != nil {
		b.fail(cc, id, err)
		return
	}
	b.memMu.Lock()
	target.text = text
	b.memMu.Unlock()
	b.reply(cc, id, map[string]any{"ok": true, "message_id": mid})
}

// specialFormRefusal names why a requested send form cannot
// ride this media type, or "" when it can (parity bundle #10).
// The check runs before any upload: a form the media cannot
// carry is refused by name, never silently sent plain.
func specialFormRefusal(viewOnce, ptv bool,
	mediaType whatsmeow.MediaType) string {
	if viewOnce && mediaType != whatsmeow.MediaImage &&
		mediaType != whatsmeow.MediaVideo {
		return "view-once applies to photos and videos only"
	}
	if ptv && mediaType != whatsmeow.MediaVideo {
		return "a video note (ptv) needs a video file"
	}
	return ""
}

// applyViewOnce sets the view-once flag on the media message
// itself (bundle #10) — the flag form this bridge's
// unwrapInbound recognizes on arrival, and the form current
// clients send.
func applyViewOnce(msg *waE2E.Message) {
	if msg.ImageMessage != nil {
		msg.ImageMessage.ViewOnce = proto.Bool(true)
	}
	if msg.VideoMessage != nil {
		msg.VideoMessage.ViewOnce = proto.Bool(true)
	}
}

// asVideoNote re-forms a built video message as a ROUND video
// note (bundle #10): the VideoMessage rides as the message's
// PtvMessage. No client renders a caption on a video note, so
// the caption is lifted out of the proto — the caller sends it
// as a follow-up text message (degraded, never dropped).
func asVideoNote(msg *waE2E.Message) *waE2E.Message {
	msg.VideoMessage.Caption = nil
	return &waE2E.Message{PtvMessage: msg.VideoMessage}
}

var extMIME = map[string]string{
	".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
	".webp": "image/webp", ".gif": "image/gif", ".mp4": "video/mp4",
	".mov": "video/quicktime", ".mp3": "audio/mpeg", ".ogg": "audio/ogg",
	".oga": "audio/ogg", ".wav": "audio/wav", ".m4a": "audio/mp4",
	".pdf": "application/pdf",
}

func (b *bridge) cmdSendMedia(ctx context.Context, cc *clientConn, id string, req request) {
	chat, err := parseJID(req.str("chat"))
	if err != nil {
		b.fail(cc, id, err)
		return
	}
	p := req.str("path")
	data, err := os.ReadFile(p)
	if err != nil {
		b.reply(cc, id, map[string]any{"ok": false, "error": "media file not found: " + p})
		return
	}
	ext := strings.ToLower(filepath.Ext(p))
	mime := extMIME[ext]
	caption := req.str("caption")
	ptt := req.boolean("ptt")
	// Special forms (parity bundle #10): view_once wraps a
	// photo/video so the recipient gets one look (the flag form
	// on the media message — the same shape this bridge's
	// unwrapInbound recognizes on arrival); ptv sends a video
	// as a ROUND video note (the protocol's PtvMessage form).
	// Both are validated against the media type BEFORE any
	// upload: a form the media cannot carry is an honest
	// refusal, never a silent plain send.
	viewOnce := req.boolean("view_once")
	ptv := req.boolean("ptv")

	var mediaType whatsmeow.MediaType
	switch {
	case ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".webp":
		mediaType = whatsmeow.MediaImage
	case ext == ".mp4" || ext == ".mov" || ext == ".gif":
		mediaType = whatsmeow.MediaVideo
	case ext == ".mp3" || ext == ".ogg" || ext == ".oga" || ext == ".wav" || ext == ".m4a":
		mediaType = whatsmeow.MediaAudio
	default:
		mediaType = whatsmeow.MediaDocument
	}
	if refusal := specialFormRefusal(viewOnce, ptv, mediaType); refusal != "" {
		b.reply(cc, id, map[string]any{"ok": false, "error": refusal})
		return
	}
	var up whatsmeow.UploadResponse
	if chat.Server == types.NewsletterServer {
		up, err = b.cli.UploadNewsletter(ctx, data, mediaType)
	} else {
		up, err = b.cli.Upload(ctx, data, mediaType)
	}
	if err != nil {
		b.fail(cc, id, err)
		return
	}
	var msg *waE2E.Message
	switch mediaType {
	case whatsmeow.MediaImage:
		msg = &waE2E.Message{ImageMessage: &waE2E.ImageMessage{
			URL: proto.String(up.URL), DirectPath: proto.String(up.DirectPath),
			MediaKey: up.MediaKey, Mimetype: proto.String(mime),
			Caption:    proto.String(caption),
			FileSHA256: up.FileSHA256, FileEncSHA256: up.FileEncSHA256,
			FileLength: proto.Uint64(up.FileLength),
		}}
	case whatsmeow.MediaVideo:
		msg = &waE2E.Message{VideoMessage: &waE2E.VideoMessage{
			URL: proto.String(up.URL), DirectPath: proto.String(up.DirectPath),
			MediaKey: up.MediaKey, Mimetype: proto.String(mime),
			Caption:    proto.String(caption),
			FileSHA256: up.FileSHA256, FileEncSHA256: up.FileEncSHA256,
			FileLength: proto.Uint64(up.FileLength),
		}}
	case whatsmeow.MediaAudio:
		if mime == "" {
			mime = "audio/ogg"
		}
		msg = &waE2E.Message{AudioMessage: &waE2E.AudioMessage{
			URL: proto.String(up.URL), DirectPath: proto.String(up.DirectPath),
			MediaKey: up.MediaKey, Mimetype: proto.String(mime),
			FileSHA256: up.FileSHA256, FileEncSHA256: up.FileEncSHA256,
			FileLength: proto.Uint64(up.FileLength),
			PTT:        proto.Bool(ptt),
		}}
	default:
		if mime == "" {
			mime = "application/octet-stream"
		}
		msg = &waE2E.Message{DocumentMessage: &waE2E.DocumentMessage{
			URL: proto.String(up.URL), DirectPath: proto.String(up.DirectPath),
			MediaKey: up.MediaKey, Mimetype: proto.String(mime),
			FileName:   proto.String(filepath.Base(p)),
			Caption:    proto.String(caption),
			FileSHA256: up.FileSHA256, FileEncSHA256: up.FileEncSHA256,
			FileLength: proto.Uint64(up.FileLength),
		}}
	}
	if viewOnce {
		applyViewOnce(msg)
	}
	// Buttons (buttons completion, 2026-11-08): an image / video /
	// document heads an InteractiveMessage whose Header carries
	// the uploaded media — the photo itself gets the buttons, in
	// the same NativeFlow form as text sends (3 native, overflow
	// as caption lines). Audio has no interactive header form, so
	// its actions ride as a follow-up text message and the form
	// reports "text" — degraded, never dropped. A refused
	// interactive send likewise falls back to the plain media with
	// every action as a caption line, buttons_form "text".
	flat := parseButtons(req)
	rememberMedia := func(sent *waE2E.Message, mid types.MessageID) {
		// Cache the payload like cmdSend does — the owner's own
		// media sends are forwardable material (cmdForward).
		if b.cli.Store != nil && b.cli.Store.ID != nil {
			b.cacheMessage(string(mid), sent,
				b.cli.Store.ID.ToNonAD())
		}
	}
	// A view-once or video-note send never takes the interactive
	// header path (bundle #10): the wrap would replace the very
	// form the caller asked for. Its buttons degrade to the
	// follow-up text lines below, like audio's.
	if len(flat) > 0 && mediaType != whatsmeow.MediaAudio &&
		!viewOnce && !ptv {
		var header *waE2E.InteractiveMessage_Header
		switch m := msg; {
		case m.ImageMessage != nil:
			header = &waE2E.InteractiveMessage_Header{
				HasMediaAttachment: proto.Bool(true),
				Media: &waE2E.InteractiveMessage_Header_ImageMessage{
					ImageMessage: m.ImageMessage}}
		case m.VideoMessage != nil:
			header = &waE2E.InteractiveMessage_Header{
				HasMediaAttachment: proto.Bool(true),
				Media: &waE2E.InteractiveMessage_Header_VideoMessage{
					VideoMessage: m.VideoMessage}}
		case m.DocumentMessage != nil:
			header = &waE2E.InteractiveMessage_Header{
				HasMediaAttachment: proto.Bool(true),
				Media: &waE2E.InteractiveMessage_Header_DocumentMessage{
					DocumentMessage: m.DocumentMessage}}
		}
		if header != nil {
			native, body := splitNative(flat, caption)
			wrapped := interactiveMessage(body, native, nil, header)
			if resp, err := b.cli.SendMessage(ctx, chat, wrapped); err == nil {
				rememberMedia(wrapped, resp.ID)
				b.reply(cc, id, map[string]any{
					"ok": true, "message_id": string(resp.ID),
					"buttons_form": "interactive"})
				return
			}
			// Refused: plain media, actions as caption lines.
			full := caption
			if full != "" {
				full += "\n"
			}
			full += buttonLines(flat)
			switch m := msg; {
			case m.ImageMessage != nil:
				m.ImageMessage.Caption = proto.String(full)
			case m.VideoMessage != nil:
				m.VideoMessage.Caption = proto.String(full)
			case m.DocumentMessage != nil:
				m.DocumentMessage.Caption = proto.String(full)
			}
			resp, err := b.cli.SendMessage(ctx, chat, msg)
			if err != nil {
				b.fail(cc, id, err)
				return
			}
			rememberMedia(msg, resp.ID)
			b.reply(cc, id, map[string]any{
				"ok": true, "message_id": string(resp.ID),
				"buttons_form": "text"})
			return
		}
	}
	out := msg
	if ptv {
		out = asVideoNote(msg)
	}
	resp, err := b.cli.SendMessage(ctx, chat, out)
	if err != nil {
		b.fail(cc, id, err)
		return
	}
	rememberMedia(out, resp.ID)
	follow := ""
	if ptv && caption != "" {
		follow = caption
	}
	if len(flat) > 0 {
		if follow != "" {
			follow += "\n"
		}
		follow += buttonLines(flat)
	}
	if follow != "" {
		// Audio (or a special form): the media is out; its
		// caption/actions follow as text in the same command.
		if _, err := b.cli.SendMessage(ctx, chat,
			b.textMessage(follow, nil)); err != nil {
			b.fail(cc, id, err)
			return
		}
		if len(flat) > 0 {
			b.reply(cc, id, map[string]any{
				"ok": true, "message_id": string(resp.ID),
				"buttons_form": "text"})
			return
		}
	}
	b.reply(cc, id, map[string]any{"ok": true, "message_id": string(resp.ID)})
}

func (b *bridge) cmdTyping(ctx context.Context, cc *clientConn, id string, req request) {
	chat, err := parseJID(req.str("chat"))
	if err != nil {
		b.fail(cc, id, err)
		return
	}
	secs := req.num("seconds", 3)
	if secs < 1 {
		secs = 1
	}
	if secs > 30 {
		secs = 30
	}
	media := types.ChatPresenceMediaText
	if req.str("presence") == "recording" {
		media = types.ChatPresenceMediaAudio
	}
	if err := b.cli.SendChatPresence(ctx, chat, types.ChatPresenceComposing, media); err != nil {
		b.fail(cc, id, err)
		return
	}
	go func() {
		time.Sleep(time.Duration(secs) * time.Second)
		pctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = b.cli.SendChatPresence(pctx, chat, types.ChatPresencePaused, media)
	}()
	b.reply(cc, id, map[string]any{"ok": true})
}

// rememberedEntry is the wire shape of one remembered
// message — the row `history` and `search` both answer with.
func rememberedEntry(m *remembered) map[string]any {
	media := m.media
	if media == nil {
		media = []map[string]any{}
	}
	ts := m.tsMs
	if ts == 0 {
		ts = time.Now().UnixMilli()
	}
	return map[string]any{
		"id":       string(m.id),
		"incoming": m.incoming, "text": m.text, "sender": m.sender,
		"media": media, "reply_to": m.replyTo, "ts": ts,
	}
}

func (b *bridge) cmdHistory(cc *clientConn, id string, req request) {
	chat, err := parseJID(req.str("chat"))
	if err != nil {
		b.fail(cc, id, err)
		return
	}
	limit := req.num("limit", 20)
	if limit < 1 {
		limit = 1
	}
	b.memMu.Lock()
	arr := append([]*remembered{}, b.recent[chat.String()]...)
	b.memMu.Unlock()
	if len(arr) > limit {
		arr = arr[len(arr)-limit:]
	}
	msgs := make([]map[string]any, 0, len(arr))
	for _, m := range arr {
		msgs = append(msgs, rememberedEntry(m))
	}
	b.reply(cc, id, map[string]any{"ok": true, "messages": msgs})
}

// searchRemembered filters one chat's remembered messages
// by a case-insensitive substring over the message text
// (parity bundle #11).  It returns the most recent `limit`
// matches in remembered (oldest-first) order plus the
// TOTAL number of matches found, so the caller can say
// "shown X of Y" instead of silently capping.  Pure, so
// search_test.go pins it without a client.
func searchRemembered(arr []*remembered, query string,
	limit int) ([]*remembered, int) {
	needle := strings.ToLower(strings.TrimSpace(query))
	if needle == "" || limit < 1 {
		return nil, 0
	}
	var matches []*remembered
	for _, m := range arr {
		if m == nil {
			continue
		}
		if strings.Contains(strings.ToLower(m.text), needle) {
			matches = append(matches, m)
		}
	}
	total := len(matches)
	if total > limit {
		matches = matches[total-limit:]
	}
	return matches, total
}

// cmdSearch is the in-chat half of the search-depth bundle:
// a filter over the bridge's OWN remembered messages for
// the chat.  WhatsApp exposes no server-side search to a
// linked device (whatsmeow has no search verb), so the
// remembered history — the same store `history` serves —
// is the honest reach, and the adapter's coverage line
// says so.  The query matches message TEXT (captions
// included: they ride the same field); a query with no
// words refuses in words, mirroring the Python shape rule.
func (b *bridge) cmdSearch(cc *clientConn, id string, req request) {
	chat, err := parseJID(req.str("chat"))
	if err != nil {
		b.fail(cc, id, err)
		return
	}
	if strings.TrimSpace(req.str("query")) == "" {
		b.reply(cc, id, map[string]any{
			"ok":    false,
			"error": "a search needs the words to look for"})
		return
	}
	limit := req.num("limit", 20)
	if limit < 1 {
		limit = 1
	}
	if limit > maxKeepPerChat {
		limit = maxKeepPerChat
	}
	b.memMu.Lock()
	arr := append([]*remembered{}, b.recent[chat.String()]...)
	b.memMu.Unlock()
	matches, total := searchRemembered(arr, req.str("query"),
		limit)
	msgs := make([]map[string]any, 0, len(matches))
	for _, m := range matches {
		msgs = append(msgs, rememberedEntry(m))
	}
	b.reply(cc, id, map[string]any{
		"ok": true, "messages": msgs, "total": total})
}

func (b *bridge) cmdRead(ctx context.Context, cc *clientConn, id string, req request) {
	chat, err := parseJID(req.str("chat"))
	if err != nil {
		b.fail(cc, id, err)
		return
	}
	b.memMu.Lock()
	arr := append([]*remembered{}, b.recent[chat.String()]...)
	b.memMu.Unlock()
	for i := len(arr) - 1; i >= 0; i-- {
		if arr[i].incoming && arr[i].id != "" {
			// best-effort, like the old bridge: the ack is what
			// the adapter checks
			_ = b.cli.MarkRead(ctx, []types.MessageID{arr[i].id},
				time.UnixMilli(arr[i].tsMs), arr[i].chatJID, arr[i].senderJID)
			break
		}
	}
	b.reply(cc, id, map[string]any{"ok": true})
}

// cmdPresence asserts the account's global presence (§15
// always-online): whatsmeow SendPresence with available /
// unavailable. One-shot — the Python side re-sends on its
// housekeeping beat while the owner wants the account online.
func (b *bridge) cmdPresence(ctx context.Context, cc *clientConn, id string, req request) {
	state := types.PresenceAvailable
	want := "available"
	if req.str("state") == "unavailable" {
		state = types.PresenceUnavailable
		want = "unavailable"
	}
	if err := b.cli.SendPresence(ctx, state); err != nil {
		b.fail(cc, id, err)
		return
	}
	// Remember the want: onConnected re-asserts it after every
	// reconnect, so a socket flap never silently drops the
	// account's presence until some later beat notices.
	b.presenceMu.Lock()
	b.presenceWant = want
	b.presenceMu.Unlock()
	b.reply(cc, id, map[string]any{"ok": true})
}

// cmdStatusSend posts a status AS the account owner (§15
// status posting — the send half of the status family): a text
// status (an ExtendedTextMessage with a background, the shape
// the phone itself sends) or an image/video status with a
// caption, addressed to status@broadcast. whatsmeow resolves
// the audience itself — SendMessage to StatusBroadcastJID fans
// out to the account's server-side status recipients — so no
// recipient list rides the wire and nobody outside his status
// privacy list can be reached. Refusals (not logged in, a dead
// file, an upload failure) come back verbatim, never a fake
// post.
func (b *bridge) cmdStatusSend(ctx context.Context, cc *clientConn, id string, req request) {
	text := strings.TrimSpace(req.str("text"))
	path := strings.TrimSpace(req.str("path"))
	caption := req.str("caption")
	if text == "" && path == "" {
		b.reply(cc, id, map[string]any{"ok": false,
			"error": "status_send needs text or a media path"})
		return
	}
	var msg *waE2E.Message
	if path != "" {
		data, err := os.ReadFile(path)
		if err != nil {
			b.reply(cc, id, map[string]any{"ok": false,
				"error": "media file not found: " + path})
			return
		}
		ext := strings.ToLower(filepath.Ext(path))
		var mediaType whatsmeow.MediaType
		switch ext {
		case ".jpg", ".jpeg", ".png", ".webp":
			mediaType = whatsmeow.MediaImage
		case ".mp4", ".mov", ".gif":
			mediaType = whatsmeow.MediaVideo
		default:
			b.reply(cc, id, map[string]any{"ok": false,
				"error": "status media must be an image or a " +
					"video: " + path})
			return
		}
		up, err := b.cli.Upload(ctx, data, mediaType)
		if err != nil {
			b.fail(cc, id, err)
			return
		}
		mime := extMIME[ext]
		if caption == "" {
			caption = text
		}
		if mediaType == whatsmeow.MediaImage {
			msg = &waE2E.Message{ImageMessage: &waE2E.ImageMessage{
				URL:        proto.String(up.URL),
				DirectPath: proto.String(up.DirectPath),
				MediaKey:   up.MediaKey, Mimetype: proto.String(mime),
				Caption:       proto.String(caption),
				FileSHA256:    up.FileSHA256,
				FileEncSHA256: up.FileEncSHA256,
				FileLength:    proto.Uint64(up.FileLength),
			}}
		} else {
			msg = &waE2E.Message{VideoMessage: &waE2E.VideoMessage{
				URL:        proto.String(up.URL),
				DirectPath: proto.String(up.DirectPath),
				MediaKey:   up.MediaKey, Mimetype: proto.String(mime),
				Caption:       proto.String(caption),
				FileSHA256:    up.FileSHA256,
				FileEncSHA256: up.FileEncSHA256,
				FileLength:    proto.Uint64(up.FileLength),
			}}
		}
	} else {
		msg = &waE2E.Message{ExtendedTextMessage: &waE2E.ExtendedTextMessage{
			Text:           proto.String(text),
			BackgroundArgb: proto.Uint32(0xFF000000),
		}}
	}
	resp, err := b.cli.SendMessage(ctx, types.StatusBroadcastJID,
		msg)
	if err != nil {
		b.fail(cc, id, err)
		return
	}
	b.reply(cc, id, map[string]any{"ok": true,
		"message_id": string(resp.ID)})
}

// statusReceiptMode names what a read receipt under the
// account's privacy settings actually does when it leaves this
// client (status mining, 2026-11-07): whatsmeow's MarkRead
// silently rewrites type="read" to type="read-self" when the
// account's Read-receipts privacy is off (the pin's
// receipt.go), so the verb returns ok while the poster is
// never notified.  "read" = the poster is notified;
// "read-self" = receipts off, the view cannot register with
// the poster; "unknown" = the setting could not be read
// (GetPrivacySettings returns an empty struct when its fetch
// fails) — reported, never guessed.
func statusReceiptMode(settings types.PrivacySettings) string {
	switch settings.ReadReceipts {
	case types.PrivacySettingNone:
		return "read-self"
	case types.PrivacySettingAll:
		return "read"
	default:
		return "unknown"
	}
}

// statusViewClient is the slice of *whatsmeow.Client the
// status-view verb rides — an interface so the verb's call
// SHAPE is pinnable in tests.  (The truth pass's recording
// fake accepted any well-shaped call; the bridge layer itself
// is where the mined divergences lived.)
type statusViewClient interface {
	MarkRead(ctx context.Context, ids []types.MessageID,
		timestamp time.Time, chat, sender types.JID,
		receiptTypeExtra ...types.ReceiptType) error
	GetPrivacySettings(ctx context.Context) types.PrivacySettings
}

var _ statusViewClient = (*whatsmeow.Client)(nil)

// markStatusViewed fires the read receipt for one status post
// and reports the receipt mode in force.  The receipt's
// timestamp is the READ time (now, like the phone) — plumbing
// the post's own timestamp into MarkRead made the receipt
// claim the owner viewed the post when it was posted.
func markStatusViewed(ctx context.Context, cli statusViewClient,
	sender types.JID, msgID types.MessageID) (string, error) {
	if err := cli.MarkRead(ctx, []types.MessageID{msgID},
		time.Now(), types.StatusBroadcastJID, sender); err != nil {
		return "", err
	}
	return statusReceiptMode(cli.GetPrivacySettings(ctx)), nil
}

// cmdStatusView marks one contact status as viewed by this
// account (§15 status viewing): a read receipt for the status
// post — MarkRead addressed to status@broadcast with the poster
// as sender, the same receipt the phone sends when the owner
// opens the status.  The reply carries receipt_mode: when the
// account's read receipts are off the receipt leaves as
// read-self and the poster is never notified, so an ok without
// the mode would be a silent no-op wearing a success's clothes.
// The request's ts (the post's timestamp, sent by older
// adapters) is deliberately unused — the receipt is stamped at
// read time inside markStatusViewed.
func (b *bridge) cmdStatusView(ctx context.Context, cc *clientConn, id string, req request) {
	// The target rides "message_id": the request's "id" is the
	// correlation id (the adapter's _send_cmd writes it over
	// any "id" a caller set), so reading the target from "id"
	// aimed every view at a random request id.
	msgID := strings.TrimSpace(req.str("message_id"))
	sender, err := parseJID(req.str("from"))
	if err != nil || msgID == "" {
		b.reply(cc, id, map[string]any{"ok": false,
			"error": "status_view needs the status message id and the poster's jid"})
		return
	}
	mode, err := markStatusViewed(ctx, b.cli, sender,
		types.MessageID(msgID))
	if err != nil {
		b.fail(cc, id, err)
		return
	}
	b.reply(cc, id, map[string]any{"ok": true,
		"receipt_mode": mode})
}

// ── chat & privacy wave (FEI gap wave F) ──────────────────────
// Owner-invoked ops on his own account.  Each maps to the
// whatsmeow verb that actually exists for it; a refusal comes
// back as the send/app-state error, verbatim.

// lastRemembered returns the newest remembered message in a
// chat (zero values when the bridge has seen none) — archive /
// mark-read app-state patches carry the last message's range
// when it's known and accept zero values when it isn't.
func (b *bridge) lastRemembered(chat types.JID) (time.Time, bool) {
	b.memMu.Lock()
	defer b.memMu.Unlock()
	arr := b.recent[chat.String()]
	if len(arr) == 0 {
		return time.Time{}, false
	}
	last := arr[len(arr)-1]
	if last.tsMs <= 0 {
		return time.Time{}, false
	}
	return time.UnixMilli(last.tsMs), true
}

// cmdChatModify applies chat-state changes — archive, pin,
// mute, mark read/unread — as whatsmeow app-state patches (the
// same patches the phone's own chat-list actions produce).
// Fields are tri-state: only the keys present are applied.
// mute_seconds: -1 = mute with no end (far future), 0 = unmute,
// >0 = mute for that many seconds.
func (b *bridge) cmdChatModify(ctx context.Context, cc *clientConn, id string, req request) {
	chat, err := parseJID(req.str("chat"))
	if err != nil {
		b.fail(cc, id, err)
		return
	}
	var patches []appstate.PatchInfo
	if req.has("archive") {
		lastTs, _ := b.lastRemembered(chat)
		patches = append(patches, appstate.BuildArchive(
			chat, req.boolean("archive"), lastTs, nil))
	}
	if req.has("pin") {
		patches = append(patches, appstate.BuildPin(
			chat, req.boolean("pin")))
	}
	if req.has("mute_seconds") {
		secs := req.num("mute_seconds", 0)
		switch {
		case secs == 0:
			patches = append(patches, appstate.BuildMute(
				chat, false, 0))
		case secs < 0:
			end := time.Now().AddDate(10, 0, 0).Unix()
			patches = append(patches, appstate.BuildMuteAbs(
				chat, true, &end))
		default:
			patches = append(patches, appstate.BuildMute(
				chat, true, time.Duration(secs)*time.Second))
		}
	}
	if req.has("read") {
		lastTs, _ := b.lastRemembered(chat)
		patches = append(patches, appstate.BuildMarkChatAsRead(
			chat, req.boolean("read"), lastTs, nil))
	}
	if len(patches) == 0 {
		b.reply(cc, id, map[string]any{"ok": false,
			"error": "chat_modify needs at least one of: " +
				"archive, pin, mute_seconds, read"})
		return
	}
	for _, patch := range patches {
		if err := b.cli.SendAppState(ctx, patch); err != nil {
			b.fail(cc, id, err)
			return
		}
	}
	b.reply(cc, id, map[string]any{"ok": true})
}

// cmdMessageStar stars/unstars one message: an app-state patch
// whose key needs the message's author, resolved from the
// bridge's own memory like message_pin — a message it never
// saw is an honest refusal, never a guessed key.
func (b *bridge) cmdMessageStar(ctx context.Context, cc *clientConn, id string, req request) {
	chat, err := parseJID(req.str("chat"))
	if err != nil {
		b.fail(cc, id, err)
		return
	}
	mid := strings.TrimSpace(req.str("message_id"))
	if mid == "" {
		b.reply(cc, id, map[string]any{"ok": false,
			"error": "message_star needs a message_id"})
		return
	}
	b.memMu.Lock()
	var target *remembered
	for _, rec := range b.recent[chat.String()] {
		if rec != nil && string(rec.id) == mid {
			target = rec
			break
		}
	}
	b.memMu.Unlock()
	if target == nil {
		b.reply(cc, id, map[string]any{"ok": false,
			"error": "message not in bridge memory — its author " +
				"is unknown, so its star key cannot be built"})
		return
	}
	patch := appstate.BuildStar(chat, target.senderJID,
		types.MessageID(mid), !target.incoming,
		req.boolean("star"))
	if err := b.cli.SendAppState(ctx, patch); err != nil {
		b.fail(cc, id, err)
		return
	}
	b.reply(cc, id, map[string]any{"ok": true})
}

// statusReactClient is the slice of *whatsmeow.Client the
// status-react verb rides (see statusViewClient for why).
type statusReactClient interface {
	BuildReaction(chat, sender types.JID, id types.MessageID,
		reaction string) *waE2E.Message
	SendMessage(ctx context.Context, to types.JID,
		message *waE2E.Message,
		extra ...whatsmeow.SendRequestExtra,
	) (whatsmeow.SendResponse, error)
}

var _ statusReactClient = (*whatsmeow.Client)(nil)

// sendStatusReaction delivers one status reaction to its
// poster.  The reaction's KEY points at the status post
// (BuildReaction over status@broadcast + poster + post id),
// but the SEND goes to the poster 1:1 — the peer shape the
// real protocol uses for status reactions.  A send ADDRESSED
// to status@broadcast is a broadcast: whatsmeow fans it out
// to the account's OWN status-privacy audience
// (getStatusBroadcastRecipients — the people allowed to see
// MY statuses), the poster may not even be in it, and
// SendRequestExtra offers no audience override (status
// mining, 2026-11-07).  The poster's client surfaces the
// peer-delivered reaction on their status, where status
// reactions and replies live.
func sendStatusReaction(ctx context.Context, cli statusReactClient,
	poster types.JID, msgID types.MessageID, emoji string) error {
	msg := cli.BuildReaction(types.StatusBroadcastJID, poster,
		msgID, emoji)
	_, err := cli.SendMessage(ctx, poster, msg)
	return err
}

// cmdStatusReact reacts to one contact status post as the
// owner (the status_react toggle's verb): a reaction keyed on
// the post, delivered to the poster — see sendStatusReaction
// for why the destination is the poster and not the
// broadcast.
func (b *bridge) cmdStatusReact(ctx context.Context, cc *clientConn, id string, req request) {
	// Target in "message_id", never "id" (the correlation id —
	// see cmdStatusView).
	msgID := strings.TrimSpace(req.str("message_id"))
	poster, err := parseJID(req.str("from"))
	emoji := strings.TrimSpace(req.str("emoji"))
	if err != nil || msgID == "" || emoji == "" {
		b.reply(cc, id, map[string]any{"ok": false,
			"error": "status_react needs the status message id, " +
				"the poster's jid and an emoji"})
		return
	}
	if err := sendStatusReaction(ctx, b.cli, poster,
		types.MessageID(msgID), emoji); err != nil {
		b.fail(cc, id, err)
		return
	}
	b.reply(cc, id, map[string]any{"ok": true})
}

// cmdReact reacts to one CHAT message as the owner (parity G1):
// whatsmeow BuildReaction against the message's key — chat +
// author + id.  The author comes from the bridge's own memory
// (the whole-message cache, then the chat's remembered ring —
// the same sources message_star reads); a message it never saw
// is an honest refusal, never a guessed key.  An empty emoji
// REMOVES the owner's reaction — the shape the phone itself
// sends.  DM and group targets both ride it, and the author
// may be addressed by phone JID or by LID: BuildMessageKey
// folds either form into the right key.
func (b *bridge) cmdReact(ctx context.Context, cc *clientConn, id string, req request) {
	chat, err := parseJID(req.str("chat"))
	if err != nil {
		b.fail(cc, id, err)
		return
	}
	mid := strings.TrimSpace(req.str("message_id"))
	if mid == "" {
		b.reply(cc, id, map[string]any{"ok": false,
			"error": "react needs a message_id"})
		return
	}
	emoji := strings.TrimSpace(req.str("emoji"))
	sender := b.cachedSender(mid)
	if sender.IsEmpty() {
		b.memMu.Lock()
		for _, rec := range b.recent[chat.String()] {
			if rec != nil && string(rec.id) == mid {
				sender = rec.senderJID
				break
			}
		}
		b.memMu.Unlock()
	}
	if sender.IsEmpty() {
		b.reply(cc, id, map[string]any{"ok": false,
			"error": "message not in bridge memory — its author " +
				"is unknown, so its reaction key cannot be built"})
		return
	}
	msg := b.cli.BuildReaction(chat, sender,
		types.MessageID(mid), emoji)
	if _, err := b.cli.SendMessage(ctx, chat, msg); err != nil {
		b.fail(cc, id, err)
		return
	}
	b.reply(cc, id, map[string]any{"ok": true})
}

// cmdForward forwards one message to another chat as the owner
// (parity G1).  WhatsApp has no forward RPC: a forward is the
// original message re-sent with its context marked forwarded
// (isForwarded + a forwarding score one past the original's) —
// the shape the phone itself sends.  The ORIGINAL payload must
// come from the bridge's own message memory (incoming messages
// are cached whole; this bridge's own sends are cached as they
// land) — a message it never saw is an honest refusal, never a
// retyped copy.  The quote linkage is stripped: a forward is
// not a reply.  Media rides its original upload — the keys
// travel with the message, as they do in the phone's forwards.
func (b *bridge) cmdForward(ctx context.Context, cc *clientConn, id string, req request) {
	if _, err := parseJID(req.str("chat")); err != nil {
		b.fail(cc, id, err)
		return
	}
	dest, err := parseJID(req.str("dest"))
	if err != nil {
		b.fail(cc, id, err)
		return
	}
	mid := strings.TrimSpace(req.str("message_id"))
	if mid == "" {
		b.reply(cc, id, map[string]any{"ok": false,
			"error": "forward needs a message_id"})
		return
	}
	b.memMu.Lock()
	cached := b.msgCache[mid]
	b.memMu.Unlock()
	if cached == nil || cached.msg == nil {
		b.reply(cc, id, map[string]any{"ok": false,
			"error": "message not in bridge memory — forwarding " +
				"needs the original message, and this bridge " +
				"never saw it"})
		return
	}
	msg, ferr := forwardCopy(cached.msg)
	if ferr != "" {
		b.reply(cc, id, map[string]any{"ok": false, "error": ferr})
		return
	}
	resp, err := b.cli.SendMessage(ctx, dest, msg)
	if err != nil {
		b.fail(cc, id, err)
		return
	}
	b.reply(cc, id, map[string]any{"ok": true,
		"message_id": string(resp.ID)})
}

// forwardCopy clones one cached message into its forward form.
// The second return is the honest refusal for shapes that have
// no forward: view-once wraps (forwarding one would break the
// promise the format makes) and non-content kinds (reactions,
// protocol, calls, polls, interactive forms).
func forwardCopy(src *waE2E.Message) (*waE2E.Message, string) {
	if src.GetViewOnceMessage() != nil ||
		src.GetViewOnceMessageV2() != nil ||
		src.GetViewOnceMessageV2Extension() != nil {
		return nil, "view-once messages can't be forwarded — " +
			"that is the promise the format makes"
	}
	inner := src
	if dsm := src.GetDeviceSentMessage(); dsm != nil &&
		dsm.GetMessage() != nil {
		inner = dsm.GetMessage()
	} else if eph := src.GetEphemeralMessage(); eph != nil &&
		eph.GetMessage() != nil {
		inner = eph.GetMessage()
	}
	out, ok := proto.Clone(inner).(*waE2E.Message)
	if !ok || out == nil {
		return nil, "the cached message could not be copied"
	}
	mark := func(ci *waE2E.ContextInfo) *waE2E.ContextInfo {
		if ci == nil {
			ci = &waE2E.ContextInfo{}
		}
		ci.IsForwarded = proto.Bool(true)
		ci.ForwardingScore = proto.Uint32(
			ci.GetForwardingScore() + 1)
		// a forward is not a reply, and the source chat's
		// disappearing-message state is not the destination's
		ci.StanzaID = nil
		ci.Participant = nil
		ci.QuotedMessage = nil
		ci.RemoteJID = nil
		ci.Expiration = nil
		ci.EphemeralSettingTimestamp = nil
		ci.EphemeralSharedSecret = nil
		ci.DisappearingMode = nil
		return ci
	}
	switch {
	case out.Conversation != nil:
		text := out.GetConversation()
		out.Conversation = nil
		out.ExtendedTextMessage = &waE2E.ExtendedTextMessage{
			Text: proto.String(text), ContextInfo: mark(nil)}
	case out.ExtendedTextMessage != nil:
		out.ExtendedTextMessage.ContextInfo = mark(
			out.ExtendedTextMessage.ContextInfo)
	case out.ImageMessage != nil:
		out.ImageMessage.ContextInfo = mark(
			out.ImageMessage.ContextInfo)
	case out.VideoMessage != nil:
		out.VideoMessage.ContextInfo = mark(
			out.VideoMessage.ContextInfo)
	case out.AudioMessage != nil:
		out.AudioMessage.ContextInfo = mark(
			out.AudioMessage.ContextInfo)
	case out.DocumentMessage != nil:
		out.DocumentMessage.ContextInfo = mark(
			out.DocumentMessage.ContextInfo)
	case out.StickerMessage != nil:
		out.StickerMessage.ContextInfo = mark(
			out.StickerMessage.ContextInfo)
	default:
		return nil, "this message kind has no forward shape " +
			"(reactions, calls, polls and interactive forms " +
			"can't be forwarded)"
	}
	return out, ""
}

// cmdRejectCall rejects one ringing call (the anticall toggle's
// verb): whatsmeow RejectCall against the offer's id + caller.
func (b *bridge) cmdRejectCall(ctx context.Context, cc *clientConn, id string, req request) {
	// The call id rides "call_id": "id" is the request
	// correlation id (see cmdStatusView).
	callID := strings.TrimSpace(req.str("call_id"))
	from, err := parseJID(req.str("from"))
	if err != nil || callID == "" {
		b.reply(cc, id, map[string]any{"ok": false,
			"error": "reject_call needs the call id and the " +
				"caller's jid"})
		return
	}
	if err := b.cli.RejectCall(ctx, from, callID); err != nil {
		b.fail(cc, id, err)
		return
	}
	b.reply(cc, id, map[string]any{"ok": true})
}

// cmdBlocklist blocks/unblocks one user: whatsmeow's blocklist
// update, which needs the current list hash — fetched fresh
// each time so a stale hash never eats the refusal silently.
func (b *bridge) cmdBlocklist(ctx context.Context, cc *clientConn, id string, req request) {
	target, err := parseJID(req.str("jid"))
	if err != nil {
		b.fail(cc, id, err)
		return
	}
	action := events.BlocklistChangeAction(req.str("action"))
	if action != events.BlocklistChangeActionBlock &&
		action != events.BlocklistChangeActionUnblock {
		b.reply(cc, id, map[string]any{"ok": false,
			"error": "blocklist action must be block or unblock"})
		return
	}
	bl, err := b.cli.GetBlocklist(ctx, "")
	if err != nil {
		b.fail(cc, id, err)
		return
	}
	if _, err := b.cli.UpdateBlocklist(ctx, target, action,
		bl.DHash); err != nil {
		b.fail(cc, id, err)
		return
	}
	b.reply(cc, id, map[string]any{"ok": true})
}

// cmdChatDisappear sets a chat's disappearing-messages timer:
// whatsmeow SetDisappearingTimer (a protocol setting message);
// 0 turns it off.  Works for DMs and groups alike.
func (b *bridge) cmdChatDisappear(ctx context.Context, cc *clientConn, id string, req request) {
	chat, err := parseJID(req.str("chat"))
	if err != nil {
		b.fail(cc, id, err)
		return
	}
	secs := req.num("seconds", -1)
	if secs < 0 {
		b.reply(cc, id, map[string]any{"ok": false,
			"error": "chat_disappear needs seconds >= 0 " +
				"(0 turns disappearing messages off)"})
		return
	}
	if err := b.cli.SetDisappearingTimer(ctx, chat,
		time.Duration(secs)*time.Second, time.Now()); err != nil {
		b.fail(cc, id, err)
		return
	}
	b.reply(cc, id, map[string]any{"ok": true})
}

// cmdCheckNumber answers the one registration question WhatsApp
// publishes (the banchecker's number signal): is this phone
// number on WhatsApp, and under which JID.  Nothing about bans
// is returned because WhatsApp publishes nothing about bans —
// the Python side says exactly that.
func (b *bridge) cmdCheckNumber(ctx context.Context, cc *clientConn, id string, req request) {
	phone := strings.TrimSpace(req.str("phone"))
	if phone == "" {
		b.reply(cc, id, map[string]any{"ok": false,
			"error": "check_number needs a phone number"})
		return
	}
	resp, err := b.cli.IsOnWhatsApp(ctx, []string{phone})
	if err != nil {
		b.fail(cc, id, err)
		return
	}
	if len(resp) == 0 {
		b.reply(cc, id, map[string]any{"ok": false,
			"error": "WhatsApp returned no answer for that number"})
		return
	}
	b.reply(cc, id, map[string]any{"ok": true,
		"exists": resp[0].IsIn, "jid": resp[0].JID.String()})
}

func (b *bridge) cmdChats(cc *clientConn, id string, req request) {
	limit := req.num("limit", 30)
	if limit < 1 {
		limit = 1
	}
	type row struct {
		id, title, kind string
		ts              int64
	}
	b.memMu.Lock()
	rows := make([]row, 0, len(b.recent))
	for jidStr, arr := range b.recent {
		if len(arr) == 0 {
			continue
		}
		jid, err := types.ParseJID(jidStr)
		if err != nil {
			continue
		}
		title := b.titles[jidStr]
		if title == "" {
			title = jid.User
		}
		rows = append(rows, row{
			id: jidStr, title: title, kind: kindOf(jid),
			ts: arr[len(arr)-1].tsMs,
		})
	}
	b.memMu.Unlock()
	sort.Slice(rows, func(i, j int) bool { return rows[i].ts > rows[j].ts })
	if len(rows) > limit {
		rows = rows[:limit]
	}
	chats := make([]map[string]any, 0, len(rows))
	for _, r := range rows {
		chats = append(chats, map[string]any{
			"id": r.id, "title": r.title, "kind": r.kind, "ts": r.ts,
		})
	}
	b.reply(cc, id, map[string]any{"ok": true, "chats": chats})
}
