// messages.go — the adapter protocol's core: inbound message /
// receipt events and the send / send_media / typing / history /
// read / chats / status commands. Shapes are the old bridge's,
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
	"time"

	"go.mau.fi/whatsmeow"
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

func (r request) boolean(key string) bool {
	v, _ := r[key].(bool)
	return v
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
	case "send_media":
		b.cmdSendMedia(ctx, cc, id, req)
	case "typing":
		b.cmdTyping(ctx, cc, id, req)
	case "history":
		b.cmdHistory(cc, id, req)
	case "read":
		b.cmdRead(ctx, cc, id, req)
	case "presence":
		b.cmdPresence(ctx, cc, id, req)
	case "status_view":
		b.cmdStatusView(ctx, cc, id, req)
	case "chats":
		b.cmdChats(cc, id, req)
	case "status":
		state := "closed"
		if b.cli.IsLoggedIn() {
			state = "open"
		}
		b.reply(cc, id, map[string]any{"ok": true, "user": b.meJID, "lid": b.myLID(), "state": state})
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

// downloadInbound fetches a message's media into the media dir and
// returns its wire row {path, mime, kind}, or nil.
func (b *bridge) downloadInbound(ctx context.Context, v *events.Message) map[string]any {
	kind := mediaKindOf(v.Message)
	if kind == "" {
		return nil
	}
	var dl whatsmeow.DownloadableMessage
	var mime, fileName string
	switch kind {
	case "image":
		dl, mime = v.Message.GetImageMessage(), v.Message.GetImageMessage().GetMimetype()
	case "video":
		dl, mime = v.Message.GetVideoMessage(), v.Message.GetVideoMessage().GetMimetype()
	case "voice", "audio":
		dl, mime = v.Message.GetAudioMessage(), v.Message.GetAudioMessage().GetMimetype()
	case "file":
		dl = v.Message.GetDocumentMessage()
		mime = v.Message.GetDocumentMessage().GetMimetype()
		fileName = v.Message.GetDocumentMessage().GetFileName()
	case "sticker":
		dl, mime = v.Message.GetStickerMessage(), v.Message.GetStickerMessage().GetMimetype()
	}
	if dl == nil {
		return nil
	}
	data, err := b.cli.Download(ctx, dl)
	if err != nil {
		logf("media download failed: %v", err)
		return nil
	}
	if mime == "" {
		mime = "application/octet-stream"
	}
	ext := ""
	if idx := strings.LastIndex(fileName, "."); idx >= 0 {
		ext = strings.ToLower(fileName[idx:])
	} else {
		ext = map[string]string{
			"image": ".jpg", "video": ".mp4", "voice": ".ogg",
			"audio": ".ogg", "sticker": ".webp",
		}[kind]
		if ext == "" {
			ext = ".bin"
		}
	}
	short := unsafeIDChars.ReplaceAllString(string(v.Info.ID), "")
	if len(short) > 12 {
		short = short[:12]
	}
	name := strconv.FormatInt(time.Now().UnixMilli(), 10) + "-" + short + ext
	p := filepath.Join(b.mediaDir, name)
	if err := os.WriteFile(p, data, 0o644); err != nil {
		logf("media write failed: %v", err)
		return nil
	}
	return map[string]any{"path": p, "mime": mime, "kind": kind}
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

	if chat == types.StatusBroadcastJID {
		if v.Info.IsFromMe {
			return
		}
		var media []map[string]any
		if m := b.downloadInbound(ctx, v); m != nil {
			media = []map[string]any{m}
		}
		b.broadcast(map[string]any{
			"type": "status_post", "id": string(v.Info.ID),
			"from":      map[string]any{"id": sender.String(), "name": v.Info.PushName},
			"text":      textOf(v.Message),
			"media":     media,
			"view_once": v.IsViewOnce,
			"ts":        tsMs,
		})
		return
	}

	kind := kindOf(chat)
	if v.Info.IsFromMe {
		out := map[string]any{
			"type": "own_message", "id": string(v.Info.ID),
			"chat": map[string]any{
				"id": chat.String(), "kind": kind, "title": b.titleOf(chat),
			},
			"text":      textOf(v.Message),
			"view_once": v.IsViewOnce,
			"ts":        tsMs,
		}
		b.remember(chat, &remembered{
			incoming: false, text: out["text"].(string),
			id: v.Info.ID, chatJID: chat, senderJID: sender, tsMs: tsMs,
		})
		b.broadcast(out)
		return
	}

	title := b.titleFor(ctx, chat, kind, v.Info.PushName, sender)
	var media []map[string]any
	if m := b.downloadInbound(ctx, v); m != nil {
		media = []map[string]any{m}
	}
	mentioned := false
	if ci := contextInfoOf(v.Message); ci != nil {
		for _, mj := range ci.GetMentionedJID() {
			parsed, err := types.ParseJID(mj)
			if err != nil {
				continue
			}
			if b.isMe(parsed.ToNonAD()) {
				mentioned = true
				break
			}
		}
	}
	replyTo := ""
	if ci := contextInfoOf(v.Message); ci != nil {
		replyTo = ci.GetStanzaID()
	}
	out := map[string]any{
		"type": "message", "id": string(v.Info.ID),
		"chat":      map[string]any{"id": chat.String(), "kind": kind, "title": title},
		"from":      map[string]any{"id": sender.String(), "name": v.Info.PushName},
		"text":      textOf(v.Message),
		"media":     media,
		"reply_to":  replyTo,
		"mentioned": mentioned,
		"view_once": v.IsViewOnce,
		"ts":        tsMs,
	}
	if v.Info.ID != "" {
		b.cacheMessage(string(v.Info.ID), v.Message, sender)
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

func (b *bridge) cmdSend(ctx context.Context, cc *clientConn, id string, req request) {
	chat, err := parseJID(req.str("chat"))
	if err != nil {
		b.fail(cc, id, err)
		return
	}
	text := req.str("text")
	replyTo := req.str("reply_to")
	quote := b.quoteContext(replyTo)

	type btn struct{ label, cmd string }
	var flat []btn
	if rows, ok := req["buttons"].([]any); ok {
		for _, rowAny := range rows {
			row, _ := rowAny.([]any)
			for _, bAny := range row {
				pair, _ := bAny.([]any)
				if len(pair) >= 2 {
					label, _ := pair[0].(string)
					cmd, _ := pair[1].(string)
					flat = append(flat, btn{label, cmd})
				}
			}
		}
	}
	send := func(msg *waE2E.Message) (types.MessageID, error) {
		resp, err := b.cli.SendMessage(ctx, chat, msg)
		if err != nil {
			return "", err
		}
		return resp.ID, nil
	}

	if len(flat) > 0 {
		asLines := func(items []btn) string {
			parts := make([]string, 0, len(items))
			for _, it := range items {
				parts = append(parts, "• "+it.label+" — /"+it.cmd)
			}
			return strings.Join(parts, "\n")
		}
		native := flat
		body := text
		if len(flat) > 3 {
			native = flat[:3]
			if body != "" {
				body += "\n"
			}
			body += asLines(flat[3:])
		}
		buttons := make([]*waE2E.ButtonsMessage_Button, 0, len(native))
		for _, it := range native {
			label := it.label
			if len(label) > 25 {
				label = label[:25]
			}
			btnType := waE2E.ButtonsMessage_Button_RESPONSE
			buttons = append(buttons, &waE2E.ButtonsMessage_Button{
				ButtonID:   proto.String(it.cmd),
				ButtonText: &waE2E.ButtonsMessage_Button_ButtonText{DisplayText: proto.String(label)},
				Type:       &btnType,
			})
		}
		headerType := waE2E.ButtonsMessage_EMPTY
		msg := &waE2E.Message{ButtonsMessage: &waE2E.ButtonsMessage{
			ContentText: proto.String(body),
			Buttons:     buttons,
			HeaderType:  &headerType,
			ContextInfo: quote,
		}}
		if mid, err := send(msg); err == nil {
			b.reply(cc, id, map[string]any{"ok": true, "message_id": string(mid)})
			return
		}
		// WhatsApp refused the button message — every action
		// degrades to a text line; none are silently dropped.
		lines := asLines(flat)
		full := text
		if full != "" {
			full += "\n"
		}
		full += lines
		mid, err := send(b.textMessage(full, quote))
		if err != nil {
			b.fail(cc, id, err)
			return
		}
		b.reply(cc, id, map[string]any{"ok": true, "message_id": string(mid)})
		return
	}

	mid, err := send(b.textMessage(text, quote))
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
	resp, err := b.cli.SendMessage(ctx, chat, msg)
	if err != nil {
		b.fail(cc, id, err)
		return
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
		media := m.media
		if media == nil {
			media = []map[string]any{}
		}
		ts := m.tsMs
		if ts == 0 {
			ts = time.Now().UnixMilli()
		}
		msgs = append(msgs, map[string]any{
			"incoming": m.incoming, "text": m.text, "sender": m.sender,
			"media": media, "reply_to": m.replyTo, "ts": ts,
		})
	}
	b.reply(cc, id, map[string]any{"ok": true, "messages": msgs})
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
	if req.str("state") == "unavailable" {
		state = types.PresenceUnavailable
	}
	if err := b.cli.SendPresence(ctx, state); err != nil {
		b.fail(cc, id, err)
		return
	}
	b.reply(cc, id, map[string]any{"ok": true})
}

// cmdStatusView marks one contact status as viewed by this
// account (§15 status viewing): a read receipt for the status
// post — MarkRead addressed to status@broadcast with the poster
// as sender, the same receipt the phone sends when the owner
// opens the status.
func (b *bridge) cmdStatusView(ctx context.Context, cc *clientConn, id string, req request) {
	msgID := req.str("id")
	sender, err := parseJID(req.str("from"))
	if err != nil || msgID == "" {
		b.reply(cc, id, map[string]any{"ok": false,
			"error": "status_view needs the status message id and the poster's jid"})
		return
	}
	tsMs := int64(req.num("ts", 0))
	if tsMs <= 0 {
		tsMs = time.Now().UnixMilli()
	}
	if err := b.cli.MarkRead(ctx, []types.MessageID{types.MessageID(msgID)},
		time.UnixMilli(tsMs), types.StatusBroadcastJID, sender); err != nil {
		b.fail(cc, id, err)
		return
	}
	b.reply(cc, id, map[string]any{"ok": true})
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
