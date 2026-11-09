package main

// profile.go — the account's OWN identity + polls (parity G3,
// census bundles #5 and #6).
//
// Bundle #5: Devon could read the account's push name and
// change nothing — no verb existed on either engine to set the
// profile name, the about text, the profile photo, or any
// privacy setting.  The verbs here ride the whatsmeow calls
// that actually exist for each act:
//
//   name    appstate.BuildSettingPushName + SendAppState — the
//           push name IS an app-state setting; the local store
//           copy is updated only after the patch lands, so a
//           refused patch never leaves the bridge claiming a
//           name the server rejected.
//   about   Client.SetStatusMessage — despite the name, this
//           is the profile "About" text (its own doc says so),
//           NOT a status broadcast.
//   photo   Client.SetGroupPhoto addressed at the account's
//           OWN jid — this pin carries no SetProfilePicture;
//           the picture IQ it sends (to the server, target =
//           the jid) is the same addressing GetProfilePictureInfo
//           reads with, and nil bytes remove the photo.  The
//           avatar must be JPEG (the library's own contract:
//           other formats are rejected server-side), so a
//           non-JPEG file refuses here in words instead of
//           surfacing a raw IQ error.
//   privacy Client.SetPrivacySetting per category, with the
//           value vocabulary the library itself documents per
//           category (types/user.go) — an out-of-vocabulary
//           value refuses BEFORE traffic, because the server
//           silently ignoring a bad category value would be a
//           fake success.
//
// Bundle #6: polls.  WhatsApp poll sending is
// Client.BuildPollCreation + SendMessage — the creation
// message carries a message secret, and votes come back as
// PollUpdateMessages whose payloads only the poll's
// participants can decrypt (Client.DecryptPollVote).  The
// bridge therefore keeps poll memory: creations it sees (its
// own sends included) and every vote it manages to decrypt,
// so poll_results can answer with counted votes.  A vote that
// fails decryption is counted as unreadable and NAMED in the
// results reply — never silently dropped, never guessed.

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"go.mau.fi/whatsmeow"
	"go.mau.fi/whatsmeow/appstate"
	waE2E "go.mau.fi/whatsmeow/proto/waE2E"
	"go.mau.fi/whatsmeow/types"
	"go.mau.fi/whatsmeow/types/events"
	"google.golang.org/protobuf/proto"
)

// ── profile / privacy ────────────────────────────────────────

// profileClient is the slice of *whatsmeow.Client the identity
// verbs ride — an interface so the verbs' call SHAPES are
// pinnable in tests (the status-fix lesson: a recording fake
// on the Python side accepts any shape; the bridge layer is
// where shapes must be pinned).
type profileClient interface {
	SendAppState(ctx context.Context,
		patch appstate.PatchInfo) error
	SetStatusMessage(ctx context.Context,
		status types.SetStatusInput) error
	SetGroupPhoto(ctx context.Context, jid types.JID,
		avatar []byte) (string, error)
	GetUserInfo(ctx context.Context, jids []types.JID,
	) (map[types.JID]types.UserInfo, error)
	TryFetchPrivacySettings(ctx context.Context,
		ignoreCache bool) (*types.PrivacySettings, error)
	SetPrivacySetting(ctx context.Context,
		name types.PrivacySettingType,
		value types.PrivacySetting,
	) (types.PrivacySettings, error)
}

var _ profileClient = (*whatsmeow.Client)(nil)

// applyProfileName sets the account's push name: the app-state
// patch goes first; the local store copy updates only on
// success (a refused patch must not leave the bridge — or the
// presence it asserts — wearing a name the server rejected).
func applyProfileName(ctx context.Context, cli profileClient,
	name string, store func(string)) error {
	if err := cli.SendAppState(ctx,
		appstate.BuildSettingPushName(name)); err != nil {
		return err
	}
	if store != nil {
		store(name)
	}
	return nil
}

// privacyCategory maps one wire category name to its whatsmeow
// type, the struct field that reads it back, and the value
// vocabulary the library documents for it.
type privacyCategory struct {
	typ    types.PrivacySettingType
	get    func(types.PrivacySettings) types.PrivacySetting
	values []types.PrivacySetting
}

var _stdPrivacyValues = []types.PrivacySetting{
	types.PrivacySettingAll, types.PrivacySettingContacts,
	types.PrivacySettingContactBlacklist, types.PrivacySettingNone,
}

var privacyCategories = map[string]privacyCategory{
	"groupadd": {
		typ: types.PrivacySettingTypeGroupAdd,
		get: func(s types.PrivacySettings) types.PrivacySetting {
			return s.GroupAdd
		},
		values: _stdPrivacyValues,
	},
	"lastseen": {
		typ: types.PrivacySettingTypeLastSeen,
		get: func(s types.PrivacySettings) types.PrivacySetting {
			return s.LastSeen
		},
		values: _stdPrivacyValues,
	},
	"status": {
		typ: types.PrivacySettingTypeStatus,
		get: func(s types.PrivacySettings) types.PrivacySetting {
			return s.Status
		},
		values: _stdPrivacyValues,
	},
	"profile": {
		typ: types.PrivacySettingTypeProfile,
		get: func(s types.PrivacySettings) types.PrivacySetting {
			return s.Profile
		},
		values: _stdPrivacyValues,
	},
	"readreceipts": {
		typ: types.PrivacySettingTypeReadReceipts,
		get: func(s types.PrivacySettings) types.PrivacySetting {
			return s.ReadReceipts
		},
		values: []types.PrivacySetting{
			types.PrivacySettingAll, types.PrivacySettingNone},
	},
	"online": {
		typ: types.PrivacySettingTypeOnline,
		get: func(s types.PrivacySettings) types.PrivacySetting {
			return s.Online
		},
		values: []types.PrivacySetting{
			types.PrivacySettingAll,
			types.PrivacySettingMatchLastSeen},
	},
	"calladd": {
		typ: types.PrivacySettingTypeCallAdd,
		get: func(s types.PrivacySettings) types.PrivacySetting {
			return s.CallAdd
		},
		values: []types.PrivacySetting{
			types.PrivacySettingAll, types.PrivacySettingKnown},
	},
	// The categories below are settable through the same IQ
	// (SetPrivacySetting sends any category name) but have no
	// field in the returned PrivacySettings struct, so a get
	// cannot read them back — get answers "" (unknown) for
	// them, never a fabricated current value.
	"messages": {
		typ: types.PrivacySettingTypeMessages,
		get: func(types.PrivacySettings) types.PrivacySetting {
			return ""
		},
		values: []types.PrivacySetting{
			types.PrivacySettingAll, types.PrivacySettingContacts},
	},
	"defense": {
		typ: types.PrivacySettingTypeDefense,
		get: func(types.PrivacySettings) types.PrivacySetting {
			return ""
		},
		values: []types.PrivacySetting{
			types.PrivacySettingOnStandard,
			types.PrivacySettingOff},
	},
	"stickers": {
		typ: types.PrivacySettingTypeStickers,
		get: func(types.PrivacySettings) types.PrivacySetting {
			return ""
		},
		values: []types.PrivacySetting{
			types.PrivacySettingContacts,
			types.PrivacySettingContactAllowlist,
			types.PrivacySettingNone},
	},
}

// privacySettingsMap renders fetched settings as the wire map:
// category name → value, only for categories the struct can
// actually read (an unread category is ABSENT, not "").
func privacySettingsMap(s types.PrivacySettings) map[string]any {
	out := map[string]any{}
	for name, cat := range privacyCategories {
		if v := cat.get(s); v != "" {
			out[name] = string(v)
		}
	}
	return out
}

// setPrivacy validates (category, value) against the library's
// own vocabulary and applies it, returning the post-set
// settings map.  Validation refuses BEFORE traffic: the server
// silently ignoring an out-of-vocabulary value would read back
// as a success that never happened.
func setPrivacy(ctx context.Context, cli profileClient,
	category, value string) (map[string]any, error) {
	cat, ok := privacyCategories[category]
	if !ok {
		return nil, fmt.Errorf("unknown privacy category %q — "+
			"known: groupadd, lastseen, status, profile, "+
			"readreceipts, online, calladd, messages, defense, "+
			"stickers", category)
	}
	want := types.PrivacySetting(value)
	valid := false
	for _, v := range cat.values {
		if v == want {
			valid = true
			break
		}
	}
	if !valid {
		names := make([]string, len(cat.values))
		for i, v := range cat.values {
			names[i] = string(v)
		}
		return nil, fmt.Errorf("privacy %s cannot be %q — "+
			"its values are: %s", category, value,
			strings.Join(names, ", "))
	}
	settings, err := cli.SetPrivacySetting(ctx, cat.typ, want)
	if err != nil {
		return nil, err
	}
	return privacySettingsMap(settings), nil
}

func (b *bridge) cmdProfileGet(ctx context.Context, cc *clientConn, id string, req request) {
	out := map[string]any{"ok": true, "name": b.myPushName()}
	if b.cli.Store == nil || b.cli.Store.ID == nil {
		b.reply(cc, id, out)
		return
	}
	infos, err := b.cli.GetUserInfo(ctx,
		[]types.JID{b.cli.Store.ID.ToNonAD()})
	if err != nil {
		// The name is local truth; the about/photo read is
		// not — a failed read is reported in the reply, never
		// papered over with invented values.
		out["info_error"] = err.Error()
		b.reply(cc, id, out)
		return
	}
	info := infos[b.cli.Store.ID.ToNonAD()]
	out["about"] = info.Status
	out["picture_id"] = info.PictureID
	out["has_photo"] = info.PictureID != ""
	b.reply(cc, id, out)
}

func (b *bridge) cmdProfileSet(ctx context.Context, cc *clientConn, id string, req request) {
	field := strings.TrimSpace(req.str("field"))
	switch field {
	case "name":
		name := strings.TrimSpace(req.str("value"))
		if name == "" {
			b.reply(cc, id, map[string]any{"ok": false,
				"error": "profile_set name needs a name"})
			return
		}
		err := applyProfileName(ctx, b.cli, name,
			func(n string) {
				if b.cli.Store != nil {
					b.cli.Store.PushName = n
				}
			})
		if err != nil {
			b.fail(cc, id, err)
			return
		}
		b.reply(cc, id, map[string]any{"ok": true,
			"field": "name", "name": name})
	case "about":
		text := req.str("value")
		if err := b.cli.SetStatusMessage(ctx,
			types.SetStatusInput{Text: &text}); err != nil {
			b.fail(cc, id, err)
			return
		}
		b.reply(cc, id, map[string]any{"ok": true,
			"field": "about", "about": text})
	case "photo":
		if b.cli.Store == nil || b.cli.Store.ID == nil {
			b.reply(cc, id, map[string]any{"ok": false,
				"error": "not logged in — no account to set " +
					"a photo on"})
			return
		}
		own := b.cli.Store.ID.ToNonAD()
		if req.boolean("remove") {
			if _, err := b.cli.SetGroupPhoto(ctx, own,
				nil); err != nil {
				b.fail(cc, id, err)
				return
			}
			b.reply(cc, id, map[string]any{"ok": true,
				"field": "photo", "removed": true})
			return
		}
		path := strings.TrimSpace(req.str("path"))
		if path == "" {
			b.reply(cc, id, map[string]any{"ok": false,
				"error": "profile_set photo needs a path " +
					"(or remove)"})
			return
		}
		data, err := os.ReadFile(path)
		if err != nil {
			b.fail(cc, id, err)
			return
		}
		if len(data) < 3 || data[0] != 0xFF || data[1] != 0xD8 {
			b.reply(cc, id, map[string]any{"ok": false,
				"error": "WhatsApp profile photos must be " +
					"JPEG — that file isn't one"})
			return
		}
		pictureID, err := b.cli.SetGroupPhoto(ctx, own, data)
		if err != nil {
			b.fail(cc, id, err)
			return
		}
		b.reply(cc, id, map[string]any{"ok": true,
			"field": "photo", "picture_id": pictureID})
	default:
		b.reply(cc, id, map[string]any{"ok": false,
			"error": "profile_set field must be name, about " +
				"or photo"})
	}
}

func (b *bridge) cmdPrivacyGet(ctx context.Context, cc *clientConn, id string, req request) {
	settings, err := b.cli.TryFetchPrivacySettings(ctx, true)
	if err != nil {
		b.fail(cc, id, err)
		return
	}
	b.reply(cc, id, map[string]any{"ok": true,
		"settings": privacySettingsMap(*settings)})
}

func (b *bridge) cmdPrivacySet(ctx context.Context, cc *clientConn, id string, req request) {
	category := strings.TrimSpace(
		strings.ToLower(req.str("category")))
	value := strings.TrimSpace(strings.ToLower(req.str("value")))
	settings, err := setPrivacy(ctx, b.cli, category, value)
	if err != nil {
		b.reply(cc, id, map[string]any{"ok": false,
			"error": err.Error()})
		return
	}
	b.reply(cc, id, map[string]any{"ok": true,
		"category": category, "value": value,
		"settings": settings})
}

// ── polls ────────────────────────────────────────────────────

// pollClient is the slice of *whatsmeow.Client the poll verbs
// ride (see profileClient for why).
type pollClient interface {
	BuildPollCreation(name string, optionNames []string,
		selectableOptionCount int) *waE2E.Message
	SendMessage(ctx context.Context, to types.JID,
		message *waE2E.Message,
		extra ...whatsmeow.SendRequestExtra,
	) (whatsmeow.SendResponse, error)
}

var _ pollClient = (*whatsmeow.Client)(nil)

// pollShapeError validates one poll against WhatsApp's own
// shape rules BEFORE traffic (the same discipline the
// Telegram adapters apply): 2–12 options, a question of at
// most 255 characters, options of at most 100, and a
// selectable count within the option list.
func pollShapeError(question string, options []string,
	selectable int) string {
	question = strings.TrimSpace(question)
	if question == "" {
		return "a poll needs a question"
	}
	if len([]rune(question)) > 255 {
		return "poll questions cap at 255 characters"
	}
	if len(options) < 2 || len(options) > 12 {
		return "a WhatsApp poll needs 2–12 options"
	}
	for _, o := range options {
		if strings.TrimSpace(o) == "" {
			return "poll options cannot be empty"
		}
		if len([]rune(o)) > 100 {
			return "poll options cap at 100 characters each"
		}
	}
	if selectable < 1 || selectable > len(options) {
		return "the selectable count must be between 1 and " +
			"the number of options"
	}
	return ""
}

// sendPoll builds and sends one poll, returning its message
// id.  selectable is the platform's own count: 1 = one choice,
// len(options) = any number of choices.
func sendPoll(ctx context.Context, cli pollClient,
	chat types.JID, question string, options []string,
	selectable int) (types.MessageID, *waE2E.Message, error) {
	msg := cli.BuildPollCreation(question, options, selectable)
	resp, err := cli.SendMessage(ctx, chat, msg)
	if err != nil {
		return "", nil, err
	}
	return resp.ID, msg, nil
}

// pollState is one poll the bridge remembers: its shape and
// every decrypted vote, keyed by voter.  Votes store option
// HASHES (hex sha256 of the option name — the protocol's own
// identity for an option), so a re-vote replaces the voter's
// earlier selection exactly like the phone's tally does.
type pollState struct {
	question   string
	options    []string
	hashes     []string
	votes      map[string][]string
	unreadable int
	tsMs       int64
}

func pollKey(chat types.JID, pollID string) string {
	return chat.String() + "\x00" + pollID
}

// optionHash is the protocol's identity for one poll option:
// sha256 of the option's name bytes (whatsmeow HashPollOptions
// — mirrored here as hex for map keys).
func optionHash(name string) string {
	sum := sha256.Sum256([]byte(name))
	return hex.EncodeToString(sum[:])
}

const maxPollsKept = 200

func (b *bridge) notePollCreation(chat types.JID, pollID string,
	pc *waE2E.PollCreationMessage, tsMs int64) {
	if pc == nil || pollID == "" {
		return
	}
	options := make([]string, 0, len(pc.GetOptions()))
	for _, o := range pc.GetOptions() {
		options = append(options, o.GetOptionName())
	}
	if len(options) == 0 {
		return
	}
	b.pollsMu.Lock()
	defer b.pollsMu.Unlock()
	if b.polls == nil {
		b.polls = map[string]*pollState{}
		b.pollOrder = nil
	}
	key := pollKey(chat, pollID)
	if _, seen := b.polls[key]; !seen {
		b.pollOrder = append(b.pollOrder, key)
		for len(b.pollOrder) > maxPollsKept {
			delete(b.polls, b.pollOrder[0])
			b.pollOrder = b.pollOrder[1:]
		}
	}
	hashes := make([]string, len(options))
	for i, name := range options {
		hashes[i] = optionHash(name)
	}
	st := b.polls[key]
	if st == nil {
		st = &pollState{votes: map[string][]string{}}
		b.polls[key] = st
	}
	st.question = pc.GetName()
	st.options = options
	st.hashes = hashes
	st.tsMs = tsMs
}

// recordPollVote lands one voter's decrypted selection on a
// remembered poll.  Returns false when the poll itself is not
// in memory (a vote for a poll this bridge never saw cannot be
// tallied — its option names are unknown).
func (b *bridge) recordPollVote(chat types.JID, pollID,
	voter string, selected [][]byte) bool {
	b.pollsMu.Lock()
	defer b.pollsMu.Unlock()
	st := b.polls[pollKey(chat, pollID)]
	if st == nil {
		return false
	}
	hashes := make([]string, 0, len(selected))
	for _, h := range selected {
		hashes = append(hashes, hex.EncodeToString(h))
	}
	st.votes[voter] = hashes
	return true
}

func (b *bridge) notePollVoteUnreadable(chat types.JID,
	pollID string) {
	b.pollsMu.Lock()
	defer b.pollsMu.Unlock()
	if st := b.polls[pollKey(chat, pollID)]; st != nil {
		st.unreadable++
	}
}

// pollResults tallies one remembered poll: per-option voter
// counts + the number of distinct voters.  Only decrypted
// votes count; unreadable ones ride along as their own number.
func (b *bridge) pollResults(chat types.JID, pollID string,
) (map[string]any, bool) {
	b.pollsMu.Lock()
	defer b.pollsMu.Unlock()
	st := b.polls[pollKey(chat, pollID)]
	if st == nil {
		return nil, false
	}
	counts := make([]int, len(st.options))
	for _, selected := range st.votes {
		for _, h := range selected {
			for i, oh := range st.hashes {
				if h == oh {
					counts[i]++
					break
				}
			}
		}
	}
	options := make([]map[string]any, len(st.options))
	for i, name := range st.options {
		options[i] = map[string]any{
			"text": name, "voters": counts[i]}
	}
	out := map[string]any{
		"question":     st.question,
		"options":      options,
		"total_voters": len(st.votes),
	}
	if st.unreadable > 0 {
		out["unreadable_votes"] = st.unreadable
	}
	return out, true
}

// notePollEvent captures poll creations and votes off the
// inbound event stream (called from onMessage for every
// message, own sends included — creations are how the bridge
// learns a poll's option names, votes arrive encrypted and
// only DecryptPollVote can open them).
func (b *bridge) notePollEvent(ctx context.Context,
	v *events.Message) {
	if v.Message == nil {
		return
	}
	chat := v.Info.Chat.ToNonAD()
	if pc := pollCreationOf(v.Message); pc != nil {
		b.notePollCreation(chat, string(v.Info.ID), pc,
			v.Info.Timestamp.UnixMilli())
	}
	pu := v.Message.GetPollUpdateMessage()
	if pu == nil {
		return
	}
	pollID := pu.GetPollCreationMessageKey().GetID()
	if pollID == "" {
		return
	}
	vote, err := b.cli.DecryptPollVote(ctx, v)
	if err != nil {
		b.notePollVoteUnreadable(chat, pollID)
		logf("poll vote on %s could not be decrypted: %v",
			pollID, err)
		return
	}
	b.recordPollVote(chat, pollID,
		v.Info.Sender.ToNonAD().String(),
		vote.GetSelectedOptions())
}

// pollCreationOf unwraps any of the three poll-creation forms
// (the original and its V2/V3 re-wraps) to the common shape.
func pollCreationOf(m *waE2E.Message) *waE2E.PollCreationMessage {
	if m == nil {
		return nil
	}
	if pc := m.GetPollCreationMessage(); pc != nil {
		return pc
	}
	if pc := m.GetPollCreationMessageV2(); pc != nil {
		return pc
	}
	return m.GetPollCreationMessageV3()
}

func (b *bridge) cmdSendPoll(ctx context.Context, cc *clientConn, id string, req request) {
	chat, err := parseJID(req.str("chat"))
	if err != nil {
		b.fail(cc, id, err)
		return
	}
	question := strings.TrimSpace(req.str("question"))
	rawOptions, _ := req["options"].([]any)
	options := make([]string, 0, len(rawOptions))
	for _, o := range rawOptions {
		if s, ok := o.(string); ok {
			options = append(options, strings.TrimSpace(s))
		}
	}
	selectable := req.num("selectable", 1)
	if msg := pollShapeError(question, options,
		selectable); msg != "" {
		b.reply(cc, id, map[string]any{"ok": false,
			"error": msg})
		return
	}
	msgID, built, err := sendPoll(ctx, b.cli, chat, question,
		options, selectable)
	if err != nil {
		b.fail(cc, id, err)
		return
	}
	if pc := pollCreationOf(built); pc != nil {
		b.notePollCreation(chat, string(msgID), pc,
			time.Now().UnixMilli())
	}
	b.reply(cc, id, map[string]any{"ok": true,
		"message_id": string(msgID)})
}

func (b *bridge) cmdPollResults(ctx context.Context, cc *clientConn, id string, req request) {
	chat, err := parseJID(req.str("chat"))
	if err != nil {
		b.fail(cc, id, err)
		return
	}
	mid := strings.TrimSpace(req.str("message_id"))
	if mid == "" {
		b.reply(cc, id, map[string]any{"ok": false,
			"error": "poll_results needs a message_id"})
		return
	}
	out, ok := b.pollResults(chat, mid)
	if !ok {
		b.reply(cc, id, map[string]any{"ok": false,
			"error": "that poll is not in bridge memory — " +
				"results live on polls this bridge has seen " +
				"since it started"})
		return
	}
	out["ok"] = true
	b.reply(cc, id, out)
}

// ── location + contact cards (parity bundle #9) ─────────────
//
// The census graded both ABSENT on WhatsApp: no location or
// contact message type existed in either bridge's send path.
// Both are plain waE2E messages on this engine — a
// LocationMessage (degrees + the name/address label fields,
// which are also WhatsApp's venue form) and a ContactMessage
// (display name + the vCard text recipients' phones parse).
// Shape rules refuse BEFORE traffic, mirroring the Python
// adapter's (base.location_shape_error / contact_shape_error)
// — one rule, stated in each layer that can be reached
// without the other.

// pinClient is the slice of *whatsmeow.Client the pin/card
// verbs ride — just SendMessage, like the poll send.
type pinClient interface {
	SendMessage(ctx context.Context, to types.JID,
		message *waE2E.Message,
		extra ...whatsmeow.SendRequestExtra,
	) (whatsmeow.SendResponse, error)
}

func locationShapeError(lat, lon float64, hasLat, hasLon bool) string {
	if !hasLat || !hasLon {
		return "a location pin needs numeric latitude and longitude"
	}
	if math.IsNaN(lat) || math.IsNaN(lon) ||
		math.IsInf(lat, 0) || math.IsInf(lon, 0) {
		return "a location pin's coordinates must be finite numbers"
	}
	if lat < -90 || lat > 90 {
		return "a location pin needs latitude between -90 and 90"
	}
	if lon < -180 || lon > 180 {
		return "a location pin needs longitude between -180 and 180"
	}
	return ""
}

func buildLocationMessage(lat, lon float64, name, address string) *waE2E.Message {
	loc := &waE2E.LocationMessage{
		DegreesLatitude:  proto.Float64(lat),
		DegreesLongitude: proto.Float64(lon),
	}
	if strings.TrimSpace(name) != "" {
		loc.Name = proto.String(name)
	}
	if strings.TrimSpace(address) != "" {
		loc.Address = proto.String(address)
	}
	return &waE2E.Message{LocationMessage: loc}
}

func contactCardError(name, vcard string) string {
	if strings.TrimSpace(name) == "" {
		return "a contact card needs a display name"
	}
	if strings.TrimSpace(vcard) == "" {
		return "a contact card needs a vcard — it is what the recipient's phone parses"
	}
	return ""
}

func buildContactMessage(name, vcard string) *waE2E.Message {
	return &waE2E.Message{ContactMessage: &waE2E.ContactMessage{
		DisplayName: proto.String(name),
		Vcard:       proto.String(vcard),
	}}
}

func sendPin(ctx context.Context, cli pinClient,
	chat types.JID, msg *waE2E.Message) (types.MessageID, error) {
	resp, err := cli.SendMessage(ctx, chat, msg)
	if err != nil {
		return "", err
	}
	return resp.ID, nil
}

func (b *bridge) cmdSendLocation(ctx context.Context, cc *clientConn, id string, req request) {
	chat, err := parseJID(req.str("chat"))
	if err != nil {
		b.fail(cc, id, err)
		return
	}
	lat, hasLat := req.flt("latitude")
	lon, hasLon := req.flt("longitude")
	if msg := locationShapeError(lat, lon, hasLat, hasLon); msg != "" {
		b.reply(cc, id, map[string]any{"ok": false,
			"error": msg})
		return
	}
	msgID, err := sendPin(ctx, b.cli, chat,
		buildLocationMessage(lat, lon, req.str("name"),
			req.str("address")))
	if err != nil {
		b.fail(cc, id, err)
		return
	}
	b.reply(cc, id, map[string]any{"ok": true,
		"message_id": string(msgID)})
}

func (b *bridge) cmdSendContact(ctx context.Context, cc *clientConn, id string, req request) {
	chat, err := parseJID(req.str("chat"))
	if err != nil {
		b.fail(cc, id, err)
		return
	}
	name := strings.TrimSpace(req.str("name"))
	vcard := req.str("vcard")
	if msg := contactCardError(name, vcard); msg != "" {
		b.reply(cc, id, map[string]any{"ok": false,
			"error": msg})
		return
	}
	msgID, err := sendPin(ctx, b.cli, chat,
		buildContactMessage(name, vcard))
	if err != nil {
		b.fail(cc, id, err)
		return
	}
	b.reply(cc, id, map[string]any{"ok": true,
		"message_id": string(msgID)})
}
