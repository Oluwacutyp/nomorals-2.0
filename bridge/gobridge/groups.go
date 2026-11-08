// groups.go — group / community / channel commands for the
// adapter protocol: the read-only set (group_list, group_info,
// group_participants, group_join_requests, community_list) and
// the mutating admin set (create / members / subject /
// description / picture / settings / join-request decisions /
// leave, communities, newsletters). Reply shapes are the old
// bridges' (whatsapp-groups.mjs + whatsapp-groups-admin.mjs),
// field for field. Where this whatsmeow build has no verb for a
// command, the reply is an honest ok:false naming the gap — the
// same guard style the Baileys modules used.
package main

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"go.mau.fi/whatsmeow"
	waE2E "go.mau.fi/whatsmeow/proto/waE2E"
	"go.mau.fi/whatsmeow/types"
	"google.golang.org/protobuf/proto"
)

// ── sanitizers: the wire gets plain JSON, never library internals ──

func groupJID(raw string) (types.JID, error) {
	jid, err := parseJID(raw)
	if err != nil || jid.IsEmpty() {
		return types.JID{}, errors.New("missing group: pass a group JID (…@g.us)")
	}
	if jid.Server != types.GroupServer {
		return types.JID{}, errors.New("not a group JID: " + raw + " (groups end in @g.us)")
	}
	return jid, nil
}

func newsletterJID(raw string) (types.JID, error) {
	jid, err := parseJID(raw)
	if err != nil || jid.IsEmpty() {
		return types.JID{}, errors.New("missing channel: pass a channel JID (…@newsletter)")
	}
	if jid.Server != types.NewsletterServer {
		return types.JID{}, errors.New("not a channel JID: " + raw + " (channels end in @newsletter)")
	}
	return jid, nil
}

func participantJIDs(raw any) ([]types.JID, error) {
	var list []any
	switch v := raw.(type) {
	case []any:
		list = v
	case string:
		list = []any{v}
	case nil:
		list = nil
	default:
		list = []any{v}
	}
	out := make([]types.JID, 0, len(list))
	for _, item := range list {
		s, _ := item.(string)
		jid, err := parseJID(s)
		if err == nil && !jid.IsEmpty() {
			out = append(out, jid)
		}
	}
	if len(out) == 0 {
		return nil, errors.New("no participants: pass at least one user JID")
	}
	return out, nil
}

func memberAction(raw string) (whatsmeow.ParticipantChange, error) {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "add":
		return whatsmeow.ParticipantChangeAdd, nil
	case "remove":
		return whatsmeow.ParticipantChangeRemove, nil
	case "promote":
		return whatsmeow.ParticipantChangePromote, nil
	case "demote":
		return whatsmeow.ParticipantChangeDemote, nil
	}
	return "", errors.New("bad action: " + raw + " (want one of add|remove|promote|demote)")
}

// friendlyError turns library failures into owner-readable errors,
// the old modules' mapping.
func friendlyError(err error) string {
	msg := err.Error()
	code := 0
	switch {
	case errors.Is(err, whatsmeow.ErrGroupNotFound):
		code = 404
	case errors.Is(err, whatsmeow.ErrNotInGroup),
		errors.Is(err, whatsmeow.ErrGroupInviteLinkUnauthorized):
		code = 403
	}
	if code == 404 || strings.Contains(msg, "item-not-found") ||
		strings.Contains(msg, "not-authorized") || strings.Contains(msg, "404") {
		return "no such group (or the account is not a member of it)"
	}
	if code == 401 || code == 403 || strings.Contains(msg, "forbidden") {
		return "not allowed — admin rights may be required"
	}
	if strings.Contains(msg, "rate") || strings.Contains(msg, "429") {
		return "WhatsApp rate-limited the request — try again shortly"
	}
	return truncate(msg, 200)
}

func (b *bridge) groupFail(cc *clientConn, id string, err error) {
	b.reply(cc, id, map[string]any{"ok": false, "error": friendlyError(err)})
}

// participantRow: role ” | 'admin' | 'superadmin'; the JID is the
// phone-number form when known (the old bridge normalized to it);
// names come from the local contact store, never invented.
func (b *bridge) participantRow(ctx context.Context, p types.GroupParticipant) map[string]any {
	jid := p.JID
	if !p.PhoneNumber.IsEmpty() {
		jid = p.PhoneNumber
	}
	name := p.DisplayName
	if contact, err := b.cli.Store.Contacts.GetContact(ctx, jid); err == nil && contact.Found {
		if contact.FullName != "" {
			name = contact.FullName
		} else if contact.PushName != "" {
			name = contact.PushName
		}
	}
	role := ""
	if p.IsSuperAdmin {
		role = "superadmin"
	} else if p.IsAdmin {
		role = "admin"
	}
	return map[string]any{
		"jid":  jid.ToNonAD().String(),
		"name": name,
		"role": role,
	}
}

// metaRow maps GroupInfo onto the wire shape. participants are
// excluded unless asked.
func (b *bridge) metaRow(ctx context.Context, meta *types.GroupInfo, withParticipants bool, invite string) map[string]any {
	size := meta.ParticipantCount
	if size == 0 {
		size = len(meta.Participants)
	}
	owner := meta.OwnerPN
	if owner.IsEmpty() {
		owner = meta.OwnerJID
	}
	admins := make([]string, 0)
	for _, p := range meta.Participants {
		if p.IsAdmin || p.IsSuperAdmin {
			aj := p.JID
			if !p.PhoneNumber.IsEmpty() {
				aj = p.PhoneNumber
			}
			admins = append(admins, aj.ToNonAD().String())
		}
	}
	ephemeralHours := 0
	if meta.DisappearingTimer > 0 {
		ephemeralHours = int((time.Duration(meta.DisappearingTimer)*time.Second + time.Hour/2) / time.Hour)
	}
	row := map[string]any{
		"id":                  meta.JID.String(),
		"subject":             meta.Name,
		"desc":                truncate(meta.Topic, 500),
		"size":                size,
		"creation":            meta.GroupCreated.Unix(),
		"owner":               owner.ToNonAD().String(),
		"isCommunity":         meta.IsParent,
		"isCommunityAnnounce": !meta.IsParent && !meta.LinkedParentJID.IsEmpty() && meta.IsDefaultSubGroup,
		"linkedParent":        meta.LinkedParentJID.String(),
		"announce":            meta.IsAnnounce,
		"restrict":            meta.IsLocked,
		"memberAddMode":       meta.MemberAddMode == types.GroupMemberAddModeAllMember,
		"joinApprovalMode":    meta.IsJoinApprovalRequired,
		"ephemeralHours":      ephemeralHours,
		"admins":              admins,
	}
	if withParticipants {
		parts := make([]map[string]any, 0, len(meta.Participants))
		for _, p := range meta.Participants {
			parts = append(parts, b.participantRow(ctx, p))
		}
		row["participants"] = parts
	}
	if invite != "" {
		row["invite"] = invite
	}
	return row
}

// ── read-only commands ────────────────────────────────────────────

func (b *bridge) handleGroupCommand(ctx context.Context, cc *clientConn, id string, req request) bool {
	switch req.str("cmd") {
	case "group_list":
		groups, err := b.cli.GetJoinedGroups(ctx)
		if err != nil {
			b.groupFail(cc, id, err)
			return true
		}
		rows := make([]map[string]any, 0, len(groups))
		for _, g := range groups {
			rows = append(rows, b.metaRow(ctx, g, false, ""))
		}
		sort.Slice(rows, func(i, j int) bool {
			return rows[i]["subject"].(string) < rows[j]["subject"].(string)
		})
		b.reply(cc, id, map[string]any{"ok": true, "groups": rows})
		return true

	case "group_info":
		jid, err := groupJID(req.str("chat"))
		if err != nil {
			b.reply(cc, id, map[string]any{"ok": false, "error": err.Error()})
			return true
		}
		meta, err := b.cli.GetGroupInfo(ctx, jid)
		if err != nil {
			b.groupFail(cc, id, err)
			return true
		}
		invite := ""
		if link, err := b.cli.GetGroupInviteLink(ctx, jid, false); err == nil {
			invite = link
		} else {
			logf("GetGroupInviteLink not permitted for %s - %s", jid, friendlyError(err))
		}
		b.reply(cc, id, map[string]any{"ok": true, "group": b.metaRow(ctx, meta, false, invite)})
		return true

	case "group_participants":
		jid, err := groupJID(req.str("chat"))
		if err != nil {
			b.reply(cc, id, map[string]any{"ok": false, "error": err.Error()})
			return true
		}
		meta, err := b.cli.GetGroupInfo(ctx, jid)
		if err != nil {
			b.groupFail(cc, id, err)
			return true
		}
		parts := make([]map[string]any, 0, len(meta.Participants))
		adminCount := 0
		for _, p := range meta.Participants {
			row := b.participantRow(ctx, p)
			if row["role"] != "" {
				adminCount++
			}
			parts = append(parts, row)
		}
		b.reply(cc, id, map[string]any{
			"ok": true,
			"group": map[string]any{
				"id": meta.JID.String(), "subject": meta.Name,
			},
			"count":        len(parts),
			"admins":       adminCount,
			"participants": parts,
		})
		return true

	// Pending join requests (DESIGN §28's request limb): who is
	// waiting at the door of an approval-mode group. whatsmeow
	// reports only the JID and the request time; the name comes
	// from the local contact store when it knows one — never
	// invented. A group whose approval mode is off simply has
	// no pending requests: an empty list, not an error.
	case "group_join_requests":
		jid, err := groupJID(req.str("chat"))
		if err != nil {
			b.reply(cc, id, map[string]any{"ok": false, "error": err.Error()})
			return true
		}
		reqs, err := b.cli.GetGroupRequestParticipants(ctx, jid)
		if err != nil {
			b.groupFail(cc, id, err)
			return true
		}
		rows := make([]map[string]any, 0, len(reqs))
		for _, r := range reqs {
			name := ""
			if contact, cerr := b.cli.Store.Contacts.GetContact(ctx, r.JID); cerr == nil && contact.Found {
				if contact.FullName != "" {
					name = contact.FullName
				} else if contact.PushName != "" {
					name = contact.PushName
				}
			}
			rows = append(rows, map[string]any{
				"jid":          r.JID.ToNonAD().String(),
				"name":         name,
				"requested_at": r.RequestedAt.Unix(),
			})
		}
		b.reply(cc, id, map[string]any{"ok": true, "requests": rows})
		return true

	case "community_list":
		all, err := b.cli.GetJoinedGroups(ctx)
		if err != nil {
			b.groupFail(cc, id, err)
			return true
		}
		communities := make([]map[string]any, 0)
		for _, c := range all {
			if !c.IsParent {
				continue
			}
			linked := make([]map[string]any, 0)
			for _, m := range all {
				if m.LinkedParentJID == c.JID {
					sz := m.ParticipantCount
					if sz == 0 {
						sz = len(m.Participants)
					}
					linked = append(linked, map[string]any{
						"id": m.JID.String(), "subject": m.Name, "size": sz,
					})
				}
			}
			sort.Slice(linked, func(i, j int) bool {
				return linked[i]["subject"].(string) < linked[j]["subject"].(string)
			})
			size := c.ParticipantCount
			if size == 0 {
				size = len(c.Participants)
			}
			communities = append(communities, map[string]any{
				"id": c.JID.String(), "subject": c.Name,
				"desc": truncate(c.Topic, 500), "size": size,
				"groups": linked,
			})
		}
		sort.Slice(communities, func(i, j int) bool {
			return communities[i]["subject"].(string) < communities[j]["subject"].(string)
		})
		b.reply(cc, id, map[string]any{"ok": true, "communities": communities})
		return true
	}
	return false
}

// ── mutating admin commands ───────────────────────────────────────

// participantResultRows shapes a per-participant outcome list
// onto the wire: the phone-number JID when known, "200" for a
// clean outcome, the WhatsApp error code otherwise; a call the
// server answered with no rows reads "sent" for everyone asked.
func participantResultRows(res []types.GroupParticipant, participants []types.JID) []map[string]any {
	rows := make([]map[string]any, 0, len(res))
	for _, r := range res {
		rj := r.JID
		if !r.PhoneNumber.IsEmpty() {
			rj = r.PhoneNumber
		}
		status := "200"
		if r.Error != 0 {
			status = strconv.Itoa(r.Error)
		}
		rows = append(rows, map[string]any{
			"jid": rj.ToNonAD().String(), "status": status,
		})
	}
	if len(rows) == 0 {
		for _, p := range participants {
			rows = append(rows, map[string]any{
				"jid": p.ToNonAD().String(), "status": "sent",
			})
		}
	}
	return rows
}

func (b *bridge) memberUpdateRows(ctx context.Context, jid types.JID, participants []types.JID, action whatsmeow.ParticipantChange, actionName string) (map[string]any, error) {
	res, err := b.cli.UpdateGroupParticipants(ctx, jid, participants, action)
	if err != nil {
		return nil, err
	}
	return map[string]any{"ok": true, "action": actionName, "results": participantResultRows(res, participants)}, nil
}

func (b *bridge) handleGroupAdminCommand(ctx context.Context, cc *clientConn, id string, req request) bool {
	cmd := req.str("cmd")
	ok := func(extra map[string]any) {
		out := map[string]any{"ok": true}
		for k, v := range extra {
			out[k] = v
		}
		b.reply(cc, id, out)
	}
	bad := func(err error) { b.reply(cc, id, map[string]any{"ok": false, "error": err.Error()}) }
	fail := func(err error) { b.groupFail(cc, id, err) }

	switch cmd {
	// ── groups ──
	case "group_create":
		subject := strings.TrimSpace(req.str("subject"))
		if subject == "" {
			bad(errors.New("missing subject for the new group"))
			return true
		}
		var participants []types.JID
		if req["participants"] != nil {
			var err error
			participants, err = participantJIDs(req["participants"])
			if err != nil {
				bad(err)
				return true
			}
		}
		meta, err := b.cli.CreateGroup(ctx, whatsmeow.ReqCreateGroup{
			Name:         subject,
			Participants: participants,
		})
		if err != nil {
			fail(err)
			return true
		}
		ok(map[string]any{"group": map[string]any{
			"id": meta.JID.String(), "subject": meta.Name,
		}})
		return true

	case "group_members_update":
		jid, err := groupJID(req.str("chat"))
		if err != nil {
			bad(err)
			return true
		}
		action, err := memberAction(req.str("action"))
		if err != nil {
			bad(err)
			return true
		}
		participants, err := participantJIDs(req["participants"])
		if err != nil {
			bad(err)
			return true
		}
		out, err := b.memberUpdateRows(ctx, jid, participants, action, string(action))
		if err != nil {
			fail(err)
			return true
		}
		b.reply(cc, id, out)
		return true

	case "group_set_subject":
		jid, err := groupJID(req.str("chat"))
		if err != nil {
			bad(err)
			return true
		}
		subject := strings.TrimSpace(req.str("subject"))
		if subject == "" {
			bad(errors.New("missing subject"))
			return true
		}
		if err := b.cli.SetGroupName(ctx, jid, subject); err != nil {
			fail(err)
			return true
		}
		ok(nil)
		return true

	case "group_set_description":
		jid, err := groupJID(req.str("chat"))
		if err != nil {
			bad(err)
			return true
		}
		if err := b.cli.SetGroupDescription(ctx, jid, req.str("description")); err != nil {
			fail(err)
			return true
		}
		ok(nil)
		return true

	case "group_set_picture":
		jid, err := groupJID(req.str("chat"))
		if err != nil {
			bad(err)
			return true
		}
		p := strings.TrimSpace(req.str("path"))
		if p == "" {
			bad(errors.New("missing path to the image"))
			return true
		}
		data, err := os.ReadFile(p)
		if err != nil {
			bad(errors.New("cannot read image at " + p))
			return true
		}
		if _, err := b.cli.SetGroupPhoto(ctx, jid, data); err != nil {
			fail(err)
			return true
		}
		ok(nil)
		return true

	case "group_leave":
		jid, err := groupJID(req.str("chat"))
		if err != nil {
			bad(err)
			return true
		}
		if err := b.cli.LeaveGroup(ctx, jid); err != nil {
			fail(err)
			return true
		}
		ok(nil)
		return true

	case "group_set_settings":
		jid, err := groupJID(req.str("chat"))
		if err != nil {
			bad(err)
			return true
		}
		changed := make([]string, 0, 2)
		if v, present := req["announce"]; present {
			announce, _ := v.(bool)
			if err := b.cli.SetGroupAnnounce(ctx, jid, announce); err != nil {
				fail(err)
				return true
			}
			changed = append(changed, "announce="+strconv.FormatBool(announce))
		}
		if v, present := req["restrict"]; present {
			restrict, _ := v.(bool)
			if err := b.cli.SetGroupLocked(ctx, jid, restrict); err != nil {
				fail(err)
				return true
			}
			changed = append(changed, "restrict="+strconv.FormatBool(restrict))
		}
		ok(map[string]any{"changed": changed})
		return true

	case "group_revoke_invite":
		jid, err := groupJID(req.str("chat"))
		if err != nil {
			bad(err)
			return true
		}
		link, err := b.cli.GetGroupInviteLink(ctx, jid, true)
		if err != nil {
			fail(err)
			return true
		}
		ok(map[string]any{"invite": link})
		return true

	// Join a group through its invite link (parity G2): the
	// resolve-first state machine. The preview read comes
	// first so a dead link fails BEFORE any join attempt —
	// whatsmeow wraps the dead-link errors
	// (ErrInviteLinkRevoked / ErrInviteLinkInvalid), and they
	// map to distinct reply lines: a revoked link and a
	// malformed one are different facts for the owner. The
	// preview's partial roster proves membership when it can;
	// when it can't (the roster is capped), one full member
	// read decides already_member authoritatively instead of
	// guessed. JoinGroupWithLink returns the same JID for a
	// real join and for a filed membership-approval request,
	// so after the accept the membership is RE-READ: present
	// is joined, absent is approval_pending — neither is ever
	// reported as the other.
	case "group_join_link":
		code := strings.TrimSpace(req.str("code"))
		if code == "" {
			bad(errors.New("missing code: pass the group invite code"))
			return true
		}
		me := []types.JID{}
		if b.cli.Store.ID != nil {
			me = append(me, *b.cli.Store.ID)
		}
		if lid := b.cli.Store.GetLID(); !lid.IsEmpty() {
			me = append(me, lid)
		}
		isMember := func(meta *types.GroupInfo) bool {
			for _, p := range meta.Participants {
				for _, own := range me {
					if p.JID.ToNonAD() == own.ToNonAD() ||
						p.PhoneNumber.ToNonAD() == own.ToNonAD() ||
						p.LID.ToNonAD() == own.ToNonAD() {
						return true
					}
				}
			}
			return false
		}
		inviteFail := func(err error) {
			switch {
			case errors.Is(err, whatsmeow.ErrInviteLinkRevoked):
				b.reply(cc, id, map[string]any{"ok": false,
					"error": "that invite link has been revoked"})
			case errors.Is(err, whatsmeow.ErrInviteLinkInvalid):
				b.reply(cc, id, map[string]any{"ok": false,
					"error": "that invite link is not valid"})
			default:
				fail(err)
			}
		}
		info, err := b.cli.GetGroupInfoFromLink(ctx, code)
		if err != nil {
			inviteFail(err)
			return true
		}
		if isMember(info) {
			ok(map[string]any{"state": "already_member",
				"chat": info.JID.String(), "subject": info.Name})
			return true
		}
		if full, ferr := b.cli.GetGroupInfo(ctx, info.JID); ferr == nil && isMember(full) {
			ok(map[string]any{"state": "already_member",
				"chat": info.JID.String(), "subject": info.Name})
			return true
		}
		jid, err := b.cli.JoinGroupWithLink(ctx, code)
		if err != nil {
			inviteFail(err)
			return true
		}
		if full, ferr := b.cli.GetGroupInfo(ctx, jid); ferr == nil && isMember(full) {
			ok(map[string]any{"state": "joined",
				"chat": jid.String(), "subject": info.Name})
			return true
		}
		ok(map[string]any{"state": "approval_pending",
			"chat": jid.String(), "subject": info.Name})
		return true

	case "group_join_approval":
		jid, err := groupJID(req.str("chat"))
		if err != nil {
			bad(err)
			return true
		}
		mode := strings.ToLower(strings.TrimSpace(req.str("mode")))
		if mode != "on" && mode != "off" {
			bad(errors.New("bad mode: " + req.str("mode") + " (want on|off)"))
			return true
		}
		if err := b.cli.SetGroupJoinApprovalMode(ctx, jid, mode == "on"); err != nil {
			fail(err)
			return true
		}
		ok(map[string]any{"join_approval": mode})
		return true

	// Decide pending join requests (DESIGN §28): approve admits
	// the requester, reject refuses them (they may request
	// again — a reject is not a ban). The reply carries the
	// per-participant rows, memberUpdateRows-shaped, so the
	// adapter honors each requester's own status.
	case "group_join_request_decide":
		jid, err := groupJID(req.str("chat"))
		if err != nil {
			bad(err)
			return true
		}
		var action whatsmeow.ParticipantRequestChange
		switch strings.ToLower(strings.TrimSpace(req.str("action"))) {
		case "approve":
			action = whatsmeow.ParticipantChangeApprove
		case "reject":
			action = whatsmeow.ParticipantChangeReject
		default:
			bad(errors.New("bad action: " + req.str("action") + " (want approve|reject)"))
			return true
		}
		participants, err := participantJIDs(req["participants"])
		if err != nil {
			bad(err)
			return true
		}
		res, err := b.cli.UpdateGroupRequestParticipants(ctx, jid, participants, action)
		if err != nil {
			fail(err)
			return true
		}
		b.reply(cc, id, map[string]any{
			"ok": true, "action": req.str("action"),
			"results": participantResultRows(res, participants),
		})
		return true

	case "group_member_add_mode":
		jid, err := groupJID(req.str("chat"))
		if err != nil {
			bad(err)
			return true
		}
		mode := strings.ToLower(strings.TrimSpace(req.str("mode")))
		var mam types.GroupMemberAddMode
		switch mode {
		case "admin":
			mam = types.GroupMemberAddModeAdmin
		case "all":
			mam = types.GroupMemberAddModeAllMember
		default:
			bad(errors.New("bad mode: " + req.str("mode") + " (want admin|all)"))
			return true
		}
		if err := b.cli.SetGroupMemberAddMode(ctx, jid, mam); err != nil {
			fail(err)
			return true
		}
		ok(map[string]any{"member_add_mode": mode})
		return true

	// ── message moderation (the runtime's admin_delete / admin_pin) ──
	// The message key needs the AUTHOR for someone else's message
	// (an admin revoke / a pin in a group). The bridge resolves the
	// author from its own message memory (cachedSender); a message
	// it never saw falls back to the own-message grammar for
	// delete (the server refuses honestly if that is wrong) and is
	// refused outright for pin, whose key cannot be guessed.

	case "message_delete":
		chat, err := parseJID(req.str("chat"))
		if err != nil || chat.IsEmpty() {
			bad(errors.New("missing chat: pass a chat JID"))
			return true
		}
		mid := strings.TrimSpace(req.str("message_id"))
		if mid == "" {
			bad(errors.New("missing message_id"))
			return true
		}
		sender := b.cachedSender(mid)
		_, err = b.cli.SendMessage(ctx, chat,
			b.cli.BuildRevoke(chat, sender, types.MessageID(mid)))
		if err != nil {
			fail(err)
			return true
		}
		ok(nil)
		return true

	case "message_pin", "message_unpin":
		jid, err := groupJID(req.str("chat"))
		if err != nil {
			bad(err)
			return true
		}
		mid := strings.TrimSpace(req.str("message_id"))
		if mid == "" {
			bad(errors.New("missing message_id"))
			return true
		}
		sender := b.cachedSender(mid)
		if sender.IsEmpty() {
			bad(errors.New("message not in bridge memory — " +
				"its author is unknown, so its key cannot be built"))
			return true
		}
		pinType := waE2E.PinInChatMessage_PIN_FOR_ALL
		if cmd == "message_unpin" {
			pinType = waE2E.PinInChatMessage_UNPIN_FOR_ALL
		}
		msg := &waE2E.Message{PinInChatMessage: &waE2E.PinInChatMessage{
			Key:               b.cli.BuildMessageKey(jid, sender, types.MessageID(mid)),
			Type:              pinType.Enum(),
			SenderTimestampMS: proto.Int64(time.Now().UnixMilli()),
		}}
		if _, err := b.cli.SendMessage(ctx, jid, msg); err != nil {
			fail(err)
			return true
		}
		ok(nil)
		return true

	// ── communities ──
	case "community_create":
		subject := strings.TrimSpace(req.str("subject"))
		if subject == "" {
			bad(errors.New("missing subject for the new community"))
			return true
		}
		meta, err := b.cli.CreateGroup(ctx, whatsmeow.ReqCreateGroup{
			Name:        subject,
			GroupParent: types.GroupParent{IsParent: true},
		})
		if err != nil {
			fail(err)
			return true
		}
		if desc := req.str("description"); desc != "" {
			// whatsmeow takes the description as a follow-up call
			// (Baileys took it in the create); a failure here is
			// logged, not hidden — the community exists either way.
			if err := b.cli.SetGroupDescription(ctx, meta.JID, desc); err != nil {
				logf("community description not set for %s: %v", meta.JID, err)
			}
		}
		ok(map[string]any{"community": map[string]any{
			"id": meta.JID.String(), "subject": subject,
		}})
		return true

	case "community_leave":
		jid, err := groupJID(req.str("chat"))
		if err != nil {
			bad(err)
			return true
		}
		if err := b.cli.LeaveGroup(ctx, jid); err != nil {
			fail(err)
			return true
		}
		ok(nil)
		return true

	case "community_members_update":
		jid, err := groupJID(req.str("chat"))
		if err != nil {
			bad(err)
			return true
		}
		action, err := memberAction(req.str("action"))
		if err != nil {
			bad(err)
			return true
		}
		participants, err := participantJIDs(req["participants"])
		if err != nil {
			bad(err)
			return true
		}
		out, err := b.memberUpdateRows(ctx, jid, participants, action, string(action))
		if err != nil {
			fail(err)
			return true
		}
		b.reply(cc, id, out)
		return true

	case "community_set_subject":
		jid, err := groupJID(req.str("chat"))
		if err != nil {
			bad(err)
			return true
		}
		subject := strings.TrimSpace(req.str("subject"))
		if subject == "" {
			bad(errors.New("missing subject"))
			return true
		}
		if err := b.cli.SetGroupName(ctx, jid, subject); err != nil {
			fail(err)
			return true
		}
		ok(nil)
		return true

	case "community_set_description":
		jid, err := groupJID(req.str("chat"))
		if err != nil {
			bad(err)
			return true
		}
		if err := b.cli.SetGroupDescription(ctx, jid, req.str("description")); err != nil {
			fail(err)
			return true
		}
		ok(nil)
		return true

	case "community_link_group":
		community, err := groupJID(req.str("community"))
		if err != nil {
			bad(err)
			return true
		}
		group, err := groupJID(req.str("group"))
		if err != nil {
			bad(err)
			return true
		}
		if err := b.cli.LinkGroup(ctx, community, group); err != nil {
			fail(err)
			return true
		}
		ok(map[string]any{"community": community.String(), "group": group.String()})
		return true

	case "community_unlink_group":
		community, err := groupJID(req.str("community"))
		if err != nil {
			bad(err)
			return true
		}
		group, err := groupJID(req.str("group"))
		if err != nil {
			bad(err)
			return true
		}
		if err := b.cli.UnlinkGroup(ctx, community, group); err != nil {
			fail(err)
			return true
		}
		ok(map[string]any{"community": community.String(), "group": group.String()})
		return true

	// ── channels (newsletters) ──
	case "channel_create":
		name := strings.TrimSpace(req.str("name"))
		if name == "" {
			bad(errors.New("missing name for the new channel"))
			return true
		}
		meta, err := b.cli.CreateNewsletter(ctx, whatsmeow.CreateNewsletterParams{
			Name:        name,
			Description: req.str("description"),
		})
		if err != nil {
			fail(err)
			return true
		}
		ok(map[string]any{"channel": map[string]any{
			"id": meta.ID.String(), "name": name,
		}})
		return true

	case "channel_delete":
		// whatsmeow (this pin) exposes no newsletter-delete verb.
		// Honest refusal, same guard style as the old module's
		// missing-verb errors — never a fake success.
		b.reply(cc, id, map[string]any{"ok": false, "error": "this bridge's whatsmeow build has no newsletter delete verb — channel deletion is unavailable"})
		return true

	case "channel_follow":
		jid, err := newsletterJID(req.str("chat"))
		if err != nil {
			bad(err)
			return true
		}
		if err := b.cli.FollowNewsletter(ctx, jid); err != nil {
			fail(err)
			return true
		}
		ok(nil)
		return true

	case "channel_unfollow":
		jid, err := newsletterJID(req.str("chat"))
		if err != nil {
			bad(err)
			return true
		}
		if err := b.cli.UnfollowNewsletter(ctx, jid); err != nil {
			fail(err)
			return true
		}
		ok(nil)
		return true

	case "channel_resolve":
		code := strings.TrimSpace(req.str("invite"))
		if code == "" {
			bad(errors.New("missing invite: pass the channel invite code"))
			return true
		}
		meta, err := b.cli.GetNewsletterInfoWithInvite(ctx, code)
		if err != nil {
			fail(err)
			return true
		}
		jid := meta.ID
		if jid.Server != types.NewsletterServer {
			bad(errors.New("channel invite did not resolve to a channel (got " + jid.String() + ")"))
			return true
		}
		ok(map[string]any{"chat": jid.String(), "name": meta.ThreadMeta.Name.Text})
		return true

	case "channel_info":
		jid, err := newsletterJID(req.str("chat"))
		if err != nil {
			bad(err)
			return true
		}
		meta, err := b.cli.GetNewsletterInfo(ctx, jid)
		if err != nil {
			fail(err)
			return true
		}
		// The follower count is the one number WhatsApp exposes
		// for a channel (the follower list is never enumerable).
		// Name/description ride along for the registry + console.
		ok(map[string]any{
			"name":        meta.ThreadMeta.Name.Text,
			"description": meta.ThreadMeta.Description.Text,
			"subscribers": meta.ThreadMeta.SubscriberCount,
		})
		return true

	case "channel_react":
		jid, err := newsletterJID(req.str("chat"))
		if err != nil {
			bad(err)
			return true
		}
		serverIDStr := strings.TrimSpace(req.str("server_id"))
		if serverIDStr == "" {
			serverIDStr = strings.TrimSpace(req.str("serverId"))
		}
		if serverIDStr == "" {
			bad(errors.New("missing server_id: the number at the end of the channel post link"))
			return true
		}
		serverID, err := strconv.ParseInt(serverIDStr, 10, 64)
		if err != nil {
			bad(errors.New("bad server_id: " + serverIDStr))
			return true
		}
		err = b.cli.NewsletterSendReaction(ctx, jid,
			types.MessageServerID(serverID), req.str("emoji"),
			b.cli.GenerateMessageID())
		if err != nil {
			fail(err)
			return true
		}
		ok(nil)
		return true

	case "channel_send":
		// Post to a channel the account admins (channel stories
		// build, 2026-11-07): a channel "story" on WhatsApp IS a
		// channel post — whatsmeow exposes no separate story
		// verb for newsletters, and SendMessage routes a
		// newsletter JID to sendNewsletter (send.go).  Text rides
		// a plain conversation message; media uploads through
		// UploadNewsletter (the newsletter media host — the
		// regular upload path produces chat-media URLs the
		// newsletter path rejects).  The reply carries BOTH ids:
		// the stanza id and the server_id — the post's public
		// identity (the number in its link, the key the view and
		// react verbs take).  A caller-supplied message_id is
		// used verbatim as the stanza id: the Python side derives
		// it deterministically from (channel, content, time
		// window), so an accidental double-fire of the same post
		// collapses onto one wire id instead of posting twice.
		jid, err := newsletterJID(req.str("chat"))
		if err != nil {
			bad(err)
			return true
		}
		text := strings.TrimSpace(req.str("text"))
		path := strings.TrimSpace(req.str("path"))
		caption := req.str("caption")
		if text == "" && path == "" {
			bad(errors.New("channel_send needs text or a media path"))
			return true
		}
		var msg *waE2E.Message
		if path != "" {
			data, err := os.ReadFile(path)
			if err != nil {
				b.reply(cc, id, map[string]any{"ok": false,
					"error": "media file not found: " + path})
				return true
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
					"error": "channel media must be an image or a " +
						"video: " + path})
				return true
			}
			up, err := b.cli.UploadNewsletter(ctx, data, mediaType)
			if err != nil {
				fail(err)
				return true
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
			msg = &waE2E.Message{Conversation: proto.String(text)}
		}
		extra := whatsmeow.SendRequestExtra{}
		if mid := strings.TrimSpace(req.str("message_id")); mid != "" {
			extra.ID = types.MessageID(mid)
		}
		resp, err := b.cli.SendMessage(ctx, jid, msg, extra)
		if err != nil {
			fail(err)
			return true
		}
		ok(map[string]any{
			"message_id": string(resp.ID),
			"server_id":  strconv.Itoa(resp.ServerID),
		})
		return true

	case "channel_view":
		// Mark channel posts viewed as the owner (channel
		// stories build): whatsmeow's NewsletterMarkViewed — a
		// "view" receipt naming the posts' server ids; this is
		// the verb that moves a channel post's view counter, the
		// channel counterpart of status_view's read receipt.
		jid, err := newsletterJID(req.str("chat"))
		if err != nil {
			bad(err)
			return true
		}
		var serverIDs []types.MessageServerID
		if raw, ok := req["server_ids"].([]any); ok {
			for _, v := range raw {
				s, _ := v.(string)
				if n, err := strconv.Atoi(strings.TrimSpace(s)); err == nil {
					serverIDs = append(serverIDs,
						types.MessageServerID(n))
				}
			}
		}
		if len(serverIDs) == 0 {
			if s := strings.TrimSpace(req.str("server_id")); s != "" {
				n, err := strconv.Atoi(s)
				if err != nil {
					bad(errors.New("bad server_id: " + s))
					return true
				}
				serverIDs = append(serverIDs,
					types.MessageServerID(n))
			}
		}
		if len(serverIDs) == 0 {
			bad(errors.New("missing server_id: the number at the end of the channel post link"))
			return true
		}
		if err := b.cli.NewsletterMarkViewed(ctx, jid, serverIDs); err != nil {
			fail(err)
			return true
		}
		ok(nil)
		return true

	case "channel_fetch":
		// The channel-post feed the personal surface's cycle
		// saves/views from (channel stories build): recent posts
		// of one channel (``chat`` given) or of EVERY channel the
		// account follows (no ``chat`` — whatsmeow's
		// GetSubscribedNewsletters enumerates them; the Baileys
		// rung has no enumeration verb and answers for one
		// channel at a time).  Rows carry the server_id (the
		// view/react key), the text, the timestamp, the view
		// count the server reports, and the media downloaded to
		// the media dir in the same {path, kind} shape status
		// posts arrive in — a post whose media could not be
		// downloaded says so (media_missing) instead of looking
		// text-only.
		count := int(req.num("count", 10))
		if count < 1 {
			count = 1
		}
		if count > 50 {
			count = 50
		}
		var chans []types.JID
		names := map[string]string{}
		if raw := strings.TrimSpace(req.str("chat")); raw != "" {
			jid, err := newsletterJID(raw)
			if err != nil {
				bad(err)
				return true
			}
			chans = []types.JID{jid}
		} else {
			subs, err := b.cli.GetSubscribedNewsletters(ctx)
			if err != nil {
				fail(err)
				return true
			}
			for _, meta := range subs {
				if meta == nil {
					continue
				}
				chans = append(chans, meta.ID)
				names[meta.ID.String()] = meta.ThreadMeta.Name.Text
			}
		}
		posts := make([]map[string]any, 0)
		for _, jid := range chans {
			msgs, err := b.cli.GetNewsletterMessages(ctx, jid,
				&whatsmeow.GetNewsletterMessagesParams{Count: count})
			if err != nil {
				// One dead channel must not sink the sweep:
				// the remaining channels still report.
				logf("channel_fetch %s: %v", jid, err)
				continue
			}
			for _, nm := range msgs {
				if nm == nil {
					continue
				}
				row := map[string]any{
					"chat":          jid.String(),
					"name":          names[jid.String()],
					"server_id":     strconv.Itoa(nm.MessageServerID),
					"message_id":    string(nm.MessageID),
					"text":          "",
					"kind":          "",
					"path":          "",
					"media_missing": false,
					"views":         nm.ViewsCount,
					"ts":            nm.Timestamp.UnixMilli(),
				}
				if nm.Message != nil {
					row["text"] = textOf(nm.Message)
					if m := b.downloadMediaOf(ctx, nm.Message,
						string(nm.MessageID)); m != nil {
						row["kind"] = m["kind"]
						row["path"] = m["path"]
						if m["path"] == "" {
							row["media_missing"] = true
						}
					} else if mediaKindOf(nm.Message) != "" {
						row["kind"] = mediaKindOf(nm.Message)
						row["media_missing"] = true
					}
				}
				posts = append(posts, row)
			}
		}
		ok(map[string]any{"posts": posts})
		return true

	case "channel_set_name":
		// No newsletter-rename verb in this whatsmeow pin.
		b.reply(cc, id, map[string]any{"ok": false, "error": "this bridge's whatsmeow build has no newsletter rename verb — channel rename is unavailable"})
		return true

	case "channel_set_description":
		// No newsletter-description verb in this whatsmeow pin.
		b.reply(cc, id, map[string]any{"ok": false, "error": "this bridge's whatsmeow build has no newsletter description verb — channel description change is unavailable"})
		return true
	}
	return false
}
