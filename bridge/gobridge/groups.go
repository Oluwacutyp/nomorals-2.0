// groups.go — group / community / channel commands for the
// adapter protocol: the read-only set (group_list, group_info,
// group_participants, community_list) and the mutating admin set
// (create / members / subject / description / picture / settings /
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
	"sort"
	"strconv"
	"strings"
	"time"

	"go.mau.fi/whatsmeow"
	"go.mau.fi/whatsmeow/types"
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

func (b *bridge) memberUpdateRows(ctx context.Context, jid types.JID, participants []types.JID, action whatsmeow.ParticipantChange, actionName string) (map[string]any, error) {
	res, err := b.cli.UpdateGroupParticipants(ctx, jid, participants, action)
	if err != nil {
		return nil, err
	}
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
	return map[string]any{"ok": true, "action": actionName, "results": rows}, nil
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
