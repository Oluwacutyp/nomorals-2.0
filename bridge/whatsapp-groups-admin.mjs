#!/usr/bin/env node
/**
 * whatsapp-groups-admin.mjs — MUTATING group/community/channel commands for
 * Devon's WhatsApp bridge (bridge/whatsapp-bridge.mjs).
 *
 * Companion to whatsapp-groups.mjs, which is deliberately READ-ONLY. This
 * module owns every mutation: create, add/remove/promote/demote, subject,
 * description, picture, settings, leave, communities, newsletters.
 *
 * These are personal-account mutations. WhatsApp may flag aggressive use —
 * every handler is owner-confirmed on the Python side (spine tools carry
 * the owner's grant; the public/private matrix denies outsiders). Nothing
 * here fires on its own.
 *
 * Protocol (python -> bridge):
 *   Groups:
 *     {"id":"<req>","cmd":"group_create","subject":"...","participants":["<jid>",...]}
 *     {"id":"<req>","cmd":"group_members_update","chat":"<@g.us>","action":"add|remove|promote|demote","participants":["<jid>",...]}
 *     {"id":"<req>","cmd":"group_set_subject","chat":"<@g.us>","subject":"..."}
 *     {"id":"<req>","cmd":"group_set_description","chat":"<@g.us>","description":"..."}
 *     {"id":"<req>","cmd":"group_set_picture","chat":"<@g.us>","path":"<local image path>"}
 *     {"id":"<req>","cmd":"group_leave","chat":"<@g.us>"}
 *     {"id":"<req>","cmd":"group_join_link","code":"<invite code>"}
 *     {"id":"<req>","cmd":"group_set_settings","chat":"<@g.us>","announce":bool,"restrict":bool}
 *     {"id":"<req>","cmd":"group_revoke_invite","chat":"<@g.us>"}
 *     {"id":"<req>","cmd":"group_join_approval","chat":"<@g.us>","mode":"on|off"}
 *     {"id":"<req>","cmd":"group_join_request_decide","chat":"<@g.us>","action":"approve|reject","participants":["<jid>",...]}
 *     {"id":"<req>","cmd":"group_member_add_mode","chat":"<@g.us>","mode":"admin|all"}
 *     {"id":"<req>","cmd":"message_delete","chat":"<@g.us>","message_id":"<stanza id>"}
 *     {"id":"<req>","cmd":"message_pin","chat":"<@g.us>","message_id":"<stanza id>"}
 *     {"id":"<req>","cmd":"message_unpin","chat":"<@g.us>","message_id":"<stanza id>"}
 *   Communities:
 *     {"id":"<req>","cmd":"community_create","subject":"...","description":"..."}
 *     {"id":"<req>","cmd":"community_leave","chat":"<@g.us>"}
 *     {"id":"<req>","cmd":"community_members_update","chat":"<@g.us>","action":"add|remove|promote|demote","participants":["<jid>",...]}
 *     {"id":"<req>","cmd":"community_set_subject","chat":"<@g.us>","subject":"..."}
 *     {"id":"<req>","cmd":"community_set_description","chat":"<@g.us>","description":"..."}
 *     {"id":"<req>","cmd":"community_link_group","community":"<@g.us>","group":"<@g.us>"}
 *     {"id":"<req>","cmd":"community_unlink_group","community":"<@g.us>","group":"<@g.us>"}
 *   Channels (newsletters):
 *     {"id":"<req>","cmd":"channel_create","name":"...","description":"..."}
 *     {"id":"<req>","cmd":"channel_delete","chat":"<@newsletter>"}
 *     {"id":"<req>","cmd":"channel_follow","chat":"<@newsletter>"}
 *     {"id":"<req>","cmd":"channel_unfollow","chat":"<@newsletter>"}
 *     {"id":"<req>","cmd":"channel_resolve","invite":"<channel invite code>"}
 *     {"id":"<req>","cmd":"channel_react","chat":"<@newsletter>","server_id":"<post id>","emoji":"🔥"}
 *     {"id":"<req>","cmd":"channel_send","chat":"<@newsletter>","text":"...","path":"<media file>","caption":"...","message_id":"<optional stanza id>"}
 *     {"id":"<req>","cmd":"channel_fetch","chat":"<@newsletter>","count":10}
 *     {"id":"<req>","cmd":"channel_view","chat":"<@newsletter>","server_id":"<post id>"}   (honest refusal — no Baileys verb)
 *     {"id":"<req>","cmd":"channel_set_name","chat":"<@newsletter>","name":"..."}
 *     {"id":"<req>","cmd":"channel_set_description","chat":"<@newsletter>","description":"..."}
 *
 * Baileys surface (verified against @whiskeysockets/baileys 6.7.18):
 *   lib/Socket/groups.js      — groupCreate, groupParticipantsUpdate,
 *                                groupUpdateSubject, groupUpdateDescription,
 *                                groupLeave, groupSettingUpdate,
 *                                groupRevokeInvite, groupJoinApprovalMode,
 *                                groupRequestParticipantsUpdate,
 *                                groupMemberAddMode
 *   lib/Socket/chats.js       — updateProfilePicture (group JIDs too)
 *   lib/Utils/messages.js     — pinInChatMessage, built by
 *                                sendMessage from {pin: key, type}
 *                                (1 = pin for all, 2 = unpin for all)
 *   lib/Socket/communities.js — communityCreate, communityLeave,
 *                                communityParticipantsUpdate,
 *                                communityUpdateSubject,
 *                                communityUpdateDescription
 *
 * Usage from whatsapp-bridge.mjs:
 *   import registerGroupAdminCommands from './whatsapp-groups-admin.mjs';
 *   // in handleCommand's default: case, after registerGroupCommands:
 *   return (await registerGroupCommands(...)(sock, req))
 *       || (await registerGroupAdminCommands({ getSocket: () => wa, reply, log })(sock, req))
 *       || reply(sock, id, { ok: false, error: `unknown cmd: ${cmd}` });
 */

import fs from 'node:fs';
import { jidNormalizedUser, proto } from '@whiskeysockets/baileys';

const isGroupJid = (jid) => typeof jid === 'string' && jid.endsWith('@g.us');
const isNewsletterJid = (jid) => typeof jid === 'string' && jid.endsWith('@newsletter');

/** Normalize a chat argument to a group JID. Honest errors, no guessing. */
function groupJid(raw) {
  const jid = jidNormalizedUser(String(raw || '').trim());
  if (!jid) throw new Error('missing group: pass a group JID (…@g.us)');
  if (!isGroupJid(jid)) throw new Error(`not a group JID: ${raw} (groups end in @g.us)`);
  return jid;
}

function newsletterJid(raw) {
  const jid = jidNormalizedUser(String(raw || '').trim());
  if (!jid) throw new Error('missing channel: pass a channel JID (…@newsletter)');
  if (!isNewsletterJid(jid)) throw new Error(`not a channel JID: ${raw} (channels end in @newsletter)`);
  return jid;
}

/** Normalize participant JIDs (user JIDs, not groups). */
function participantJids(raw) {
  const list = Array.isArray(raw) ? raw : [raw];
  const out = list.map((p) => jidNormalizedUser(String(p || '').trim())).filter(Boolean);
  if (!out.length) throw new Error('no participants: pass at least one user JID');
  return out;
}

const ACTIONS = ['add', 'remove', 'promote', 'demote'];
function memberAction(raw) {
  const a = String(raw || '').trim().toLowerCase();
  if (!ACTIONS.includes(a)) throw new Error(`bad action: ${raw} (want one of ${ACTIONS.join('|')})`);
  return a;
}

/** Turn Baileys/Boom failures into owner-readable errors. */
function friendlyError(e) {
  const msg = e?.message || String(e);
  const code = e?.output?.statusCode;
  if (code === 404 || /not-authorized|item-not-found|404/.test(msg)) {
    return 'no such group (or the account is not a member of it)';
  }
  if (code === 401 || code === 403) return 'not allowed — admin rights may be required';
  if (/rate|429/.test(msg)) return 'WhatsApp rate-limited the request — try again shortly';
  return msg.slice(0, 200);
}

export function registerGroupAdminCommands(ctx) {
  const { getSocket, reply, log, getCachedMessage } = ctx;
  const sock = () => {
    const s = getSocket();
    if (!s) throw new Error('whatsapp not connected yet');
    return s;
  };

  const handlers = {
    // ── groups ──────────────────────────────────────────────────────

    group_create: async (req) => {
      const subject = String(req.subject || '').trim();
      if (!subject) throw new Error('missing subject for the new group');
      const participants = req.participants ? participantJids(req.participants) : [];
      const meta = await sock().groupCreate(subject, participants);
      return { ok: true, group: { id: meta.id || '', subject: meta.subject || subject } };
    },

    group_members_update: async (req) => {
      const jid = groupJid(req.chat);
      const action = memberAction(req.action);
      const participants = participantJids(req.participants);
      const res = await sock().groupParticipantsUpdate(jid, participants, action);
      const rows = (Array.isArray(res) ? res : []).map((r) => ({
        jid: jidNormalizedUser(r.jid || ''),
        status: String(r.status || ''),
      }));
      return { ok: true, action, results: rows.length ? rows : participants.map((p) => ({ jid: p, status: 'sent' })) };
    },

    group_set_subject: async (req) => {
      const jid = groupJid(req.chat);
      const subject = String(req.subject || '').trim();
      if (!subject) throw new Error('missing subject');
      await sock().groupUpdateSubject(jid, subject);
      return { ok: true };
    },

    group_set_description: async (req) => {
      const jid = groupJid(req.chat);
      await sock().groupUpdateDescription(jid, String(req.description || ''));
      return { ok: true };
    },

    group_set_picture: async (req) => {
      const jid = groupJid(req.chat);
      const p = String(req.path || '').trim();
      if (!p) throw new Error('missing path to the image');
      let buf;
      try {
        buf = fs.readFileSync(p);
      } catch {
        throw new Error(`cannot read image at ${p}`);
      }
      await sock().updateProfilePicture(jid, { img: buf });
      return { ok: true };
    },

    group_leave: async (req) => {
      const jid = groupJid(req.chat);
      await sock().groupLeave(jid);
      return { ok: true };
    },

    group_join_link: async (req) => {
      // Join a group through its invite link (parity G2).
      // Resolve-first: groupGetInviteInfo reads the group
      // behind the code BEFORE any join attempt, so a dead
      // link fails distinctly — 410 means the link was
      // revoked, 401/404 mean it is not a live link at all.
      // groupAcceptInvite returns the group JID only for a
      // REAL join; for an approval-mode group it files the
      // request and returns nothing — so the landed state
      // hinges on a membership re-read (groupMetadata),
      // never on the accept's return value alone, and a
      // failed re-read after a landed accept reports the
      // conservative approval_pending rather than claiming
      // joined.
      const code = String(req.code || '').trim();
      if (!code) throw new Error('missing code: pass the group invite code');
      const wa = sock();
      const me = wa.user && wa.user.id ? jidNormalizedUser(wa.user.id) : '';
      let info;
      try {
        info = await wa.groupGetInviteInfo(code);
      } catch (e) {
        const status = e && e.output && e.output.statusCode;
        if (status === 410) throw new Error('that invite link has been revoked');
        if (status === 401 || status === 404) throw new Error('that invite link is not valid');
        throw e;
      }
      const jid = info && info.id;
      if (!isGroupJid(jid)) throw new Error('that invite link did not resolve to a group');
      const subject = info.subject || '';
      const memberOf = (meta) => !!me && !!meta && Array.isArray(meta.participants)
        && meta.participants.some((p) => jidNormalizedUser(p.id || p.jid || '') === me);
      if (memberOf(info)) {
        return { ok: true, state: 'already_member', chat: jid, subject };
      }
      await wa.groupAcceptInvite(code);
      let meta = null;
      try {
        meta = await wa.groupMetadata(jid);
      } catch {
        meta = null;
      }
      if (memberOf(meta)) {
        return { ok: true, state: 'joined', chat: jid, subject };
      }
      return { ok: true, state: 'approval_pending', chat: jid, subject };
    },

    group_set_settings: async (req) => {
      const jid = groupJid(req.chat);
      const wa = sock();
      const changed = [];
      if (req.announce !== undefined) {
        await wa.groupSettingUpdate(jid, req.announce ? 'announcement' : 'not_announcement');
        changed.push(`announce=${!!req.announce}`);
      }
      if (req.restrict !== undefined) {
        await wa.groupSettingUpdate(jid, req.restrict ? 'locked' : 'unlocked');
        changed.push(`restrict=${!!req.restrict}`);
      }
      return { ok: true, changed };
    },

    group_revoke_invite: async (req) => {
      const jid = groupJid(req.chat);
      const code = await sock().groupRevokeInvite(jid);
      return { ok: true, invite: code ? `https://chat.whatsapp.com/${code}` : '' };
    },

    group_join_approval: async (req) => {
      const jid = groupJid(req.chat);
      const mode = String(req.mode || '').trim().toLowerCase();
      if (!['on', 'off'].includes(mode)) throw new Error(`bad mode: ${req.mode} (want on|off)`);
      await sock().groupJoinApprovalMode(jid, mode);
      return { ok: true, join_approval: mode };
    },

    group_member_add_mode: async (req) => {
      const jid = groupJid(req.chat);
      const mode = String(req.mode || '').trim().toLowerCase();
      if (!['admin', 'all'].includes(mode)) throw new Error(`bad mode: ${req.mode} (want admin|all)`);
      await sock().groupMemberAddMode(jid, mode);
      return { ok: true, member_add_mode: mode };
    },

    // Decide pending join requests (DESIGN §28): approve admits
    // the requester, reject refuses them (they may request
    // again — a reject is not a ban). The reply carries the
    // per-participant rows, group_members_update-shaped, so the
    // adapter honors each requester's own status.
    group_join_request_decide: async (req) => {
      const jid = groupJid(req.chat);
      const action = String(req.action || '').trim().toLowerCase();
      if (!['approve', 'reject'].includes(action)) {
        throw new Error(`bad action: ${req.action} (want approve|reject)`);
      }
      const participants = participantJids(req.participants);
      const res = await sock().groupRequestParticipantsUpdate(jid, participants, action);
      const rows = (Array.isArray(res) ? res : []).map((r) => ({
        jid: jidNormalizedUser(r.jid || ''),
        status: String(r.status || ''),
      }));
      return { ok: true, action, results: rows.length ? rows : participants.map((p) => ({ jid: p, status: 'sent' })) };
    },

    // ── message moderation (the runtime's admin_delete / admin_pin) ──
    // Delete-for-everyone is Baileys' sendMessage with a delete key;
    // the key (author, fromMe, participant) comes from the bridge's
    // own message memory — a message it never saw is an honest
    // refusal, never a guessed key. Pinning rides the same memory:
    // sendMessage with a pin key builds the protocol's
    // pinInChatMessage (lib/Utils/messages.js: 'pin' in message →
    // {key, type, senderTimestampMs}, relayed with the pin edit
    // attribute by lib/Socket/messages-send.js) — type 1 pins for
    // all, type 2 unpins for all, the same message the Go engine
    // sends via whatsmeow's PinInChatMessage.

    message_delete: async (req) => {
      const jid = groupJid(req.chat);
      const mid = String(req.message_id || '').trim();
      if (!mid) throw new Error('missing message_id');
      const cached = typeof getCachedMessage === 'function'
        ? getCachedMessage(mid) : null;
      if (!cached || !cached.key) {
        throw new Error('message not in bridge memory — its key cannot be built');
      }
      await sock().sendMessage(jid, { delete: cached.key });
      return { ok: true };
    },

    message_pin: async (req) => {
      const jid = groupJid(req.chat);
      const mid = String(req.message_id || '').trim();
      if (!mid) throw new Error('missing message_id');
      const cached = typeof getCachedMessage === 'function'
        ? getCachedMessage(mid) : null;
      if (!cached || !cached.key) {
        throw new Error('message not in bridge memory — its key cannot be built');
      }
      await sock().sendMessage(jid, { pin: cached.key, type: 1 });
      return { ok: true };
    },

    message_unpin: async (req) => {
      const jid = groupJid(req.chat);
      const mid = String(req.message_id || '').trim();
      if (!mid) throw new Error('missing message_id');
      const cached = typeof getCachedMessage === 'function'
        ? getCachedMessage(mid) : null;
      if (!cached || !cached.key) {
        throw new Error('message not in bridge memory — its key cannot be built');
      }
      await sock().sendMessage(jid, { pin: cached.key, type: 2 });
      return { ok: true };
    },

    // ── communities ─────────────────────────────────────────────────

    community_create: async (req) => {
      const subject = String(req.subject || '').trim();
      if (!subject) throw new Error('missing subject for the new community');
      const meta = await sock().communityCreate(subject, String(req.description || ''));
      return { ok: true, community: { id: (meta && meta.id) || '', subject } };
    },

    community_leave: async (req) => {
      const jid = groupJid(req.chat);
      await sock().communityLeave(jid);
      return { ok: true };
    },

    community_members_update: async (req) => {
      const jid = groupJid(req.chat);
      const action = memberAction(req.action);
      const participants = participantJids(req.participants);
      const res = await sock().communityParticipantsUpdate(jid, participants, action);
      const rows = (Array.isArray(res) ? res : []).map((r) => ({
        jid: jidNormalizedUser(r.jid || ''),
        status: String(r.status || ''),
      }));
      return { ok: true, action, results: rows.length ? rows : participants.map((p) => ({ jid: p, status: 'sent' })) };
    },

    community_set_subject: async (req) => {
      const jid = groupJid(req.chat);
      const subject = String(req.subject || '').trim();
      if (!subject) throw new Error('missing subject');
      await sock().communityUpdateSubject(jid, subject);
      return { ok: true };
    },

    community_set_description: async (req) => {
      const jid = groupJid(req.chat);
      await sock().communityUpdateDescription(jid, String(req.description || ''));
      return { ok: true };
    },

    // Link a subgroup into a community: the Baileys 6.7.x pattern is adding
    // the subgroup JID as a participant of the community with action 'add'.
    community_link_group: async (req) => {
      const community = groupJid(req.community);
      const group = groupJid(req.group);
      await sock().groupParticipantsUpdate(community, [group], 'add');
      return { ok: true, community, group };
    },

    community_unlink_group: async (req) => {
      const community = groupJid(req.community);
      const group = groupJid(req.group);
      await sock().groupParticipantsUpdate(community, [group], 'remove');
      return { ok: true, community, group };
    },

    // ── channels (newsletters) ──────────────────────────────────────

    channel_create: async (req) => {
      const name = String(req.name || '').trim();
      if (!name) throw new Error('missing name for the new channel');
      const meta = await sock().newsletterCreate(name, String(req.description || ''));
      return { ok: true, channel: { id: (meta && meta.id) || '', name } };
    },

    channel_delete: async (req) => {
      const jid = newsletterJid(req.chat);
      await sock().newsletterDelete(jid);
      return { ok: true };
    },

    channel_follow: async (req) => {
      const jid = newsletterJid(req.chat);
      await sock().newsletterFollow(jid);
      return { ok: true };
    },

    channel_unfollow: async (req) => {
      const jid = newsletterJid(req.chat);
      await sock().newsletterUnfollow(jid);
      return { ok: true };
    },

    channel_resolve: async (req) => {
      // A whatsapp.com/channel/<code> link carries an INVITE
      // CODE, not the channel JID. Baileys resolves it through
      // newsletterMetadata('invite', code) — the metadata's id
      // IS the JID. Guarded like every verb here: a build
      // without it gets an honest error, not a guess.
      const code = String(req.invite || '').trim();
      if (!code) throw new Error('missing invite: pass the channel invite code');
      if (typeof sock().newsletterMetadata !== 'function') {
        throw new Error('this Baileys build has no newsletterMetadata verb — channel links cannot be resolved; pass the …@newsletter JID instead');
      }
      const meta = await sock().newsletterMetadata('invite', code);
      const jid = (meta && (meta.id || meta.jid)) || '';
      if (!isNewsletterJid(jid)) {
        throw new Error(`channel invite did not resolve to a channel (got ${jid || 'nothing'})`);
      }
      return { ok: true, chat: jid, name: (meta && meta.name) || '' };
    },

    channel_info: async (req) => {
      // The count half of the channel system: a channel's
      // follower count comes from its newsletter metadata
      // ('jid' form) — the follower LIST is never enumerable
      // on WhatsApp, a count is all the platform exposes.
      // Guarded like channel_resolve: a build without the
      // verb gets an honest error, never a guessed number.
      const jid = newsletterJid(req.chat);
      if (typeof sock().newsletterMetadata !== 'function') {
        throw new Error('this Baileys build has no newsletterMetadata verb — channel info is unavailable');
      }
      const meta = await sock().newsletterMetadata('jid', jid);
      if (!meta) throw new Error('channel metadata came back empty');
      const subs = meta.subscribers ?? meta.subscriber_count ?? null;
      return { ok: true, name: meta.name || '',
               description: meta.description || '',
               subscribers: subs === null ? null : Number(subs) };
    },

    channel_react: async (req) => {
      // The §14 mass-react mechanism for WhatsApp channels
      // (DESIGN §16): followers interact with a channel ONLY by
      // reacting, and newsletterReactMessage is the verb. The
      // server_id is the number at the end of the post link
      // (whatsapp.com/channel/<code>/<server_id>).
      const jid = newsletterJid(req.chat);
      const serverId = String(req.server_id || req.serverId || '').trim();
      if (!serverId) throw new Error('missing server_id: the number at the end of the channel post link');
      if (typeof sock().newsletterReactMessage !== 'function') {
        throw new Error('this Baileys build has no newsletterReactMessage verb — channel reactions unavailable');
      }
      await sock().newsletterReactMessage(jid, serverId, String(req.emoji || ''));
      return { ok: true };
    },

    channel_send: async (req) => {
      // Post to a channel the account admins (channel stories
      // build, 2026-11-07): on WhatsApp a channel "story" IS a
      // channel post — Baileys has no separate story verb for
      // newsletters; sendMessage to the newsletter JID is the
      // post.  Media rides as a buffer (image/video only — the
      // shapes channels carry).  The reply carries the stanza
      // id; Baileys' send result exposes no server_id, so none
      // is reported (absence, never an invented one).  A
      // caller-supplied message_id is passed through as the
      // message key id — the Python side derives it
      // deterministically so an accidental double-fire of the
      // same post collapses onto one wire id.
      const jid = newsletterJid(req.chat);
      const text = String(req.text || '').trim();
      const path = String(req.path || '').trim();
      const caption = String(req.caption || '');
      if (!text && !path) throw new Error('channel_send needs text or a media path');
      let content;
      if (path) {
        if (!fs.existsSync(path)) throw new Error(`media file not found: ${path}`);
        const lower = path.toLowerCase();
        const buf = fs.readFileSync(path);
        const cap = caption || text || undefined;
        if (/\.(jpg|jpeg|png|webp)$/.test(lower)) {
          content = { image: buf, caption: cap };
        } else if (/\.(mp4|mov|gif)$/.test(lower)) {
          content = { video: buf, caption: cap, gifPlayback: lower.endsWith('.gif') };
        } else {
          throw new Error(`channel media must be an image or a video: ${path}`);
        }
      } else {
        content = { text };
      }
      const opts = {};
      const mid = String(req.message_id || '').trim();
      if (mid) opts.messageId = mid;
      const sent = await sock().sendMessage(jid, content, opts);
      return { ok: true, message_id: (sent && sent.key && sent.key.id) || '' };
    },

    channel_fetch: async (req) => {
      // The channel-post feed the personal surface's cycle
      // saves from (channel stories build): Baileys'
      // newsletterFetchMessages returns the RAW iq node; the
      // rows are walked here into the same shape the Go rung
      // emits ({chat, server_id, text, kind, path, ts, views}).
      // Two honest limits of this rung: it has NO subscribed-
      // channel enumeration verb, so a channel JID is required
      // (the Python side enumerates from its known channel
      // chats); and media bytes are not downloaded here — a
      // media post reports its kind with media_missing set
      // rather than pretending to be text.
      const jid = newsletterJid(req.chat);
      if (typeof sock().newsletterFetchMessages !== 'function') {
        throw new Error('this Baileys build has no newsletterFetchMessages verb — channel posts cannot be fetched');
      }
      const count = Math.max(1, Math.min(50, parseInt(req.count || '10', 10) || 10));
      const result = await sock().newsletterFetchMessages(jid, count);
      const posts = [];
      const container = (result && Array.isArray(result.content))
        ? result.content.find((n) => n && n.tag === 'messages')
        : null;
      const children = (container && Array.isArray(container.content))
        ? container.content : [];
      const textOfDecoded = (msg) => {
        if (!msg) return { text: '', kind: '' };
        if (typeof msg.conversation === 'string' && msg.conversation) {
          return { text: msg.conversation, kind: '' };
        }
        if (msg.extendedTextMessage && msg.extendedTextMessage.text) {
          return { text: String(msg.extendedTextMessage.text), kind: '' };
        }
        if (msg.imageMessage) {
          return { text: String(msg.imageMessage.caption || ''), kind: 'image' };
        }
        if (msg.videoMessage) {
          return { text: String(msg.videoMessage.caption || ''), kind: 'video' };
        }
        if (msg.audioMessage) return { text: '', kind: 'audio' };
        if (msg.stickerMessage) return { text: '', kind: 'sticker' };
        if (msg.documentMessage) return { text: '', kind: 'file' };
        return { text: '', kind: '' };
      };
      for (const node of children) {
        if (!node || node.tag !== 'message') continue;
        const attrs = node.attrs || {};
        let text = '';
        let kind = '';
        let mediaMissing = false;
        let views = 0;
        for (const sub of (Array.isArray(node.content) ? node.content : [])) {
          if (!sub) continue;
          if (sub.tag === 'plaintext' && sub.content) {
            try {
              const decoded = proto.Message.decode(
                Buffer.isBuffer(sub.content) ? sub.content : Buffer.from(sub.content));
              const got = textOfDecoded(decoded);
              text = got.text;
              kind = got.kind;
              if (kind) mediaMissing = true;
            } catch { /* an undecodable post still reports its id */ }
          } else if (sub.tag === 'views_count') {
            views = parseInt((sub.attrs || {}).count || '0', 10) || 0;
          }
        }
        posts.push({
          chat: jid, name: '',
          server_id: String(attrs.server_id || ''),
          message_id: String(attrs.id || ''),
          text, kind, path: '', media_missing: mediaMissing,
          views, ts: (parseInt(attrs.t || '0', 10) || 0) * 1000,
        });
      }
      return { ok: true, posts };
    },

    channel_view: async () => {
      // The wall, named (channel stories build): marking a
      // channel post viewed is a "view" receipt keyed on
      // server ids — whatsmeow's NewsletterMarkViewed.  Baileys
      // 6.7.24's newsletter socket (verified against
      // lib/Socket/newsletter.d.ts) exposes create/update/
      // follow/react/fetch verbs and NO mark-viewed equivalent,
      // so this rung cannot move a post's view counter.  An
      // honest ok:false, never a fake success.
      return { ok: false,
               error: 'this Baileys build has no newsletter mark-viewed verb — ' +
                      'channel view receipts are unavailable on this engine ' +
                      '(the Go engine carries them)' };
    },

    channel_set_name: async (req) => {
      const jid = newsletterJid(req.chat);
      const name = String(req.name || '').trim();
      if (!name) throw new Error('missing name');
      await sock().newsletterUpdateName(jid, name);
      return { ok: true };
    },

    channel_set_description: async (req) => {
      const jid = newsletterJid(req.chat);
      await sock().newsletterUpdateDescription(jid, String(req.description || ''));
      return { ok: true };
    },
  };

  /** Returns true when this module handled the command (replied itself). */
  return async (sockArg, req) => {
    const fn = handlers[req && req.cmd];
    if (!fn) return false;
    try {
      reply(sockArg, req.id, await fn(req));
    } catch (e) {
      log(`group-admin cmd ${req.cmd} failed:`, e?.message || e);
      reply(sockArg, req.id, { ok: false, error: friendlyError(e) });
    }
    return true;
  };
}

export default registerGroupAdminCommands;
