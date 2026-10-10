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
 *     {"id":"<req>","cmd":"group_set_settings","chat":"<@g.us>","announce":bool,"restrict":bool}
 *     {"id":"<req>","cmd":"group_revoke_invite","chat":"<@g.us>"}
 *     {"id":"<req>","cmd":"group_join_approval","chat":"<@g.us>","mode":"on|off"}
 *     {"id":"<req>","cmd":"group_member_add_mode","chat":"<@g.us>","mode":"admin|all"}
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
 *     {"id":"<req>","cmd":"channel_set_name","chat":"<@newsletter>","name":"..."}
 *     {"id":"<req>","cmd":"channel_set_description","chat":"<@newsletter>","description":"..."}
 *
 * Baileys surface (verified against @whiskeysockets/baileys 6.7.18):
 *   lib/Socket/groups.js      — groupCreate, groupParticipantsUpdate,
 *                                groupUpdateSubject, groupUpdateDescription,
 *                                groupLeave, groupSettingUpdate,
 *                                groupRevokeInvite, groupJoinApprovalMode,
 *                                groupMemberAddMode
 *   lib/Socket/chats.js       — updateProfilePicture (group JIDs too)
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
import { jidNormalizedUser } from '@whiskeysockets/baileys';

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
  const { getSocket, reply, log } = ctx;
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
