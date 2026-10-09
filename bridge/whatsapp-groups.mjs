#!/usr/bin/env node
/**
 * whatsapp-groups.mjs — Baileys group/community command handlers for Devon's
 * WhatsApp bridge (bridge/whatsapp-bridge.mjs).
 *
 * This module owns everything group-shaped in the JSON-lines protocol.
 * It is intentionally READ-ONLY: list, info, participants, communities.
 * Nothing here adds, removes, promotes, renames, or changes group settings —
 * those stay owner-typed, owner-confirmed actions, never silent defaults.
 *
 * Protocol (python -> bridge), all additive to the existing bridge protocol:
 *   {"id":"<req>","cmd":"group_list"}
 *   {"id":"<req>","cmd":"group_info","chat":"<group jid>"}
 *   {"id":"<req>","cmd":"group_participants","chat":"<group jid>"}
 *   {"id":"<req>","cmd":"community_list"}
 *
 * Replies: {"id":"<req>","ok":true, ...} or {"id":"<req>","ok":false,"error":"..."}.
 *
 * Baileys surface used (verified against @whiskeysockets/baileys 6.7.24):
 *   lib/Socket/groups.js      — groupFetchAllParticipating, groupMetadata,
 *                                groupInviteCode
 *   lib/Socket/communities.js — communityMetadata (= groupMetadata shape)
 *   lib/Types/GroupMetadata.d.ts — GroupMetadata, GroupParticipant,
 *                                  ParticipantAction
 *   lib/Socket/index.js       — makeWASocket wraps makeCommunitiesSocket,
 *                                so the socket carries ALL of the above.
 *
 * Usage from whatsapp-bridge.mjs:
 *   import registerGroupCommands from './whatsapp-groups.mjs';
 *   // in handleCommand's default: case:
 *   return (await registerGroupCommands({ getSocket: () => wa, reply, log })(sock, req))
 *     || reply(sock, id, { ok: false, error: `unknown cmd: ${cmd}` });
 */

import { jidNormalizedUser } from '@whiskeysockets/baileys';

// ── sanitizers: the wire gets plain JSON, never Baileys internals ──────────

const isGroupJid = (jid) => typeof jid === 'string' && jid.endsWith('@g.us');

/** Normalize a chat argument to a group JID. Honest errors, no guessing. */
function groupJid(raw) {
  const jid = jidNormalizedUser(String(raw || '').trim());
  if (!jid) throw new Error('missing group: pass a group JID (…@g.us)');
  if (!isGroupJid(jid)) throw new Error(`not a group JID: ${raw} (groups end in @g.us)`);
  return jid;
}

function participantRow(p) {
  const admin = p.admin === 'superadmin' || !!p.isSuperAdmin ? 'superadmin'
    : p.admin === 'admin' || !!p.isAdmin ? 'admin' : '';
  return {
    jid: jidNormalizedUser(p.id || ''),
    // name/notify come from the local contact store when known; may be
    // empty for strangers — never invented.
    name: p.name || p.notify || '',
    role: admin, // '' | 'admin' | 'superadmin'
  };
}

/** GroupMetadata -> wire shape. participants excluded unless asked. */
function metaRow(meta, { withParticipants = false, inviteCode = '' } = {}) {
  const participants = Array.isArray(meta.participants) ? meta.participants : [];
  const admins = participants
    .filter((p) => p.isAdmin || p.isSuperAdmin || p.admin)
    .map((p) => jidNormalizedUser(p.id || ''));
  const row = {
    id: meta.id || '',
    subject: meta.subject || '',
    desc: (meta.desc || '').slice(0, 500),
    size: typeof meta.size === 'number' ? meta.size : participants.length,
    creation: meta.creation || 0,
    owner: meta.owner ? jidNormalizedUser(meta.owner) : '',
    isCommunity: !!meta.isCommunity,
    isCommunityAnnounce: !!meta.isCommunityAnnounce,
    linkedParent: meta.linkedParent || '',
    announce: !!meta.announce,       // only admins can send
    restrict: !!meta.restrict,        // only admins can change settings
    memberAddMode: !!meta.memberAddMode,
    joinApprovalMode: !!meta.joinApprovalMode,
    ephemeralHours: meta.ephemeralDuration ? Math.round(meta.ephemeralDuration / 3600) : 0,
    admins,
  };
  if (withParticipants) row.participants = participants.map(participantRow);
  if (inviteCode) row.invite = `https://chat.whatsapp.com/${inviteCode}`;
  return row;
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

// ── the command table ──────────────────────────────────────────────────────
// Each handler is read-only against the socket. Mutating group ops
// (add/remove/promote/subject/settings) are deliberately NOT exposed here.

export function registerGroupCommands(ctx) {
  const { getSocket, reply, log } = ctx;
  const sock = () => {
    const s = getSocket();
    if (!s) throw new Error('whatsapp not connected yet');
    return s;
  };

  const handlers = {
    // Every group the account participates in — slim rows, no member lists.
    group_list: async () => {
      const all = await sock().groupFetchAllParticipating();
      const groups = Object.values(all || {})
        .map((m) => metaRow(m))
        .sort((a, b) => a.subject.localeCompare(b.subject));
      return { ok: true, groups };
    },

    // Full metadata for one group, plus the invite link when the
    // account is allowed to see it (admins; non-admins get the rest).
    group_info: async (req) => {
      const jid = groupJid(req.chat);
      const wa = sock();
      const meta = await wa.groupMetadata(jid);
      let inviteCode = '';
      try {
        inviteCode = await wa.groupInviteCode(jid);
      } catch (e) {
        log('groupInviteCode not permitted for', jid, '-', friendlyError(e));
      }
      return { ok: true, group: metaRow(meta, { inviteCode }) };
    },

    // Member roster with roles; names only where the contact store knows them.
    group_participants: async (req) => {
      const jid = groupJid(req.chat);
      const meta = await sock().groupMetadata(jid);
      const participants = (meta.participants || []).map(participantRow);
      const adminCount = participants.filter((p) => p.role).length;
      return {
        ok: true,
        group: { id: meta.id || jid, subject: meta.subject || '' },
        count: participants.length,
        admins: adminCount,
        participants,
      };
    },

    // Communities the account participates in, each with its linked groups.
    // (Baileys 6.7.24: community membership shows up in the same
    // groupFetchAllParticipating result, flagged isCommunity; linked
    // groups carry linkedParent = the community JID.)
    community_list: async () => {
      const all = Object.values(await sock().groupFetchAllParticipating() || {});
      const communities = all
        .filter((m) => m.isCommunity)
        .map((c) => {
          const linked = all
            .filter((m) => m.linkedParent === c.id)
            .map((m) => ({ id: m.id, subject: m.subject || '', size: m.size ?? (m.participants || []).length }));
          return {
            id: c.id,
            subject: c.subject || '',
            desc: (c.desc || '').slice(0, 500),
            size: c.size ?? (c.participants || []).length,
            groups: linked.sort((a, b) => a.subject.localeCompare(b.subject)),
          };
        })
        .sort((a, b) => a.subject.localeCompare(b.subject));
      return { ok: true, communities };
    },
  };

  /** Returns true when this module handled the command (replied itself). */
  return async (sockArg, req) => {
    const fn = handlers[req && req.cmd];
    if (!fn) return false;
    try {
      reply(sockArg, req.id, await fn(req));
    } catch (e) {
      log(`group cmd ${req.cmd} failed:`, e?.message || e);
      reply(sockArg, req.id, { ok: false, error: friendlyError(e) });
    }
    return true;
  };
}

export default registerGroupCommands;
