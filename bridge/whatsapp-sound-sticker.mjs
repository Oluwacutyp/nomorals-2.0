// whatsapp-sound-sticker.mjs — sound stickers on the Baileys rung.
//
// The pinned Baileys 6.7.24 WAProto has NO field 26 on
// StickerMessage (its StickerMessage ends at field 22,
// accessibilityLabel — see WAProto/WAProto.proto in the
// published 6.7.24 package), so a plain assignment of
// audioMessage would be dropped by the static protobuf
// encoder. This module patches the generated StickerMessage
// encode/decode at bridge startup so field 26 (the nested
// AudioMessage — the transport pinned from @aunucore/baileys
// 2.7.0, where the audio travels as a second media upload,
// not bytes inside the WebP) survives the wire both ways.
//
// The patch is wire-level and minimal: encode appends the field
// after the known fields; decode wire-scans the raw bytes for
// it. Nothing else about the generated types moves, and the
// patch installs exactly once (guarded by a marker on the
// type). Unpatching is not needed — the bridge process is
// single-purpose.

import fs from 'node:fs';

const FIELD_26_TAG = (26 << 3) | 2; // 210, length-delimited

function readVarint(buf, pos) {
  let value = 0;
  let shift = 0;
  let p = pos;
  while (p < buf.length) {
    const b = buf[p++];
    value |= (b & 0x7f) << shift;
    if (!(b & 0x80)) return { value, next: p };
    shift += 7;
    if (shift > 63) return null;
  }
  return null;
}

// Walk one message's wire bytes for the field-26
// length-delimited value; null when the field is absent or the
// bytes cannot be walked safely (groups / unknown wire types).
function scanForField26(buf, start, end) {
  let pos = start;
  while (pos < end) {
    const tag = readVarint(buf, pos);
    if (!tag) return null;
    pos = tag.next;
    const fieldNo = tag.value >>> 3;
    const wireType = tag.value & 7;
    if (fieldNo === 26 && wireType === 2) {
      const len = readVarint(buf, pos);
      if (!len) return null;
      const dataStart = len.next;
      const dataEnd = dataStart + len.value;
      if (dataEnd > end) return null;
      return buf.subarray(dataStart, dataEnd);
    }
    if (wireType === 0) {
      const v = readVarint(buf, pos);
      if (!v) return null;
      pos = v.next;
    } else if (wireType === 1) {
      pos += 8;
    } else if (wireType === 2) {
      const len = readVarint(buf, pos);
      if (!len) return null;
      pos = len.next + len.value;
    } else if (wireType === 5) {
      pos += 4;
    } else {
      return null;
    }
    if (pos > end) return null;
  }
  return null;
}

// Install the field-26 patch on Baileys' StickerMessage type.
// Returns true when the patch is live (or was already), false
// when the type is not what this was written against — the
// bridge then refuses sound sends instead of emitting a
// silent sticker in their place.
export function installStickerAudioPatch(proto) {
  const StickerMessage = proto?.Message?.StickerMessage;
  const AudioMessage = proto?.Message?.AudioMessage;
  if (!StickerMessage || !AudioMessage
    || typeof StickerMessage.encode !== 'function'
    || typeof StickerMessage.decode !== 'function') return false;
  if (StickerMessage.__soundStickerPatched) return true;
  const origEncode = StickerMessage.encode;
  const origDecode = StickerMessage.decode;
  StickerMessage.encode = function (message, writer) {
    const w = origEncode.call(this, message, writer);
    if (message && message.audioMessage) {
      const audioBytes = AudioMessage.encode(message.audioMessage)
        .finish();
      w.uint32(FIELD_26_TAG).bytes(audioBytes);
    }
    return w;
  };
  StickerMessage.decode = function (reader, length) {
    // Capture the raw slice BEFORE decoding: the reader's
    // position is the message start; length bounds it. A
    // protobufjs Reader carries .buf/.pos/.len; a bare
    // Uint8Array is the whole message.
    let buf = null;
    let start = 0;
    let end = 0;
    if (reader instanceof Uint8Array) {
      buf = reader;
      start = 0;
      end = length ? length : reader.length;
    } else if (reader && reader.buf instanceof Uint8Array
      && typeof reader.pos === 'number') {
      buf = reader.buf;
      start = reader.pos;
      end = length ? start + length
        : (typeof reader.len === 'number' ? reader.len : buf.length);
    }
    const msg = origDecode.call(this, reader, length);
    if (buf) {
      const audioBytes = scanForField26(buf, start, end);
      if (audioBytes) {
        try {
          msg.audioMessage = AudioMessage.decode(audioBytes);
        } catch { /* a corrupt field 26 reads as no audio */ }
      }
    }
    return msg;
  };
  StickerMessage.__soundStickerPatched = true;
  return true;
}

// Animated-WebP sniff (the Go bridge's webpIsAnimated, same
// rule): a VP8X chunk with the animation flag (bit 0x02) or an
// ANIM chunk means animated. Malformed input reads false.
export function isAnimatedWebPBuffer(buf) {
  if (!buf || buf.length < 12) return false;
  if (buf.toString('ascii', 0, 4) !== 'RIFF'
    || buf.toString('ascii', 8, 12) !== 'WEBP') return false;
  let off = 12;
  while (off + 8 <= buf.length) {
    const id = buf.toString('ascii', off, off + 4);
    const size = buf.readUInt32LE(off + 4);
    if (id === 'ANIM') return true;
    if (id === 'VP8X' && off + 9 <= buf.length
      && (buf[off + 8] & 0x02) !== 0) return true;
    off += 8 + size + (size % 2);
  }
  return false;
}

// Build the sound sticker's StickerMessage: the sticker and the
// audio are prepared (uploaded) in parallel, like the fork's
// stickerAudio branch, then the audio nests on field 26.
// `seconds` rides only when known (> 0); the isAnimated flag
// comes from the WebP bytes, and stickerSentTs is stamped like
// every sticker prepare. Throws when either side fails to
// build — the caller refuses honestly, never sends silent.
export async function buildSoundStickerMessage({
  generateWAMessageContent, upload,
  stickerPath, audioPath, seconds,
}) {
  const [stickerContent, audioContent] = await Promise.all([
    generateWAMessageContent({ sticker: { url: stickerPath } },
      { upload }),
    generateWAMessageContent(
      {
        audio: { url: audioPath },
        mimetype: 'audio/ogg; codecs=opus',
        ...(seconds > 0
          ? { seconds: Math.trunc(seconds) } : {}),
      },
      { upload }),
  ]);
  const stickerMessage = stickerContent?.stickerMessage;
  if (!stickerMessage) {
    throw new Error('Baileys did not build a sticker message');
  }
  if (!audioContent?.audioMessage) {
    throw new Error('Baileys did not build an audio message');
  }
  stickerMessage.audioMessage = audioContent.audioMessage;
  stickerMessage.isAnimated = isAnimatedWebPBuffer(
    fs.readFileSync(stickerPath));
  stickerMessage.stickerSentTs = Date.now();
  return stickerMessage;
}
