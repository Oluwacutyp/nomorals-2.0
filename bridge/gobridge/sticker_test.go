package main

// Sticker sends on WhatsApp (the 2026-11-26 /sticker incident):
// a .webm sticker must be REFUSED honestly instead of landing
// as a document wearing a content-hash name, a .webp sticker
// must ride as the protocol's StickerMessage, and the animated
// flag must come from the bytes — proven here at the wire-shape
// level, so what these tests bless is the bytes' shape, not a
// struct that merely looks right in memory.

import (
	"testing"

	waE2E "go.mau.fi/whatsmeow/proto/waE2E"
	"google.golang.org/protobuf/proto"
)

func TestStickerRefusalMatrix(t *testing.T) {
	cases := []struct {
		kind, ext string
		viewOnce  bool
		want      bool
	}{
		// A .webm can never be a WhatsApp sticker: refused.
		{"sticker", ".webm", false, true},
		// A .webp sticker is the real form: allowed.
		{"sticker", ".webp", false, false},
		// A sticker cannot carry view-once (the flag would
		// silently drop — applyViewOnce only touches
		// Image/Video).
		{"sticker", ".webp", true, true},
		// Anything that is not a sticker is untouched by this
		// gate: a .webm video/animation without the sticker
		// kind keeps the existing document behaviour.
		{"video", ".webm", false, false},
		{"image", ".webp", false, false},
		{"", ".webm", false, false},
	}
	for _, c := range cases {
		got := stickerRefusal(c.kind, c.ext, c.viewOnce)
		if c.want && got == "" {
			t.Fatalf("kind=%q ext=%q viewOnce=%v: want a refusal, got none",
				c.kind, c.ext, c.viewOnce)
		}
		if !c.want && got != "" {
			t.Fatalf("kind=%q ext=%q viewOnce=%v: want no refusal, got %q",
				c.kind, c.ext, c.viewOnce, got)
		}
	}
}

// webpBytes builds a minimal RIFF/WebP with the given chunks:
// each chunk is (id, payload); sizes are written little-endian
// and odd payloads get their pad byte, the way a real encoder
// writes them.
func webpBytes(chunks ...[2][]byte) []byte {
	body := []byte("WEBP")
	for _, c := range chunks {
		body = append(body, c[0]...)
		n := len(c[1])
		body = append(body, byte(n), byte(n>>8), byte(n>>16), byte(n>>24))
		body = append(body, c[1]...)
		if n%2 != 0 {
			body = append(body, 0)
		}
	}
	out := []byte("RIFF")
	total := len(body)
	out = append(out, byte(total), byte(total>>8), byte(total>>16), byte(total>>24))
	return append(out, body...)
}

func TestWebpIsAnimated(t *testing.T) {
	// A VP8X chunk with the animation flag set (bit 0x02).
	animatedVP8X := webpBytes(
		[2][]byte{[]byte("VP8X"), []byte{0x02, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
	)
	// A VP8X chunk WITHOUT the animation flag: static.
	staticVP8X := webpBytes(
		[2][]byte{[]byte("VP8X"), []byte{0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
	)
	// An ANIM chunk on its own: animated.
	animatedANIM := webpBytes(
		[2][]byte{[]byte("ANIM"), []byte{0, 0, 0, 0, 0, 0}},
	)
	// A plain lossy WebP (VP8 chunk only): static.
	staticVP8 := webpBytes(
		[2][]byte{[]byte("VP8 "), []byte{1, 2, 3, 4}},
	)
	cases := []struct {
		name string
		data []byte
		want bool
	}{
		{"vp8x animated flag", animatedVP8X, true},
		{"vp8x static", staticVP8X, false},
		{"anim chunk", animatedANIM, true},
		{"plain vp8", staticVP8, false},
		{"empty", nil, false},
		{"not a webp", []byte("GIF89a"), false},
		{"truncated riff", []byte("RIFF"), false},
	}
	for _, c := range cases {
		if got := webpIsAnimated(c.data); got != c.want {
			t.Fatalf("%s: webpIsAnimated = %v, want %v",
				c.name, got, c.want)
		}
	}
}

func TestStickerMessageWireRoundTrip(t *testing.T) {
	// The shape cmdSendMedia builds for a .webp with kind
	// sticker: mimetype image/webp, IsAnimated from the bytes.
	msg := &waE2E.Message{StickerMessage: &waE2E.StickerMessage{
		URL:        proto.String("https://mmg.whatsapp.net/stk.webp"),
		DirectPath: proto.String("/direct/stk.webp"),
		MediaKey:   []byte("k"),
		Mimetype:   proto.String("image/webp"),
		FileSHA256: []byte("s"), FileEncSHA256: []byte("e"),
		FileLength: proto.Uint64(1234),
		IsAnimated: proto.Bool(true),
	}}
	raw, err := proto.Marshal(msg)
	if err != nil {
		t.Fatal(err)
	}
	back := &waE2E.Message{}
	if err := proto.Unmarshal(raw, back); err != nil {
		t.Fatal(err)
	}
	stk := back.GetStickerMessage()
	if stk == nil {
		t.Fatal("StickerMessage did not survive the wire round-trip")
	}
	if stk.GetMimetype() != "image/webp" {
		t.Fatalf("mimetype = %q, want image/webp", stk.GetMimetype())
	}
	if !stk.GetIsAnimated() {
		t.Fatal("IsAnimated did not survive the wire round-trip")
	}
	if stk.GetURL() != "https://mmg.whatsapp.net/stk.webp" {
		t.Fatal("URL did not survive the wire round-trip")
	}
}
