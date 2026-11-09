package main

// Inbound video-note classification pins (census re-grade gap,
// 2026-11-09).  Bundle #10 built video-note SENDING: the send
// side has called the round form "video_note" since then.  The
// RECEIVE side never saw it: a video note arrives as PtvMessage
// — a VideoMessage in its OWN field, not in VideoMessage — and
// mediaKindOf had no PtvMessage case, so a received note
// classified as kind "" and its media never downloaded (the
// Python side then could not tell a note arrived at all).
// These pins exercise the real waE2E shapes the bridge decodes:
// the bare PtvMessage, the same note inside an ephemeral
// wrapper, and an ordinary video as the discriminator (the PTV
// field is what makes a note a note — a plain VideoMessage
// still reads "video").

import (
	"testing"

	waE2E "go.mau.fi/whatsmeow/proto/waE2E"
	"go.mau.fi/whatsmeow/types/events"
	"google.golang.org/protobuf/proto"
)

func ptvNote() *waE2E.Message {
	return &waE2E.Message{PtvMessage: &waE2E.VideoMessage{
		Mimetype: proto.String("video/mp4"),
		URL:      proto.String("https://mmg.whatsapp.net/note.mp4"),
		Seconds:  proto.Uint32(7),
	}}
}

func TestInboundVideoNoteClassifies(t *testing.T) {
	msg, viewOnce := unwrapInbound(
		&events.Message{Message: ptvNote()})
	if viewOnce {
		t.Fatal("a video note is not view-once")
	}
	if kind := mediaKindOf(msg); kind != "video_note" {
		t.Fatalf("mediaKindOf(ptv) = %q, want video_note", kind)
	}
	kind, dl, mime, _ := mediaPartsOf(msg)
	if kind != "video_note" || dl == nil {
		t.Fatalf("mediaPartsOf(ptv) = kind %q, dl %v — "+
			"the note's bytes are unreachable", kind, dl)
	}
	if mime != "video/mp4" {
		t.Fatalf("mediaPartsOf(ptv) mime = %q, want video/mp4", mime)
	}
	if ext := mediaExtFor(kind, ""); ext != ".mp4" {
		t.Fatalf("mediaExtFor(video_note) = %q, want .mp4 "+
			"(an unmapped kind lands in the cache as .bin)", ext)
	}
}

func TestInboundVideoNoteWrappedShapes(t *testing.T) {
	shapes := map[string]*waE2E.Message{
		"ephemeral around ptv": {
			EphemeralMessage: &waE2E.FutureProofMessage{
				Message: ptvNote()}},
		"device-sent ptv": {
			DeviceSentMessage: &waE2E.DeviceSentMessage{
				Message: ptvNote()}},
	}
	for name, raw := range shapes {
		msg, _ := unwrapInbound(&events.Message{Message: raw})
		kind, dl, _, _ := mediaPartsOf(msg)
		if kind != "video_note" || dl == nil {
			t.Fatalf("%s: kind %q, dl %v — want a downloadable "+
				"video_note", name, kind, dl)
		}
	}
}

func TestInboundOrdinaryVideoStaysVideo(t *testing.T) {
	// The discriminator pin: PTV is a field, not a vibe.  A
	// plain VideoMessage — even a short, round-cropped one —
	// is "video", never "video_note".
	plain := &waE2E.Message{VideoMessage: &waE2E.VideoMessage{
		Mimetype: proto.String("video/mp4"),
		URL:      proto.String("https://mmg.whatsapp.net/plain.mp4"),
		Seconds:  proto.Uint32(7),
	}}
	msg, _ := unwrapInbound(&events.Message{Message: plain})
	if kind := mediaKindOf(msg); kind != "video" {
		t.Fatalf("mediaKindOf(plain video) = %q, want video", kind)
	}
}
