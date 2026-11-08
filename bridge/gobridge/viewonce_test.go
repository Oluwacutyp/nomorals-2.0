package main

// View-once classification pins (2026-11-08 incident fix).
//
// The owner's journal held ZERO view-once rows of any kind for
// the feature's whole life while the toggle sat ON: no capture
// was ever even attempted, because recognition depended on
// exactly one signal — events.Message.IsViewOnce, which
// whatsmeow's UnwrapRaw sets ONLY for the wrapper protos
// (ViewOnceMessage / V2 / V2Extension).  A media message
// carrying the view-once FLAG directly (no wrapper) sailed
// through as an ordinary photo, and any event reaching the
// handler still wrapped had its media buried where
// mediaKindOf cannot see it.  unwrapInbound is the bridge's
// own classification: every real shape, pinned here.

import (
	"testing"

	waE2E "go.mau.fi/whatsmeow/proto/waE2E"
	"go.mau.fi/whatsmeow/types/events"
	"google.golang.org/protobuf/proto"
)

func voImage() *waE2E.Message {
	return &waE2E.Message{ImageMessage: &waE2E.ImageMessage{
		Mimetype: proto.String("image/jpeg"),
		URL:      proto.String("https://mmg.whatsapp.net/vo.jpg"),
	}}
}

func voVideo() *waE2E.Message {
	return &waE2E.Message{VideoMessage: &waE2E.VideoMessage{
		Mimetype: proto.String("video/mp4"),
		URL:      proto.String("https://mmg.whatsapp.net/vo.mp4"),
	}}
}

// dispatched mimics whatsmeow's own delivery: the handler's
// event is the UnwrapRaw of the raw stanza.
func dispatched(raw *waE2E.Message) *events.Message {
	evt := &events.Message{RawMessage: raw}
	return evt.UnwrapRaw()
}

func TestUnwrapInboundWrapperShapes(t *testing.T) {
	img, vid := voImage(), voVideo()
	shapes := map[string]*waE2E.Message{
		"v1 wrapper": {
			ViewOnceMessage: &waE2E.FutureProofMessage{
				Message: img}},
		"v2 wrapper": {
			ViewOnceMessageV2: &waE2E.FutureProofMessage{
				Message: img}},
		"v2 extension wrapper": {
			ViewOnceMessageV2Extension: &waE2E.FutureProofMessage{
				Message: vid}},
		"ephemeral around v2": {
			EphemeralMessage: &waE2E.FutureProofMessage{
				Message: &waE2E.Message{
					ViewOnceMessageV2: &waE2E.FutureProofMessage{
						Message: img}}}},
	}
	for name, raw := range shapes {
		// Both deliveries must classify: the whatsmeow-
		// dispatched event (already unwrapped, IsViewOnce
		// set) and the same stanza still wrapped (a dispatch
		// path that skipped UnwrapRaw).
		for _, evt := range []*events.Message{
			dispatched(raw), {Message: raw},
		} {
			msg, viewOnce := unwrapInbound(evt)
			if !viewOnce {
				t.Fatalf("%s: viewOnce = false, want true", name)
			}
			kind, dl, _, _ := mediaPartsOf(msg)
			if kind == "" || dl == nil {
				t.Fatalf("%s: unwrapped message carries no "+
					"downloadable media (kind %q)", name, kind)
			}
		}
	}
}

func TestUnwrapInboundFlagShapes(t *testing.T) {
	// The flag form: no wrapper at all — the media message
	// itself carries ViewOnce=true.  whatsmeow never folds
	// this into IsViewOnce; the old handler stamped
	// view_once:false and the arrival was an ordinary photo.
	flagImg := &waE2E.Message{ImageMessage: &waE2E.ImageMessage{
		Mimetype: proto.String("image/jpeg"),
		URL:      proto.String("https://mmg.whatsapp.net/f.jpg"),
		ViewOnce: proto.Bool(true)}}
	flagVid := &waE2E.Message{VideoMessage: &waE2E.VideoMessage{
		Mimetype: proto.String("video/mp4"),
		URL:      proto.String("https://mmg.whatsapp.net/f.mp4"),
		ViewOnce: proto.Bool(true)}}
	flagAudio := &waE2E.Message{AudioMessage: &waE2E.AudioMessage{
		Mimetype: proto.String("audio/ogg"),
		URL:      proto.String("https://mmg.whatsapp.net/f.ogg"),
		ViewOnce: proto.Bool(true)}}
	flagInEphemeral := &waE2E.Message{
		EphemeralMessage: &waE2E.FutureProofMessage{
			Message: flagImg}}
	for name, raw := range map[string]*waE2E.Message{
		"flag image": flagImg, "flag video": flagVid,
		"flag audio":              flagAudio,
		"flag image in ephemeral": flagInEphemeral,
	} {
		msg, viewOnce := unwrapInbound(
			&events.Message{Message: raw})
		if !viewOnce {
			t.Fatalf("%s: viewOnce = false, want true", name)
		}
		kind, dl, _, _ := mediaPartsOf(msg)
		if kind == "" || dl == nil {
			t.Fatalf("%s: no downloadable media (kind %q)",
				name, kind)
		}
	}
}

func TestUnwrapInboundOrdinaryMessagesUntouched(t *testing.T) {
	plain := voImage()
	msg, viewOnce := unwrapInbound(&events.Message{Message: plain})
	if viewOnce {
		t.Fatal("ordinary image classified view-once")
	}
	if msg != plain {
		t.Fatal("ordinary message was replaced")
	}
	text := &waE2E.Message{Conversation: proto.String("hi")}
	msg, viewOnce = unwrapInbound(&events.Message{Message: text})
	if viewOnce || msg != text {
		t.Fatal("ordinary text message altered")
	}
	msg, viewOnce = unwrapInbound(&events.Message{})
	if msg != nil || viewOnce {
		t.Fatal("empty event misclassified")
	}
}
