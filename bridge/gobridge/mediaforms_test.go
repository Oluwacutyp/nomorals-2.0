package main

// Parity bundle #10 — the send-side special forms, pinned at
// the wire-shape level: the view-once FLAG on the media
// message itself, and a video note as the protocol's
// PtvMessage (field 66) — proven by a marshal round-trip, so
// what these tests bless is the bytes' shape, not a struct
// that merely looks right in memory.

import (
	"testing"

	"go.mau.fi/whatsmeow"
	"go.mau.fi/whatsmeow/proto/waE2E"
	"google.golang.org/protobuf/proto"
)

func TestSpecialFormRefusalMatrix(t *testing.T) {
	cases := []struct {
		viewOnce, ptv bool
		mediaType     whatsmeow.MediaType
		wantRefusal   bool
	}{
		{true, false, whatsmeow.MediaImage, false},
		{true, false, whatsmeow.MediaVideo, false},
		{true, false, whatsmeow.MediaAudio, true},
		{true, false, whatsmeow.MediaDocument, true},
		{false, true, whatsmeow.MediaVideo, false},
		{false, true, whatsmeow.MediaImage, true},
		{false, true, whatsmeow.MediaAudio, true},
		{false, false, whatsmeow.MediaDocument, false},
	}
	for _, c := range cases {
		got := specialFormRefusal(c.viewOnce, c.ptv, c.mediaType)
		if c.wantRefusal && got == "" {
			t.Fatalf("viewOnce=%v ptv=%v type=%v: want a refusal, got none",
				c.viewOnce, c.ptv, c.mediaType)
		}
		if !c.wantRefusal && got != "" {
			t.Fatalf("viewOnce=%v ptv=%v type=%v: want no refusal, got %q",
				c.viewOnce, c.ptv, c.mediaType, got)
		}
	}
}

func TestApplyViewOnceSetsTheFlagOnTheWire(t *testing.T) {
	img := &waE2E.Message{ImageMessage: &waE2E.ImageMessage{
		Caption: proto.String("one look")}}
	applyViewOnce(img)
	raw, err := proto.Marshal(img)
	if err != nil {
		t.Fatal(err)
	}
	back := &waE2E.Message{}
	if err := proto.Unmarshal(raw, back); err != nil {
		t.Fatal(err)
	}
	if !back.GetImageMessage().GetViewOnce() {
		t.Fatal("image viewOnce flag did not survive the wire round-trip")
	}
	vid := &waE2E.Message{VideoMessage: &waE2E.VideoMessage{}}
	applyViewOnce(vid)
	if !vid.GetVideoMessage().GetViewOnce() {
		t.Fatal("video viewOnce flag not set")
	}
	// A plain message keeps no flag — the form is opt-in only.
	plain := &waE2E.Message{ImageMessage: &waE2E.ImageMessage{}}
	if plain.GetImageMessage().GetViewOnce() {
		t.Fatal("unrequested image carries a viewOnce flag")
	}
}

func TestAsVideoNoteIsARealPtvMessage(t *testing.T) {
	built := &waE2E.Message{VideoMessage: &waE2E.VideoMessage{
		Caption: proto.String("a caption no client would render"),
		Seconds: proto.Uint32(7)}}
	note := asVideoNote(built)
	raw, err := proto.Marshal(note)
	if err != nil {
		t.Fatal(err)
	}
	back := &waE2E.Message{}
	if err := proto.Unmarshal(raw, back); err != nil {
		t.Fatal(err)
	}
	if back.GetPtvMessage() == nil {
		t.Fatal("no PtvMessage after the wire round-trip — not a video note")
	}
	if back.GetVideoMessage() != nil {
		t.Fatal("a video note must not also ride as a plain VideoMessage")
	}
	if back.GetPtvMessage().GetCaption() != "" {
		t.Fatal("the caption must be lifted out of the note proto")
	}
	if back.GetPtvMessage().GetSeconds() != 7 {
		t.Fatal("the video payload itself must survive the re-form")
	}
}
