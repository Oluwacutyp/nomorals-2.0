package main

// Bridge-level pins for INBOUND location shares (Atlas
// M6/M9 remainder, 2026-11-10).  Outbound pins shipped in
// bundle #9 (pin_test.go), but onMessage never read a
// LocationMessage / LiveLocationMessage off an inbound
// event, so a WhatsApp location share crossed the wire as
// an empty text message and Devon's inbound-location seam
// (pending marks, then the live-location broker) never saw
// a WhatsApp fix.  These tests pin locationOf's extraction
// against the real waE2E proto shapes — all-optional
// pointer fields, so ABSENT must stay absent on the wire:
// a field the sender's client never set is never
// zero-filled into fake precision.

import (
	"math"
	"testing"

	waE2E "go.mau.fi/whatsmeow/proto/waE2E"
	"google.golang.org/protobuf/proto"
)

func TestLocationOfStaticPin(t *testing.T) {
	msg := &waE2E.Message{LocationMessage: &waE2E.LocationMessage{
		DegreesLatitude:  proto.Float64(7.5989),
		DegreesLongitude: proto.Float64(5.1057),
		AccuracyInMeters: proto.Uint32(12),
	}}
	out := locationOf(msg)
	if out == nil {
		t.Fatal("a static pin must produce a location object")
	}
	if out["lat"] != 7.5989 || out["lon"] != 5.1057 {
		t.Fatalf("coordinates = %v/%v, want the pin's degrees",
			out["lat"], out["lon"])
	}
	if out["live"] != false {
		t.Fatalf("live = %v, want false for a static pin", out["live"])
	}
	if out["accuracy"] != uint32(12) {
		t.Fatalf("accuracy = %v, want 12 (carried by the source)",
			out["accuracy"])
	}
	if _, ok := out["name"]; ok {
		t.Fatalf("name rode the wire though the pin never set one: %v", out)
	}
	if _, ok := out["address"]; ok {
		t.Fatalf("address rode the wire though the pin never set one: %v", out)
	}
}

func TestLocationOfVenuePin(t *testing.T) {
	// WhatsApp's venue form is a LocationMessage with the
	// label fields set — they ride verbatim.
	msg := &waE2E.Message{LocationMessage: &waE2E.LocationMessage{
		DegreesLatitude:  proto.Float64(6.5244),
		DegreesLongitude: proto.Float64(3.3792),
		Name:             proto.String("Mama's shop"),
		Address:          proto.String("12 Allen Avenue, Ikeja"),
	}}
	out := locationOf(msg)
	if out == nil {
		t.Fatal("a venue pin must produce a location object")
	}
	if out["name"] != "Mama's shop" ||
		out["address"] != "12 Allen Avenue, Ikeja" {
		t.Fatalf("venue labels = %v/%v", out["name"], out["address"])
	}
	if _, ok := out["accuracy"]; ok {
		t.Fatalf("accuracy rode the wire though the venue never set one: %v",
			out)
	}
}

func TestLocationOfLiveShare(t *testing.T) {
	msg := &waE2E.Message{LiveLocationMessage: &waE2E.LiveLocationMessage{
		DegreesLatitude:  proto.Float64(7.5989),
		DegreesLongitude: proto.Float64(5.1057),
		AccuracyInMeters: proto.Uint32(8),
		Caption:          proto.String("on my way"),
	}}
	out := locationOf(msg)
	if out == nil {
		t.Fatal("a live share update must produce a location object")
	}
	if out["live"] != true {
		t.Fatalf("live = %v, want true for a LiveLocationMessage",
			out["live"])
	}
	if out["accuracy"] != uint32(8) {
		t.Fatalf("accuracy = %v, want 8", out["accuracy"])
	}
	// The live share's caption is its text.
	if got := textOf(msg); got != "on my way" {
		t.Fatalf("textOf(live share) = %q, want its caption", got)
	}
}

func TestLocationOfLiveOpeningFrame(t *testing.T) {
	// A live share's opening frame can arrive as a
	// LocationMessage with IsLive set — the flag rides.
	msg := &waE2E.Message{LocationMessage: &waE2E.LocationMessage{
		DegreesLatitude:  proto.Float64(7.5989),
		DegreesLongitude: proto.Float64(5.1057),
		IsLive:           proto.Bool(true),
	}}
	out := locationOf(msg)
	if out == nil || out["live"] != true {
		t.Fatalf("opening frame = %v, want live=true", out)
	}
}

func TestLocationOfAbsentShapes(t *testing.T) {
	if out := locationOf(nil); out != nil {
		t.Fatalf("nil message produced %v", out)
	}
	text := &waE2E.Message{Conversation: proto.String("hey")}
	if out := locationOf(text); out != nil {
		t.Fatalf("a text message produced a location: %v", out)
	}
	half := &waE2E.Message{LocationMessage: &waE2E.LocationMessage{
		DegreesLatitude: proto.Float64(7.5989),
	}}
	if out := locationOf(half); out != nil {
		t.Fatalf("a pin missing its longitude produced %v", out)
	}
	wild := &waE2E.Message{LocationMessage: &waE2E.LocationMessage{
		DegreesLatitude:  proto.Float64(math.NaN()),
		DegreesLongitude: proto.Float64(5.1057),
	}}
	if out := locationOf(wild); out != nil {
		t.Fatalf("a NaN coordinate produced %v — it would poison "+
			"the whole event at marshal time", out)
	}
	far := &waE2E.Message{LiveLocationMessage: &waE2E.LiveLocationMessage{
		DegreesLatitude:  proto.Float64(91.0),
		DegreesLongitude: proto.Float64(5.1057),
	}}
	if out := locationOf(far); out != nil {
		t.Fatalf("an out-of-range latitude produced %v", out)
	}
}
