package main

// Bridge-level pins for the location + contact verbs
// (parity bundle #9).  Same standard as profile_test.go:
// the Python side's recording fakes accept any shape, so
// the message shapes are pinned HERE against a fake
// whatsmeow client that records exactly what it was asked
// to send — a LocationMessage with swapped degrees or a
// ContactMessage missing its vCard would be a silent
// live failure otherwise.

import (
	"context"
	"errors"
	"math"
	"strings"
	"testing"

	"go.mau.fi/whatsmeow"
	waE2E "go.mau.fi/whatsmeow/proto/waE2E"
	"go.mau.fi/whatsmeow/types"
)

type fakePinClient struct {
	sent []*waE2E.Message
	to   types.JID
	err  error
}

func (f *fakePinClient) SendMessage(_ context.Context,
	to types.JID, message *waE2E.Message,
	_ ...whatsmeow.SendRequestExtra,
) (whatsmeow.SendResponse, error) {
	if f.err != nil {
		return whatsmeow.SendResponse{}, f.err
	}
	f.to = to
	f.sent = append(f.sent, message)
	return whatsmeow.SendResponse{ID: "PIN-1"}, nil
}

func TestLocationShapeError(t *testing.T) {
	if got := locationShapeError(6.5244, 3.3792, true, true); got != "" {
		t.Fatalf("valid coordinates refused: %s", got)
	}
	if got := locationShapeError(0, 0, false, true); !strings.Contains(got, "numeric") {
		t.Fatalf("absent latitude must refuse as missing, got: %s", got)
	}
	if got := locationShapeError(91, 0, true, true); !strings.Contains(got, "latitude between -90 and 90") {
		t.Fatalf("latitude range refusal missing, got: %s", got)
	}
	if got := locationShapeError(0, -181, true, true); !strings.Contains(got, "longitude between -180 and 180") {
		t.Fatalf("longitude range refusal missing, got: %s", got)
	}
	if got := locationShapeError(math.NaN(), 0, true, true); !strings.Contains(got, "finite") {
		t.Fatalf("NaN refusal missing, got: %s", got)
	}
}

func TestBuildLocationMessage(t *testing.T) {
	msg := buildLocationMessage(6.5244, 3.3792, "", "")
	loc := msg.GetLocationMessage()
	if loc == nil {
		t.Fatal("no LocationMessage built")
	}
	if loc.GetDegreesLatitude() != 6.5244 ||
		loc.GetDegreesLongitude() != 3.3792 {
		t.Fatalf("degrees swapped or wrong: %v %v",
			loc.GetDegreesLatitude(),
			loc.GetDegreesLongitude())
	}
	if loc.Name != nil || loc.Address != nil {
		t.Fatal("a bare pin must carry no name/address")
	}
	named := buildLocationMessage(6.5, 3.3, "Lekki Gate",
		"Admiralty Way").GetLocationMessage()
	if named.GetName() != "Lekki Gate" ||
		named.GetAddress() != "Admiralty Way" {
		t.Fatalf("venue fields lost: %q %q", named.GetName(),
			named.GetAddress())
	}
}

func TestContactCardErrorAndMessage(t *testing.T) {
	if got := contactCardError("", "BEGIN:VCARD"); !strings.Contains(got, "display name") {
		t.Fatalf("nameless card refusal missing, got: %s", got)
	}
	if got := contactCardError("Ada", "  "); !strings.Contains(got, "vcard") {
		t.Fatalf("cardless refusal missing, got: %s", got)
	}
	if got := contactCardError("Ada", "BEGIN:VCARD\r\nEND:VCARD"); got != "" {
		t.Fatalf("valid card refused: %s", got)
	}
	card := buildContactMessage("Ada Lovelace",
		"BEGIN:VCARD\r\nEND:VCARD").GetContactMessage()
	if card == nil || card.GetDisplayName() != "Ada Lovelace" ||
		!strings.HasPrefix(card.GetVcard(), "BEGIN:VCARD") {
		t.Fatalf("contact message shape wrong: %+v", card)
	}
}

func TestSendPinRecordsAndRefuses(t *testing.T) {
	chat := types.JID{User: "2348012345678",
		Server: types.DefaultUserServer}
	fake := &fakePinClient{}
	id, err := sendPin(context.Background(), fake, chat,
		buildLocationMessage(6.5244, 3.3792, "", ""))
	if err != nil || id != "PIN-1" {
		t.Fatalf("sendPin = %q, %v", id, err)
	}
	if fake.to != chat || len(fake.sent) != 1 {
		t.Fatalf("the pin went to the wrong place: %v", fake.to)
	}
	boom := errors.New("socket closed")
	fake = &fakePinClient{err: boom}
	if _, err := sendPin(context.Background(), fake, chat,
		buildLocationMessage(1, 2, "", "")); !errors.Is(err, boom) {
		t.Fatalf("a failed send must surface its error, got %v", err)
	}
}
