package main

// Bridge-level pins for the newsletter metadata updates
// (parity G4, census bundle #7).  Same standard as the
// status-fix and G3 tests: the shapes that fail SILENTLY
// live — a Mex call with the wrong query ID, an updates
// envelope missing its settings key — are pinned HERE,
// against a fake that records exactly what the internal
// whatsmeow client was asked to send.

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	"go.mau.fi/whatsmeow/types"
)

type fakeMexClient struct {
	queryID   string
	variables any
	calls     int
	err       error
}

func (f *fakeMexClient) SendMexIQ(_ context.Context,
	queryID string, variables any) (json.RawMessage, error) {
	f.queryID = queryID
	f.variables = variables
	f.calls++
	return json.RawMessage(`{}`), f.err
}

var testChannelJID = types.JID{
	User:   "120207123456789012",
	Server: types.NewsletterServer,
}

func TestSetNewsletterNameSendsTheUpdateMutation(t *testing.T) {
	mex := &fakeMexClient{}
	err := setNewsletterName(context.Background(), mex,
		testChannelJID, "New Channel Name")
	if err != nil {
		t.Fatalf("setNewsletterName: %v", err)
	}
	if mex.calls != 1 {
		t.Fatalf("SendMexIQ calls = %d, want exactly 1", mex.calls)
	}
	// whatsmeow's own mutation ID (vendored newsletter.go).
	if mex.queryID != "7150902998257522" {
		t.Fatalf("query ID = %q, want the newsletter update "+
			"mutation 7150902998257522", mex.queryID)
	}
	want := map[string]any{
		"newsletter_id": "120207123456789012@newsletter",
		"updates": map[string]any{
			"name":     "New Channel Name",
			"settings": nil,
		},
	}
	if !reflect.DeepEqual(mex.variables, want) {
		t.Fatalf("variables = %#v, want %#v", mex.variables, want)
	}
}

func TestSetNewsletterDescriptionSendsOnlyTheDescription(t *testing.T) {
	mex := &fakeMexClient{}
	err := setNewsletterDescription(context.Background(), mex,
		testChannelJID, "Fresh episodes every Friday")
	if err != nil {
		t.Fatalf("setNewsletterDescription: %v", err)
	}
	want := map[string]any{
		"newsletter_id": "120207123456789012@newsletter",
		"updates": map[string]any{
			"description": "Fresh episodes every Friday",
			"settings":    nil,
		},
	}
	if !reflect.DeepEqual(mex.variables, want) {
		t.Fatalf("variables = %#v, want %#v", mex.variables, want)
	}
}

func TestSetNewsletterDescriptionEmptyClears(t *testing.T) {
	// An empty description is a real update (it clears the
	// field), not a no-op — the Node rung passes "" through
	// the same way, so the Go rung must send it, not skip it.
	mex := &fakeMexClient{}
	err := setNewsletterDescription(context.Background(), mex,
		testChannelJID, "")
	if err != nil {
		t.Fatalf("setNewsletterDescription: %v", err)
	}
	if mex.calls != 1 {
		t.Fatalf("SendMexIQ calls = %d, want exactly 1", mex.calls)
	}
	updates := mex.variables.(map[string]any)["updates"].(map[string]any)
	if updates["description"] != "" {
		t.Fatalf("description = %v, want \"\"", updates["description"])
	}
}

func TestUpdateNewsletterPropagatesTheMexError(t *testing.T) {
	// A refused update surfaces the transport's error — the
	// handler turns it into the bridge's failure reply, never
	// an ok.
	mex := &fakeMexClient{err: errors.New("graphql error: not authorized")}
	err := setNewsletterName(context.Background(), mex,
		testChannelJID, "Nope")
	if err == nil || err.Error() != "graphql error: not authorized" {
		t.Fatalf("err = %v, want the Mex error verbatim", err)
	}
}
