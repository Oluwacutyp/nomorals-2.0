package main

// newsletter.go — channel (newsletter) metadata updates
// (parity G4, census bundle #7).
//
// The Node/Baileys bridge renamed channels from the start
// (newsletterUpdateName / newsletterUpdateDescription in
// whatsapp-groups-admin.mjs); this bridge answered an honest
// "no verb in this whatsmeow pin" refusal because whatsmeow
// exposes no public UpdateNewsletter method.  The verb exists
// one level down: the vendored whatsmeow's newsletter.go
// carries the update mutation's query ID
// (mutationUpdateNewsletter) and every newsletter verb rides
// sendMexIQ — which the library also exposes, deliberately,
// as DangerousInternals().SendMexIQ (internals.go).  The
// update's wire shape is the one the working Baileys
// implementation sends (lib/Socket/newsletter.js,
// newsletterUpdate): variables {newsletter_id, updates:
// {<one field>, settings: null}} — one field per call, and
// settings always present and null (the mute mutation's
// documented variables in whatsmeow's newsletter.go carry
// the same {newsletter_id, updates} envelope).  sendMexIQ
// swaps in the desktop query-ID variant when the client
// payload calls for it, exactly as it does for the library's
// own newsletter verbs, so this bridge passes the base ID.

import (
	"context"
	"encoding/json"

	"go.mau.fi/whatsmeow"
	"go.mau.fi/whatsmeow/types"
)

// mexClient is the slice of whatsmeow's internal client the
// newsletter update rides — an interface so the verb's call
// SHAPE is pinnable in tests (the status-fix lesson: a
// recording fake on the Python side accepts any shape; the
// bridge layer is where shapes must be pinned).
type mexClient interface {
	SendMexIQ(ctx context.Context, queryID string,
		variables any) (json.RawMessage, error)
}

var _ mexClient = (*whatsmeow.DangerousInternalClient)(nil)

// mutationUpdateNewsletter is the query ID of WhatsApp's
// newsletter-metadata update mutation — the vendored
// whatsmeow's own constant for it (newsletter.go:
// mutationUpdateNewsletter = "7150902998257522"), not a
// number copied from another library.
const mutationUpdateNewsletter = "7150902998257522"

// updateNewsletter sends ONE newsletter metadata update.
// The caller's updates map gains settings: null
// unconditionally: Baileys' shipped shape carries it on
// every update, and an absent settings key is not the same
// update on the wire.
func updateNewsletter(ctx context.Context, mex mexClient,
	jid types.JID, updates map[string]any) error {
	full := map[string]any{"settings": nil}
	for k, v := range updates {
		full[k] = v
	}
	_, err := mex.SendMexIQ(ctx, mutationUpdateNewsletter,
		map[string]any{
			"newsletter_id": jid.String(),
			"updates":       full,
		})
	return err
}

// setNewsletterName renames one channel the account owns.
func setNewsletterName(ctx context.Context, mex mexClient,
	jid types.JID, name string) error {
	return updateNewsletter(ctx, mex, jid,
		map[string]any{"name": name})
}

// setNewsletterDescription sets one channel's description
// (an empty string clears it — the Node rung passes the
// description through the same way).
func setNewsletterDescription(ctx context.Context,
	mex mexClient, jid types.JID, description string) error {
	return updateNewsletter(ctx, mex, jid,
		map[string]any{"description": description})
}
