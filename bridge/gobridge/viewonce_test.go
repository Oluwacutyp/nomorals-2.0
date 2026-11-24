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
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"net"
	"os"
	"testing"
	"time"

	"go.mau.fi/whatsmeow"
	waE2E "go.mau.fi/whatsmeow/proto/waE2E"
	"go.mau.fi/whatsmeow/types"
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

// ── arrival download + redownload (2026-11-23 vault fix) ──────

// stubDownloader is the scripted media fetch: bytes or a named
// failure, counting calls so a test can prove exactly one fetch
// happened (or none, when the cache served).
type stubDownloader struct {
	data  []byte
	err   error
	calls int
}

func (s *stubDownloader) Download(
	ctx context.Context,
	msg whatsmeow.DownloadableMessage) ([]byte, error) {
	s.calls++
	return s.data, s.err
}

func TestDownloadMediaOfFailureRowCarriesError(t *testing.T) {
	dir := t.TempDir()
	dl := &stubDownloader{err: errors.New("410 Gone: media expired")}
	b := &bridge{mediaDir: dir, dl: dl}
	row := b.downloadMediaOf(context.Background(), voImage(), "VOFAIL1")
	if row == nil {
		t.Fatal("failed download returned nil, want the trace row")
	}
	if row["path"] != "" {
		t.Fatalf("path = %q, want empty on failure", row["path"])
	}
	if row["download_error"] != "410 Gone: media expired" {
		t.Fatalf("download_error = %q, want the failure string",
			row["download_error"])
	}
	if row["kind"] != "image" || row["mime"] != "image/jpeg" {
		t.Fatalf("row lost kind/mime: %v", row)
	}
	if dl.calls != 1 {
		t.Fatalf("calls = %d, want exactly 1 fetch", dl.calls)
	}
}

func TestDownloadMediaOfSuccessWritesFile(t *testing.T) {
	dir := t.TempDir()
	dl := &stubDownloader{data: []byte("vo-bytes")}
	b := &bridge{mediaDir: dir, dl: dl}
	row := b.downloadMediaOf(context.Background(), voImage(), "VOOK1")
	if row == nil || row["path"] == "" {
		t.Fatalf("row = %v, want a written file", row)
	}
	got, err := os.ReadFile(row["path"].(string))
	if err != nil || string(got) != "vo-bytes" {
		t.Fatalf("file = %q, err %v, want the stub bytes", got, err)
	}
	if _, hasErr := row["download_error"]; hasErr {
		t.Fatalf("success row carries download_error: %v", row)
	}
}

// cmdRedownloadOverPipe runs the real wire handler once and
// returns its decoded reply line (the poll-results test's
// pattern, reused — one JSON line per command).
func cmdRedownloadOverPipe(t *testing.T, b *bridge,
	req request) map[string]any {
	t.Helper()
	server, client := net.Pipe()
	defer server.Close()
	defer client.Close()
	cc := &clientConn{conn: server}
	done := make(chan struct{})
	go func() {
		defer close(done)
		b.cmdRedownload(context.Background(), cc, "req-1", req)
	}()
	line, err := bufio.NewReader(client).ReadBytes('\n')
	if err != nil {
		t.Fatalf("reading the reply line: %v", err)
	}
	<-done
	var out map[string]any
	if err := json.Unmarshal(line, &out); err != nil {
		t.Fatalf("the reply is not one JSON line: %v", err)
	}
	return out
}

func TestCmdRedownloadHitAndMiss(t *testing.T) {
	dir := t.TempDir()
	dl := &stubDownloader{data: []byte("redownloaded")}
	b := &bridge{
		mediaDir: dir, dl: dl,
		msgCache: map[string]*cachedMsg{},
	}
	sender := types.NewJID("2347111111111", types.DefaultUserServer)
	b.cacheMessage("VORED1", voImage(), sender)

	out := cmdRedownloadOverPipe(t, b,
		request{"message_id": "VORED1"})
	if out["ok"] != true {
		t.Fatalf("hit reply = %v, want ok", out)
	}
	media, _ := out["media"].(map[string]any)
	if media == nil || media["path"] == "" {
		t.Fatalf("hit reply carries no media row: %v", out)
	}
	got, err := os.ReadFile(media["path"].(string))
	if err != nil || string(got) != "redownloaded" {
		t.Fatalf("redownloaded file = %q, err %v", got, err)
	}

	// The second call serves the cached file, never re-fetches.
	calls := dl.calls
	out = cmdRedownloadOverPipe(t, b,
		request{"message_id": "VORED1"})
	if out["ok"] != true {
		t.Fatalf("second hit reply = %v, want ok", out)
	}
	if dl.calls != calls {
		t.Fatalf("calls rose %d -> %d, want the cache served",
			calls, dl.calls)
	}

	// A cached proto with no media is a named miss, not a fetch.
	b.cacheMessage("VOTEXT1",
		&waE2E.Message{Conversation: proto.String("hi")}, sender)
	out = cmdRedownloadOverPipe(t, b,
		request{"message_id": "VOTEXT1"})
	if out["ok"] != false {
		t.Fatalf("no-media reply = %v, want ok:false", out)
	}

	// An id the bridge never saw is a named miss.
	out = cmdRedownloadOverPipe(t, b,
		request{"message_id": "NEVER-SEEN"})
	if out["ok"] != false {
		t.Fatalf("unknown-id reply = %v, want ok:false", out)
	}

	// No id at all is a usage error, not a fetch.
	out = cmdRedownloadOverPipe(t, b, request{})
	if out["ok"] != false {
		t.Fatalf("empty-id reply = %v, want ok:false", out)
	}
}

func TestCmdRedownloadFailedDownloadNamesReason(t *testing.T) {
	dir := t.TempDir()
	dl := &stubDownloader{err: errors.New("410 Gone: media expired")}
	b := &bridge{
		mediaDir: dir, dl: dl,
		msgCache: map[string]*cachedMsg{},
	}
	sender := types.NewJID("2347111111111", types.DefaultUserServer)
	b.cacheMessage("VORED2", voImage(), sender)
	out := cmdRedownloadOverPipe(t, b,
		request{"message_id": "VORED2"})
	if out["ok"] != true {
		t.Fatalf("reply = %v, want ok with the trace row", out)
	}
	media, _ := out["media"].(map[string]any)
	if media == nil || media["path"] != "" {
		t.Fatalf("media = %v, want the pathless trace row", out["media"])
	}
	if media["download_error"] != "410 Gone: media expired" {
		t.Fatalf("download_error = %q, want the failure string",
			media["download_error"])
	}
}

// TestOnMessageOwnMessageDownloadsViewOnce drives the real
// onMessage for an own view-once photo (sent from another
// device): the broadcast own_message event must carry the
// downloaded media row and the view_once stamp — before this
// fix the own path stamped the flag and never fetched.
func TestOnMessageOwnMessageDownloadsViewOnce(t *testing.T) {
	dir := t.TempDir()
	dl := &stubDownloader{data: []byte("own-vo-bytes")}
	server, client := net.Pipe()
	defer server.Close()
	defer client.Close()
	cc := &clientConn{conn: server}
	b := &bridge{
		mediaDir: dir, dl: dl,
		clients:  map[*clientConn]struct{}{cc: {}},
		recent:   map[string][]*remembered{},
		msgCache: map[string]*cachedMsg{},
		titles:   map[string]string{},
	}
	chat := types.NewJID("2347111111111", types.DefaultUserServer)
	me := types.NewJID("2348012345678", types.DefaultUserServer)
	raw := &waE2E.Message{ImageMessage: &waE2E.ImageMessage{
		Mimetype: proto.String("image/jpeg"),
		URL:      proto.String("https://mmg.whatsapp.net/own-vo.jpg"),
		ViewOnce: proto.Bool(true)}}
	evt := &events.Message{
		Info: types.MessageInfo{
			Chat: chat, Sender: me, ID: "OWNVO1",
			PushName:  "Owner",
			IsFromMe:  true,
			Timestamp: time.Now(),
		},
		Message: raw,
	}
	lines := make(chan []byte, 1)
	go func() {
		line, err := bufio.NewReader(client).ReadBytes('\n')
		if err == nil {
			lines <- line
		}
	}()
	b.onMessage(evt)
	select {
	case line := <-lines:
		var out map[string]any
		if err := json.Unmarshal(line, &out); err != nil {
			t.Fatalf("broadcast is not one JSON line: %v", err)
		}
		if out["type"] != "own_message" {
			t.Fatalf("type = %v, want own_message", out["type"])
		}
		if out["view_once"] != true {
			t.Fatalf("view_once = %v, want true", out["view_once"])
		}
		media, _ := out["media"].([]any)
		if len(media) != 1 {
			t.Fatalf("media = %v, want exactly one row", out["media"])
		}
		row, _ := media[0].(map[string]any)
		if row["path"] == "" {
			t.Fatalf("media row = %v, want the downloaded file", row)
		}
		got, err := os.ReadFile(row["path"].(string))
		if err != nil || string(got) != "own-vo-bytes" {
			t.Fatalf("captured file = %q, err %v", got, err)
		}
		// The proto is cached, so a later redownload finds it.
		if _, ok := b.msgCache["OWNVO1"]; !ok {
			t.Fatal("own message was not cached for redownload")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("no broadcast line arrived for the own message")
	}
}
