package main

// Bridge-level pins for the media-cache discipline
// (2026-11-08).  The live incident: the channel sweep
// re-downloaded the newest posts' media on every housekeeping
// cycle because downloadMediaOf named each file
// "<download-time>-<id fragment>" and never checked the cache —
// one 76MB channel video accumulated ~145 copies in a morning
// and the directory reached 11.07 GB, filling the owner's disk.
// These tests pin the repair at the bridge seam: one media
// identity = one file, served — never re-fetched.

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	waE2E "go.mau.fi/whatsmeow/proto/waE2E"
	"google.golang.org/protobuf/proto"
)

func imageMsg() *waE2E.Message {
	return &waE2E.Message{ImageMessage: &waE2E.ImageMessage{
		Mimetype: proto.String("image/jpeg")}}
}

func TestMediaCacheNameIdentity(t *testing.T) {
	// The full sanitized id is the identity — NOT the old
	// 12-char fragment (two different messages must never
	// collide into one file).
	name := mediaCacheName("A50946F5F931EXTRA", ".mp4")
	if name != "A50946F5F931EXTRA.mp4" {
		t.Fatalf("name = %q, want the full id + ext", name)
	}
	if got := mediaCacheName("a b@c/d", ".jpg"); got != "abcd.jpg" {
		t.Fatalf("sanitize: got %q", got)
	}
	if got := mediaCacheName("", ".jpg"); got != "" {
		t.Fatalf("empty identity must yield no name, got %q", got)
	}
	long := mediaCacheName(strings.Repeat("x", 100), ".bin")
	if len(long) != 64+len(".bin") {
		t.Fatalf("cap: name len = %d", len(long))
	}
}

// The writer-layer pin: an identity whose file already exists
// is served from the cache.  The bridge here has NO whatsmeow
// client at all — any download attempt would panic — so a
// returned row proves no fetch happened.
func TestDownloadMediaOfServesExistingFile(t *testing.T) {
	dir := t.TempDir()
	b := &bridge{mediaDir: dir}
	p := filepath.Join(dir, "MSGID123.jpg")
	if err := os.WriteFile(p, []byte("jpeg-bytes"), 0o644); err != nil {
		t.Fatal(err)
	}
	row := b.downloadMediaOf(context.Background(), imageMsg(),
		"MSGID123")
	if row == nil || row["path"] != p {
		t.Fatalf("row = %v, want the existing file %s", row, p)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 {
		t.Fatalf("cache holds %d files, want exactly 1", len(entries))
	}
	data, err := os.ReadFile(p)
	if err != nil || string(data) != "jpeg-bytes" {
		t.Fatalf("cached file was rewritten: %q, %v", data, err)
	}
}

// Concurrent duplicate triggers of one identity (a redelivery
// racing a sweep) still resolve to the one existing file.
func TestDownloadMediaOfConcurrentDuplicates(t *testing.T) {
	dir := t.TempDir()
	b := &bridge{mediaDir: dir}
	p := filepath.Join(dir, "RACEID42.jpg")
	if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	paths := make([]string, 8)
	for i := range paths {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			row := b.downloadMediaOf(context.Background(),
				imageMsg(), "RACEID42")
			if row != nil {
				paths[i], _ = row["path"].(string)
			}
		}(i)
	}
	wg.Wait()
	for i, got := range paths {
		if got != p {
			t.Fatalf("goroutine %d got %q, want %q", i, got, p)
		}
	}
	entries, _ := os.ReadDir(dir)
	if len(entries) != 1 {
		t.Fatalf("cache holds %d files, want exactly 1", len(entries))
	}
}

func TestProbeMediaMissThenHit(t *testing.T) {
	dir := t.TempDir()
	b := &bridge{mediaDir: dir}
	if _, ok := b.probeMedia(imageMsg(), "NOPE1"); ok {
		t.Fatal("probe hit on an empty cache")
	}
	p := filepath.Join(dir, "NOPE1.jpg")
	if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	row, ok := b.probeMedia(imageMsg(), "NOPE1")
	if !ok || row["path"] != p || row["kind"] != "image" {
		t.Fatalf("probe = %v, %v", row, ok)
	}
	// A zero-byte file is a failed write, never a cache hit.
	empty := filepath.Join(dir, "ZERO1.jpg")
	if err := os.WriteFile(empty, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, ok := b.probeMedia(imageMsg(), "ZERO1"); ok {
		t.Fatal("probe served a zero-byte file")
	}
}
