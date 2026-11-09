package main

import (
	"encoding/json"
	"strings"
	"testing"
)

// The protocol shapes the Python organ depends on: request
// decoding, response envelope, chord parsing. These run on every
// platform — the bridge must behave identically in shape wherever
// it is built.

func TestParseChord(t *testing.T) {
	mods, key, err := parseChord("ctrl+shift+l")
	if err != nil {
		t.Fatal(err)
	}
	if key != "l" || len(mods) != 2 || mods[0] != "ctrl" || mods[1] != "shift" {
		t.Fatalf("got mods=%v key=%q", mods, key)
	}
	mods, key, err = parseChord("Return")
	if err != nil || key != "Return" || len(mods) != 0 {
		t.Fatalf("bare key: mods=%v key=%q err=%v", mods, key, err)
	}
	if _, _, err := parseChord("ctrl+"); err == nil {
		t.Fatal("empty part must error")
	}
}

func TestDispatchUnknownMethod(t *testing.T) {
	drv := &fakeDriver{}
	var req request
	if err := json.Unmarshal([]byte(`{"id":7,"method":"nope","params":{}}`), &req); err != nil {
		t.Fatal(err)
	}
	if _, err := dispatch(drv, req); err == nil || !strings.Contains(err.Error(), "unknown method") {
		t.Fatalf("expected unknown method error, got %v", err)
	}
}

func TestDispatchProbeShape(t *testing.T) {
	drv := &fakeDriver{}
	var req request
	if err := json.Unmarshal([]byte(`{"id":1,"method":"probe"}`), &req); err != nil {
		t.Fatal(err)
	}
	res, err := dispatch(drv, req)
	if err != nil {
		t.Fatal(err)
	}
	info, ok := res.(ProbeInfo)
	if !ok {
		t.Fatalf("probe result type %T", res)
	}
	raw, err := json.Marshal(response{ID: 1, OK: true, Result: info})
	if err != nil {
		t.Fatal(err)
	}
	var env map[string]interface{}
	if err := json.Unmarshal(raw, &env); err != nil {
		t.Fatal(err)
	}
	if env["ok"] != true {
		t.Fatalf("envelope not ok: %s", raw)
	}
	result := env["result"].(map[string]interface{})
	for _, field := range []string{"protocol", "platform", "backend", "available", "width", "height", "features"} {
		if _, ok := result[field]; !ok {
			t.Fatalf("probe result missing %q: %s", field, raw)
		}
	}
}

func TestDispatchScreenshotBase64(t *testing.T) {
	drv := &fakeDriver{}
	var req request
	if err := json.Unmarshal([]byte(`{"id":2,"method":"screenshot"}`), &req); err != nil {
		t.Fatal(err)
	}
	res, err := dispatch(drv, req)
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := json.Marshal(res)
	// PNG magic "\x89PNG" base64-prefix is iVBORw0K…
	if !strings.Contains(string(raw), "iVBORw0K") {
		t.Fatalf("screenshot data is not base64 PNG: %s", raw)
	}
}

type fakeDriver struct{}

func (f *fakeDriver) Probe() ProbeInfo {
	return ProbeInfo{Protocol: 1, Platform: "test", Backend: "fake",
		Available: true, Width: 800, Height: 600, Features: []string{"screenshot"}}
}
func (f *fakeDriver) Ping() error { return nil }
func (f *fakeDriver) Screenshot() ([]byte, int, int, error) {
	return []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'}, 800, 600, nil
}
func (f *fakeDriver) Windows() ([]WindowInfo, uint32, error) { return nil, 0, nil }
func (f *fakeDriver) Pointer() (int, int, error)             { return 0, 0, nil }
func (f *fakeDriver) MouseMove(x, y int) error               { return nil }
func (f *fakeDriver) MouseClick(x, y int, b string, c int) error {
	return nil
}
func (f *fakeDriver) MouseDrag(x1, y1, x2, y2 int, b string) error { return nil }
func (f *fakeDriver) MouseScroll(x, y, dx, dy int) error           { return nil }
func (f *fakeDriver) Key(chord string) error                       { return nil }
func (f *fakeDriver) Type(text string) (int, error)                { return len(text), nil }
func (f *fakeDriver) Focus(window uint32) error                    { return nil }
func (f *fakeDriver) Launch(argv []string, cwd string) (int, error) {
	return 4242, nil
}
func (f *fakeDriver) Kill(pid int) error            { return nil }
func (f *fakeDriver) CloseWindow(window uint32) error { return nil }
func (f *fakeDriver) Close()                        {}
