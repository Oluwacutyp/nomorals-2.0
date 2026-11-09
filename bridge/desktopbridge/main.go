// desktop-bridge: Devon's own desktop driver.
//
// One half of the computer organ's driver layer. The Python organ
// (nomorals/computer) speaks ONE JSON-lines protocol over this
// process's stdio; this binary implements that protocol against the
// host's real desktop facilities — Win32 on Windows, a minimal
// hand-rolled X11 client on Linux — with no third-party automation
// library anywhere in the chain. Protocol version 1.
//
// Request:  {"id": <int>, "method": <string>, "params": {...}}
// Response: {"id": <int>, "ok": true,  "result": {...}}
//           {"id": <int>, "ok": false, "error": "..."}
// One response line per request line. Requests are handled serially.
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// WindowInfo is one top-level window as the perception layer sees it.
type WindowInfo struct {
	ID      uint32 `json:"id"`
	Title   string `json:"title"`
	X       int    `json:"x"`
	Y       int    `json:"y"`
	Width   int    `json:"width"`
	Height  int    `json:"height"`
	Focused bool   `json:"focused"`
	PID     int    `json:"pid,omitempty"`
}

// ProbeInfo describes the backend's reach on this host.
type ProbeInfo struct {
	Protocol  int      `json:"protocol"`
	Platform  string   `json:"platform"`
	Backend   string   `json:"backend"`
	Available bool     `json:"available"`
	Reason    string   `json:"reason,omitempty"`
	Display   string   `json:"display,omitempty"`
	Width     int      `json:"width"`
	Height    int      `json:"height"`
	Features  []string `json:"features"`
}

// Driver is the platform backend. Every method must be safe to call
// after Probe reports Available=false — it should return an error,
// never panic.
type Driver interface {
	Probe() ProbeInfo
	Ping() error
	// Screenshot returns a PNG of the full desktop.
	Screenshot() (png []byte, width, height int, err error)
	Windows() ([]WindowInfo, uint32, error)
	Pointer() (int, int, error)
	MouseMove(x, y int) error
	MouseClick(x, y int, button string, clicks int) error
	MouseDrag(x1, y1, x2, y2 int, button string) error
	MouseScroll(x, y, dx, dy int) error
	// Key sends one chord: "Return", "ctrl+l", "alt+F4", "shift+Tab".
	Key(chord string) error
	// Type enters literal text.
	Type(text string) (int, error)
	Focus(window uint32) error
	Launch(argv []string, cwd string) (int, error)
	Kill(pid int) error
	CloseWindow(window uint32) error
	Close()
}

type request struct {
	ID     int             `json:"id"`
	Method string          `json:"method"`
	Params json.RawMessage `json:"params"`
}

type response struct {
	ID     int         `json:"id"`
	OK     bool        `json:"ok"`
	Result interface{} `json:"result,omitempty"`
	Error  string      `json:"error,omitempty"`
}

func main() {
	drv := newDriver()
	defer drv.Close()
	sc := bufio.NewScanner(os.Stdin)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	out := bufio.NewWriter(os.Stdout)
	enc := func(v interface{}) {
		b, err := json.Marshal(v)
		if err != nil {
			return
		}
		out.Write(b)
		out.WriteByte('\n')
		out.Flush()
	}
	for sc.Scan() {
		line := sc.Bytes()
		if len(strings.TrimSpace(string(line))) == 0 {
			continue
		}
		var req request
		if err := json.Unmarshal(line, &req); err != nil {
			enc(response{ID: -1, OK: false, Error: "bad request json: " + err.Error()})
			continue
		}
		res, err := dispatch(drv, req)
		if err != nil {
			enc(response{ID: req.ID, OK: false, Error: err.Error()})
			continue
		}
		enc(response{ID: req.ID, OK: true, Result: res})
		if req.Method == "shutdown" {
			return
		}
	}
}

func params(req request, v interface{}) error {
	if len(req.Params) == 0 {
		return nil
	}
	if err := json.Unmarshal(req.Params, v); err != nil {
		return fmt.Errorf("bad params for %s: %w", req.Method, err)
	}
	return nil
}

func dispatch(drv Driver, req request) (interface{}, error) {
	switch req.Method {
	case "probe":
		return drv.Probe(), nil
	case "ping":
		if err := drv.Ping(); err != nil {
			return nil, err
		}
		return map[string]interface{}{"pong": true}, nil
	case "screenshot":
		png, w, h, err := drv.Screenshot()
		if err != nil {
			return nil, err
		}
		return map[string]interface{}{
			"width": w, "height": h, "format": "png",
			"data": png, // []byte marshals as base64
		}, nil
	case "windows":
		wins, focused, err := drv.Windows()
		if err != nil {
			return nil, err
		}
		if wins == nil {
			wins = []WindowInfo{}
		}
		return map[string]interface{}{"windows": wins, "focused_id": focused}, nil
	case "pointer":
		x, y, err := drv.Pointer()
		if err != nil {
			return nil, err
		}
		return map[string]interface{}{"x": x, "y": y}, nil
	case "mouse_move":
		var p struct {
			X int `json:"x"`
			Y int `json:"y"`
		}
		if err := params(req, &p); err != nil {
			return nil, err
		}
		if err := drv.MouseMove(p.X, p.Y); err != nil {
			return nil, err
		}
		x, y, err := drv.Pointer()
		if err != nil {
			return nil, err
		}
		return map[string]interface{}{"x": x, "y": y}, nil
	case "mouse_click":
		var p struct {
			X      int    `json:"x"`
			Y      int    `json:"y"`
			Button string `json:"button"`
			Clicks int    `json:"clicks"`
		}
		if err := params(req, &p); err != nil {
			return nil, err
		}
		if p.Button == "" {
			p.Button = "left"
		}
		if p.Clicks <= 0 {
			p.Clicks = 1
		}
		if err := drv.MouseClick(p.X, p.Y, p.Button, p.Clicks); err != nil {
			return nil, err
		}
		return map[string]interface{}{"clicked": true}, nil
	case "mouse_drag":
		var p struct {
			X1     int    `json:"x1"`
			Y1     int    `json:"y1"`
			X2     int    `json:"x2"`
			Y2     int    `json:"y2"`
			Button string `json:"button"`
		}
		if err := params(req, &p); err != nil {
			return nil, err
		}
		if p.Button == "" {
			p.Button = "left"
		}
		if err := drv.MouseDrag(p.X1, p.Y1, p.X2, p.Y2, p.Button); err != nil {
			return nil, err
		}
		return map[string]interface{}{"dragged": true}, nil
	case "mouse_scroll":
		var p struct {
			X  int `json:"x"`
			Y  int `json:"y"`
			DX int `json:"dx"`
			DY int `json:"dy"`
		}
		if err := params(req, &p); err != nil {
			return nil, err
		}
		if err := drv.MouseScroll(p.X, p.Y, p.DX, p.DY); err != nil {
			return nil, err
		}
		return map[string]interface{}{"scrolled": true}, nil
	case "key":
		var p struct {
			Keys string `json:"keys"`
		}
		if err := params(req, &p); err != nil {
			return nil, err
		}
		if strings.TrimSpace(p.Keys) == "" {
			return nil, fmt.Errorf("key: empty chord")
		}
		if err := drv.Key(p.Keys); err != nil {
			return nil, err
		}
		return map[string]interface{}{"sent": p.Keys}, nil
	case "type":
		var p struct {
			Text string `json:"text"`
		}
		if err := params(req, &p); err != nil {
			return nil, err
		}
		n, err := drv.Type(p.Text)
		if err != nil {
			return nil, err
		}
		return map[string]interface{}{"typed": n}, nil
	case "focus":
		var p struct {
			Window uint32 `json:"window"`
		}
		if err := params(req, &p); err != nil {
			return nil, err
		}
		if err := drv.Focus(p.Window); err != nil {
			return nil, err
		}
		return map[string]interface{}{"focused": p.Window}, nil
	case "launch":
		var p struct {
			Argv []string `json:"argv"`
			CWD  string   `json:"cwd"`
		}
		if err := params(req, &p); err != nil {
			return nil, err
		}
		if len(p.Argv) == 0 {
			return nil, fmt.Errorf("launch: empty argv")
		}
		pid, err := drv.Launch(p.Argv, p.CWD)
		if err != nil {
			return nil, err
		}
		return map[string]interface{}{"pid": pid}, nil
	case "kill":
		var p struct {
			PID int `json:"pid"`
		}
		if err := params(req, &p); err != nil {
			return nil, err
		}
		if err := drv.Kill(p.PID); err != nil {
			return nil, err
		}
		return map[string]interface{}{"killed": p.PID}, nil
	case "close_window":
		var p struct {
			Window uint32 `json:"window"`
		}
		if err := params(req, &p); err != nil {
			return nil, err
		}
		if err := drv.CloseWindow(p.Window); err != nil {
			return nil, err
		}
		return map[string]interface{}{"closed": p.Window}, nil
	case "shutdown":
		return map[string]interface{}{"bye": true}, nil
	}
	return nil, fmt.Errorf("unknown method %q", req.Method)
}

// parseChord splits "ctrl+shift+l" into modifiers and the main key.
// Shared by both backends; resolution to platform codes is theirs.
func parseChord(chord string) (mods []string, key string, err error) {
	parts := strings.Split(chord, "+")
	for i, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			return nil, "", fmt.Errorf("key: empty part in chord %q", chord)
		}
		if i == len(parts)-1 {
			key = p
		} else {
			mods = append(mods, strings.ToLower(p))
		}
	}
	return mods, key, nil
}
