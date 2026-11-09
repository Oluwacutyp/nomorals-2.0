//go:build linux

package main

// Linux backend: a minimal, hand-rolled X11 client.
//
// The desktop speaks the X11 wire protocol directly over the
// display's unix socket — connection setup, GetImage for
// screenshots, QueryTree/GetProperty for the window list, and the
// XTEST extension for input. No X library, no xdotool, no
// third-party automation stack: Devon's own code end to end, the
// same posture as the Windows backend's raw Win32 calls.

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	"image/png"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ---------------------------------------------------------------- X11 core

type xError struct {
	Code      byte
	MajorOp   byte
	MinorOp   uint16
	BadValue  uint32
	Seq       uint16
}

func (e *xError) Error() string {
	return fmt.Sprintf("x11 error code=%d major=%d minor=%d badvalue=%d",
		e.Code, e.MajorOp, e.MinorOp, e.BadValue)
}

type xConn struct {
	mu       sync.Mutex
	conn     net.Conn
	seq      uint32
	root     uint32
	rootW    int
	rootH    int
	rootDep  byte
	minKey   byte
	maxKey   byte
	xtestOp  byte
	atoms    map[string]uint32
	keymap   []uint32 // keysyms, (maxKey-minKey+1) * per
	keyPer   int
	spareKey byte // remapped on demand for unmapped keysyms
	spareSym uint32
}

func pad4(n int) int { return (4 - n%4) % 4 }

func parseDisplay(display string) (network, address string, num int, err error) {
	d := display
	if d == "" {
		return "", "", 0, fmt.Errorf("DISPLAY is empty")
	}
	// forms: :99, :99.0, unix:99, host:99, host:99.0
	colon := strings.LastIndex(d, ":")
	if colon < 0 {
		return "", "", 0, fmt.Errorf("cannot parse DISPLAY %q", display)
	}
	host := d[:colon]
	rest := d[colon+1:]
	if dot := strings.Index(rest, "."); dot >= 0 {
		rest = rest[:dot]
	}
	num, err = strconv.Atoi(rest)
	if err != nil {
		return "", "", 0, fmt.Errorf("cannot parse DISPLAY %q", display)
	}
	if host == "" || host == "unix" {
		return "unix", filepath.Join("/tmp/.X11-unix", "X"+strconv.Itoa(num)), num, nil
	}
	return "tcp", net.JoinHostPort(host, strconv.Itoa(6000+num)), num, nil
}

type xauthEntry struct {
	family  uint16
	address []byte
	number  string
	name    string
	data    []byte
}

func readXauthority(path string) []xauthEntry {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var out []xauthEntry
	n := 0
	readStr := func() ([]byte, bool) {
		if n+2 > len(raw) {
			return nil, false
		}
		l := int(binary.BigEndian.Uint16(raw[n:]))
		n += 2
		if n+l > len(raw) {
			return nil, false
		}
		b := raw[n : n+l]
		n += l
		return b, true
	}
	for n < len(raw) {
		if n+2 > len(raw) {
			break
		}
		fam := binary.BigEndian.Uint16(raw[n:])
		n += 2
		addr, ok1 := readStr()
		num, ok2 := readStr()
		name, ok3 := readStr()
		data, ok4 := readStr()
		if !(ok1 && ok2 && ok3 && ok4) {
			break
		}
		out = append(out, xauthEntry{family: fam, address: addr,
			number: string(num), name: string(name), data: data})
	}
	return out
}

func authForDisplay(num int) (name string, data []byte) {
	path := os.Getenv("XAUTHORITY")
	if path == "" {
		home, _ := os.UserHomeDir()
		path = filepath.Join(home, ".Xauthority")
	}
	host, _ := os.Hostname()
	numStr := strconv.Itoa(num)
	for _, e := range readXauthority(path) {
		if e.name != "MIT-MAGIC-COOKIE-1" || e.number != numStr {
			continue
		}
		// FamilyLocal (256) with our hostname, or FamilyWild (65535).
		if e.family == 65535 || (e.family == 256 && string(e.address) == host) {
			return e.name, e.data
		}
	}
	return "", nil
}

func dialX(display string) (*xConn, error) {
	network, address, num, err := parseDisplay(display)
	if err != nil {
		return nil, err
	}
	d := net.Dialer{Timeout: 5 * time.Second}
	raw, err := d.Dial(network, address)
	if err != nil {
		return nil, fmt.Errorf("cannot connect to display %s: %w", display, err)
	}
	c := &xConn{conn: raw, atoms: map[string]uint32{}}
	name, data := authForDisplay(num)
	var setup bytes.Buffer
	setup.WriteByte('l')
	setup.WriteByte(0)
	put16 := func(v uint16) {
		var b [2]byte
		binary.LittleEndian.PutUint16(b[:], v)
		setup.Write(b[:])
	}
	put16(11)
	put16(0)
	put16(uint16(len(name)))
	put16(uint16(len(data)))
	put16(0)
	setup.WriteString(name)
	setup.Write(make([]byte, pad4(len(name))))
	setup.Write(data)
	setup.Write(make([]byte, pad4(len(data))))
	if _, err := raw.Write(setup.Bytes()); err != nil {
		raw.Close()
		return nil, fmt.Errorf("x11 setup write: %w", err)
	}
	head := make([]byte, 8)
	if err := readFull(raw, head); err != nil {
		raw.Close()
		return nil, fmt.Errorf("x11 setup read: %w", err)
	}
	if head[0] == 0 {
		reasonLen := int(head[1])
		body := make([]byte, 4*int(binary.LittleEndian.Uint16(head[6:8])))
		_ = readFull(raw, body)
		raw.Close()
		reason := ""
		if reasonLen <= len(body) {
			reason = string(body[:reasonLen])
		}
		return nil, fmt.Errorf("x11 server refused the connection: %s", reason)
	}
	if head[0] != 1 {
		raw.Close()
		return nil, fmt.Errorf("x11 setup: unexpected status byte %d", head[0])
	}
	extra := 4 * int(binary.LittleEndian.Uint16(head[6:8]))
	body := make([]byte, extra)
	if err := readFull(raw, body); err != nil {
		raw.Close()
		return nil, fmt.Errorf("x11 setup body: %w", err)
	}
	if len(body) < 32 {
		raw.Close()
		return nil, fmt.Errorf("x11 setup body too short (%d bytes)", len(body))
	}
	c.minKey = body[26]
	c.maxKey = body[27]
	vendorLen := int(binary.LittleEndian.Uint16(body[16:18]))
	numFormats := int(body[21])
	off := 32 + vendorLen + pad4(vendorLen) + 8*numFormats
	if off+24 > len(body) {
		raw.Close()
		return nil, fmt.Errorf("x11 setup: screen record out of range")
	}
	scr := body[off:]
	c.root = binary.LittleEndian.Uint32(scr[0:4])
	c.rootW = int(binary.LittleEndian.Uint16(scr[20:22]))
	c.rootH = int(binary.LittleEndian.Uint16(scr[22:24]))
	c.rootDep = scr[38]
	// XTEST extension?
	op, present, qerr := c.queryExtension("XTEST")
	if qerr == nil && present {
		c.xtestOp = op
	}
	return c, nil
}

func readFull(conn net.Conn, buf []byte) error {
	for len(buf) > 0 {
		n, err := conn.Read(buf)
		if err != nil {
			return err
		}
		buf = buf[n:]
	}
	return nil
}

// do sends one request. When wantReply is true it reads until the
// reply for this request's sequence arrives and returns the full
// reply packet. Errors reported by the server are returned as
// *xError. No-reply requests are followed by a GetInputFocus sync
// so a server error is attributed before we move on.
func (c *xConn) do(req []byte, wantReply bool) ([]byte, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.doLocked(req, wantReply)
}

func (c *xConn) doLocked(req []byte, wantReply bool) ([]byte, error) {
	c.seq++
	mySeq := uint16(c.seq)
	c.conn.SetDeadline(time.Now().Add(30 * time.Second))
	defer c.conn.SetDeadline(time.Time{})
	if _, err := c.conn.Write(req); err != nil {
		return nil, fmt.Errorf("x11 write: %w", err)
	}
	if !wantReply {
		// Sync point: GetInputFocus always replies; any error the
		// server queued for our request surfaces before it.
		c.seq++
		syncSeq := uint16(c.seq)
		if _, err := c.conn.Write([]byte{43, 0, 1, 0}); err != nil {
			return nil, fmt.Errorf("x11 sync write: %w", err)
		}
		for {
			pkt, err := c.readPacket()
			if err != nil {
				return nil, err
			}
			if pkt[0] == 0 {
				xe := parseXError(pkt)
				if xe.Seq == mySeq {
					return nil, xe
				}
				continue
			}
			if uint16(pkt[2])|uint16(pkt[3])<<8 == syncSeq {
				return nil, nil
			}
		}
	}
	for {
		pkt, err := c.readPacket()
		if err != nil {
			return nil, err
		}
		if pkt[0] == 0 {
			xe := parseXError(pkt)
			if xe.Seq == mySeq {
				return nil, xe
			}
			continue
		}
		if uint16(pkt[2])|uint16(pkt[3])<<8 == mySeq {
			return pkt, nil
		}
	}
}

func parseXError(pkt []byte) *xError {
	xe := &xError{Code: pkt[1]}
	xe.Seq = uint16(pkt[2]) | uint16(pkt[3])<<8
	if len(pkt) >= 12 {
		xe.BadValue = binary.LittleEndian.Uint32(pkt[4:8])
		xe.MinorOp = binary.LittleEndian.Uint16(pkt[8:10])
		xe.MajorOp = pkt[10]
	}
	return xe
}

// readPacket reads one generic packet (error, reply, or event).
func (c *xConn) readPacket() ([]byte, error) {
	head := make([]byte, 32)
	if err := readFull(c.conn, head); err != nil {
		return nil, fmt.Errorf("x11 read: %w", err)
	}
	if head[0] == 1 {
		extra := 4 * int(binary.LittleEndian.Uint32(head[4:8]))
		if extra > 0 {
			rest := make([]byte, extra)
			if err := readFull(c.conn, rest); err != nil {
				return nil, fmt.Errorf("x11 read body: %w", err)
			}
			return append(head, rest...), nil
		}
	}
	return head, nil
}

func reqHead(op byte, b1 byte, length uint16) []byte {
	b := make([]byte, 4)
	b[0] = op
	b[1] = b1
	binary.LittleEndian.PutUint16(b[2:4], length)
	return b
}

func u32(v uint32) []byte {
	b := make([]byte, 4)
	binary.LittleEndian.PutUint32(b, v)
	return b
}

func (c *xConn) internAtom(name string, onlyIfExists bool) (uint32, error) {
	if v, ok := c.atoms[name]; ok {
		return v, nil
	}
	req := reqHead(16, boolByte(onlyIfExists), uint16(2+(len(name)+pad4(len(name)))/4))
	req = append(req, []byte{byte(len(name)), byte(len(name) >> 8), 0, 0}...)
	req = append(req, name...)
	req = append(req, make([]byte, pad4(len(name)))...)
	rep, err := c.do(req, true)
	if err != nil {
		return 0, err
	}
	atom := binary.LittleEndian.Uint32(rep[8:12])
	if atom != 0 {
		c.atoms[name] = atom
	}
	return atom, nil
}

func boolByte(b bool) byte {
	if b {
		return 1
	}
	return 0
}

func (c *xConn) queryExtension(name string) (byte, bool, error) {
	req := reqHead(98, 0, uint16(2+(len(name)+pad4(len(name)))/4))
	req = append(req, []byte{byte(len(name)), byte(len(name) >> 8), 0, 0}...)
	req = append(req, name...)
	req = append(req, make([]byte, pad4(len(name)))...)
	rep, err := c.do(req, true)
	if err != nil {
		return 0, false, err
	}
	// Reply layout: present @8, major-opcode @9, first-event @10.
	return rep[9], rep[8] != 0, nil
}

func (c *xConn) getInputFocus() (uint32, error) {
	rep, err := c.do([]byte{43, 0, 1, 0}, true)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(rep[8:12]), nil
}

func (c *xConn) queryPointer() (int, int, uint32, error) {
	req := reqHead(38, 0, 2)
	req = append(req, u32(c.root)...)
	rep, err := c.do(req, true)
	if err != nil {
		return 0, 0, 0, err
	}
	x := int(int16(binary.LittleEndian.Uint16(rep[16:18])))
	y := int(int16(binary.LittleEndian.Uint16(rep[18:20])))
	child := binary.LittleEndian.Uint32(rep[12:16])
	return x, y, child, nil
}

func (c *xConn) queryTree(win uint32) (parent uint32, children []uint32, err error) {
	req := reqHead(15, 0, 2)
	req = append(req, u32(win)...)
	rep, err := c.do(req, true)
	if err != nil {
		return 0, nil, err
	}
	parent = binary.LittleEndian.Uint32(rep[12:16])
	n := int(binary.LittleEndian.Uint16(rep[16:18]))
	for i := 0; i < n && 32+i*4+4 <= len(rep); i++ {
		children = append(children, binary.LittleEndian.Uint32(rep[32+i*4:]))
	}
	return parent, children, nil
}

type winAttrs struct {
	MapState        byte // 0 unmapped, 1 unviewable, 2 viewable
	OverrideRedir   bool
}

func (c *xConn) getWindowAttributes(win uint32) (winAttrs, error) {
	req := reqHead(3, 0, 2)
	req = append(req, u32(win)...)
	rep, err := c.do(req, true)
	if err != nil {
		return winAttrs{}, err
	}
	return winAttrs{MapState: rep[26], OverrideRedir: rep[27] != 0}, nil
}

func (c *xConn) getGeometry(win uint32) (x, y, w, h int, err error) {
	req := reqHead(14, 0, 2)
	req = append(req, u32(win)...)
	rep, err := c.do(req, true)
	if err != nil {
		return 0, 0, 0, 0, err
	}
	x = int(int16(binary.LittleEndian.Uint16(rep[12:14])))
	y = int(int16(binary.LittleEndian.Uint16(rep[14:16])))
	w = int(binary.LittleEndian.Uint16(rep[16:18]))
	h = int(binary.LittleEndian.Uint16(rep[18:20]))
	return x, y, w, h, nil
}

// absGeometry walks the parent chain summing offsets to the root.
func (c *xConn) absGeometry(win uint32) (int, int, int, int, error) {
	x, y, w, h, err := c.getGeometry(win)
	if err != nil {
		return 0, 0, 0, 0, err
	}
	cur := win
	for depth := 0; depth < 32; depth++ {
		parent, _, err := c.queryTree(cur)
		if err != nil || parent == 0 || parent == c.root {
			break
		}
		px, py, _, _, err := c.getGeometry(parent)
		if err != nil {
			break
		}
		x += px
		y += py
		cur = parent
	}
	return x, y, w, h, nil
}

func (c *xConn) getProperty(win, property, typ uint32) ([]byte, byte, error) {
	req := reqHead(20, 0, 6)
	req = append(req, u32(win)...)
	req = append(req, u32(property)...)
	req = append(req, u32(typ)...)
	req = append(req, u32(0)...)       // long-offset
	req = append(req, u32(1<<24)...)   // long-length (generous)
	rep, err := c.do(req, true)
	if err != nil {
		return nil, 0, err
	}
	format := rep[1]
	valLen := int(binary.LittleEndian.Uint32(rep[16:20]))
	n := valLen * int(format) / 8
	if 32+n > len(rep) {
		n = len(rep) - 32
		if n < 0 {
			n = 0
		}
	}
	return rep[32 : 32+n], format, nil
}

func (c *xConn) windowTitle(win uint32) string {
	if atom, err := c.internAtom("_NET_WM_NAME", true); err == nil && atom != 0 {
		if utf8Atom, err := c.internAtom("UTF8_STRING", true); err == nil && utf8Atom != 0 {
			if data, _, err := c.getProperty(win, atom, utf8Atom); err == nil && len(data) > 0 {
				return string(data)
			}
		}
	}
	// WM_NAME = atom 39, XA_STRING = 31
	if data, _, err := c.getProperty(win, 39, 31); err == nil && len(data) > 0 {
		return string(data)
	}
	return ""
}

func (c *xConn) windowPID(win uint32) int {
	atom, err := c.internAtom("_NET_WM_PID", true)
	if err != nil || atom == 0 {
		return 0
	}
	data, format, err := c.getProperty(win, atom, 6) // CARDINAL
	if err != nil || format != 32 || len(data) < 4 {
		return 0
	}
	return int(binary.LittleEndian.Uint32(data[0:4]))
}

func (c *xConn) getImage(x, y, w, h int) (data []byte, depth byte, err error) {
	req := reqHead(73, 2, 5) // ZPixmap
	req = append(req, u32(c.root)...)
	var xy [8]byte
	binary.LittleEndian.PutUint16(xy[0:2], uint16(int16(x)))
	binary.LittleEndian.PutUint16(xy[2:4], uint16(int16(y)))
	binary.LittleEndian.PutUint16(xy[4:6], uint16(w))
	binary.LittleEndian.PutUint16(xy[6:8], uint16(h))
	req = append(req, xy[:]...)
	req = append(req, u32(0xFFFFFFFF)...)
	rep, err := c.do(req, true)
	if err != nil {
		return nil, 0, err
	}
	depth = rep[1]
	if len(rep) > 32 {
		data = rep[32:]
	}
	return data, depth, nil
}

func (c *xConn) loadKeymap() error {
	if c.keymap != nil {
		return nil
	}
	count := int(c.maxKey) - int(c.minKey) + 1
	// GetKeyboardMapping: byte1 is a pad; first-keycode and count
	// sit after the length field.
	req := reqHead(101, 0, 2)
	req = append(req, c.minKey, byte(count), 0, 0)
	rep, err := c.do(req, true)
	if err != nil {
		return err
	}
	c.keyPer = int(rep[1])
	total := count * c.keyPer
	c.keymap = make([]uint32, 0, total)
	for i := 0; i < total && 32+i*4+4 <= len(rep); i++ {
		c.keymap = append(c.keymap, binary.LittleEndian.Uint32(rep[32+i*4:]))
	}
	return nil
}

func (c *xConn) changeKeyMapping(firstKey byte, syms []uint32, per int) error {
	req := reqHead(100, byte(1), uint16(2+len(syms)))
	req = append(req, firstKey, byte(per), 0, 0)
	for _, s := range syms {
		req = append(req, u32(s)...)
	}
	_, err := c.do(req, false)
	return err
}

// keycodeFor finds a keycode producing keysym at level 0 (no shift)
// or level 1 (shift). ok=false when the keysym is not mapped.
func (c *xConn) keycodeFor(keysym uint32) (code byte, shift bool, ok bool) {
	if c.keymap == nil {
		return 0, false, false
	}
	for i, sym := range c.keymap {
		if sym != keysym || sym == 0 {
			continue
		}
		level := i % c.keyPer
		if level == 0 {
			return c.minKey + byte(i/c.keyPer), false, true
		}
		if level == 1 {
			return c.minKey + byte(i/c.keyPer), true, true
		}
	}
	return 0, false, false
}

func (c *xConn) spareKeycode() (byte, error) {
	if c.spareKey != 0 {
		return c.spareKey, nil
	}
	if err := c.loadKeymap(); err != nil {
		return 0, err
	}
	for code := int(c.maxKey); code >= int(c.minKey); code-- {
		base := (code - int(c.minKey)) * c.keyPer
		empty := true
		for j := 0; j < c.keyPer && base+j < len(c.keymap); j++ {
			if c.keymap[base+j] != 0 {
				empty = false
				break
			}
		}
		if empty {
			c.spareKey = byte(code)
			return c.spareKey, nil
		}
	}
	return 0, fmt.Errorf("no spare keycode for remapping")
}

func (c *xConn) fakeInput(evType byte, detail byte, x, y int, useRoot bool) error {
	if c.xtestOp == 0 {
		return fmt.Errorf("XTEST extension is not available on this display")
	}
	req := make([]byte, 36)
	req[0] = c.xtestOp
	req[1] = 2 // XTestFakeInput
	binary.LittleEndian.PutUint16(req[2:4], 9)
	req[4] = evType
	req[5] = detail
	// time = CurrentTime (0) at offset 8
	if useRoot {
		binary.LittleEndian.PutUint32(req[12:16], c.root)
	}
	binary.LittleEndian.PutUint16(req[24:26], uint16(int16(x)))
	binary.LittleEndian.PutUint16(req[26:28], uint16(int16(y)))
	_, err := c.do(req, false)
	return err
}

const (
	evKeyPress    = 2
	evKeyRelease  = 3
	evBtnPress    = 4
	evBtnRelease  = 5
	evMotion      = 6
)

// ---------------------------------------------------------------- driver

type linuxDriver struct {
	display string
	xc      *xConn
	probe   *ProbeInfo
}

func newDriver() Driver {
	return &linuxDriver{display: os.Getenv("DISPLAY")}
}

func (d *linuxDriver) connect() (*xConn, error) {
	if d.xc != nil {
		return d.xc, nil
	}
	if d.display == "" {
		return nil, fmt.Errorf("no display: DISPLAY is not set")
	}
	xc, err := dialX(d.display)
	if err != nil {
		return nil, err
	}
	d.xc = xc
	return xc, nil
}

func (d *linuxDriver) Probe() ProbeInfo {
	info := ProbeInfo{
		Protocol: 1, Platform: "linux", Backend: "x11 (own client)",
		Display: d.display,
		Features: []string{"screenshot", "windows", "pointer", "mouse", "keyboard",
			"launch", "focus", "close_window"},
	}
	xc, err := d.connect()
	if err != nil {
		info.Available = false
		info.Reason = err.Error()
		d.probe = &info
		return info
	}
	info.Available = true
	info.Width = xc.rootW
	info.Height = xc.rootH
	if xc.xtestOp == 0 {
		info.Available = false
		info.Reason = "display has no XTEST extension; input is impossible"
	}
	d.probe = &info
	return info
}

func (d *linuxDriver) Ping() error {
	xc, err := d.connect()
	if err != nil {
		return err
	}
	_, err = xc.getInputFocus()
	return err
}

func (d *linuxDriver) Screenshot() ([]byte, int, int, error) {
	xc, err := d.connect()
	if err != nil {
		return nil, 0, 0, err
	}
	data, depth, err := xc.getImage(0, 0, xc.rootW, xc.rootH)
	if err != nil {
		return nil, 0, 0, err
	}
	if depth != 24 && depth != 32 {
		return nil, 0, 0, fmt.Errorf("unsupported root depth %d (want 24 or 32)", depth)
	}
	img := image.NewRGBA(image.Rect(0, 0, xc.rootW, xc.rootH))
	stride := xc.rootW * 4
	for y := 0; y < xc.rootH; y++ {
		row := data[y*stride:]
		for x := 0; x < xc.rootW && (x+1)*4 <= len(row); x++ {
			src := row[x*4:]
			dst := img.Pix[y*img.Stride+x*4:]
			dst[0] = src[2] // R
			dst[1] = src[1] // G
			dst[2] = src[0] // B
			dst[3] = 255
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return nil, 0, 0, err
	}
	return buf.Bytes(), xc.rootW, xc.rootH, nil
}

func (d *linuxDriver) Windows() ([]WindowInfo, uint32, error) {
	xc, err := d.connect()
	if err != nil {
		return nil, 0, err
	}
	_, children, err := xc.queryTree(xc.root)
	if err != nil {
		return nil, 0, err
	}
	focusWin, _ := xc.getInputFocus()
	// Top-level ancestor of the focus window, if any.
	focusedTop := uint32(0)
	if focusWin != 0 && focusWin != 1 && focusWin != xc.root {
		cur := focusWin
		for depth := 0; depth < 32; depth++ {
			parent, _, err := xc.queryTree(cur)
			if err != nil || parent == 0 {
				break
			}
			if parent == xc.root {
				focusedTop = cur
				break
			}
			cur = parent
		}
	}
	var out []WindowInfo
	for _, win := range children {
		attrs, err := xc.getWindowAttributes(win)
		if err != nil || attrs.MapState != 2 || attrs.OverrideRedir {
			continue
		}
		x, y, w, h, err := xc.absGeometry(win)
		if err != nil {
			continue
		}
		out = append(out, WindowInfo{
			ID: win, Title: xc.windowTitle(win),
			X: x, Y: y, Width: w, Height: h,
			Focused: win == focusedTop, PID: xc.windowPID(win),
		})
		if win == focusedTop {
			focusedTop = win
		}
	}
	return out, focusedTop, nil
}

func (d *linuxDriver) Pointer() (int, int, error) {
	xc, err := d.connect()
	if err != nil {
		return 0, 0, err
	}
	x, y, _, err := xc.queryPointer()
	return x, y, err
}

func (d *linuxDriver) MouseMove(x, y int) error {
	xc, err := d.connect()
	if err != nil {
		return err
	}
	return xc.fakeInput(evMotion, 0, x, y, true)
}

func buttonNumber(button string) (byte, error) {
	switch strings.ToLower(button) {
	case "left", "":
		return 1, nil
	case "middle":
		return 2, nil
	case "right":
		return 3, nil
	}
	return 0, fmt.Errorf("unknown button %q", button)
}

func (d *linuxDriver) MouseClick(x, y int, button string, clicks int) error {
	btn, err := buttonNumber(button)
	if err != nil {
		return err
	}
	if err := d.MouseMove(x, y); err != nil {
		return err
	}
	xc, _ := d.connect()
	for i := 0; i < clicks; i++ {
		if err := xc.fakeInput(evBtnPress, btn, x, y, true); err != nil {
			return err
		}
		time.Sleep(15 * time.Millisecond)
		if err := xc.fakeInput(evBtnRelease, btn, x, y, true); err != nil {
			return err
		}
		time.Sleep(15 * time.Millisecond)
	}
	return nil
}

func (d *linuxDriver) MouseDrag(x1, y1, x2, y2 int, button string) error {
	btn, err := buttonNumber(button)
	if err != nil {
		return err
	}
	xc, _ := d.connect()
	if err := d.MouseMove(x1, y1); err != nil {
		return err
	}
	if err := xc.fakeInput(evBtnPress, btn, x1, y1, true); err != nil {
		return err
	}
	steps := 12
	for i := 1; i <= steps; i++ {
		mx := x1 + (x2-x1)*i/steps
		my := y1 + (y2-y1)*i/steps
		if err := xc.fakeInput(evMotion, 0, mx, my, true); err != nil {
			return err
		}
		time.Sleep(10 * time.Millisecond)
	}
	return xc.fakeInput(evBtnRelease, btn, x2, y2, true)
}

func (d *linuxDriver) MouseScroll(x, y, dx, dy int) error {
	xc, err := d.connect()
	if err != nil {
		return err
	}
	if err := d.MouseMove(x, y); err != nil {
		return err
	}
	scroll := func(btn byte, times int) error {
		for i := 0; i < times; i++ {
			if err := xc.fakeInput(evBtnPress, btn, x, y, true); err != nil {
				return err
			}
			if err := xc.fakeInput(evBtnRelease, btn, x, y, true); err != nil {
				return err
			}
		}
		return nil
	}
	// X11: 4 = up, 5 = down, 6 = left, 7 = right.
	if dy > 0 {
		if err := scroll(5, dy); err != nil {
			return err
		}
	} else if dy < 0 {
		if err := scroll(4, -dy); err != nil {
			return err
		}
	}
	if dx > 0 {
		if err := scroll(7, dx); err != nil {
			return err
		}
	} else if dx < 0 {
		if err := scroll(6, -dx); err != nil {
			return err
		}
	}
	return nil
}

var namedKeysyms = map[string]uint32{
	"return": 0xFF0D, "enter": 0xFF0D, "kp_enter": 0xFF8D,
	"escape": 0xFF1B, "esc": 0xFF1B,
	"tab": 0xFF09, "backspace": 0xFF08, "delete": 0xFFFF, "del": 0xFFFF,
	"insert": 0xFF63, "home": 0xFF50, "end": 0xFF57,
	"pageup": 0xFF55, "page_up": 0xFF55, "prior": 0xFF55,
	"pagedown": 0xFF56, "page_down": 0xFF56, "next": 0xFF56,
	"left": 0xFF51, "up": 0xFF52, "right": 0xFF53, "down": 0xFF54,
	"space": 0x20, "minus": 0x2D, "equal": 0x3D,
	"shift": 0xFFE1, "shift_l": 0xFFE1, "shift_r": 0xFFE2,
	"ctrl": 0xFFE3, "control": 0xFFE3, "control_l": 0xFFE3, "control_r": 0xFFE4,
	"alt": 0xFFE9, "alt_l": 0xFFE9, "alt_r": 0xFFEA,
	"super": 0xFFEB, "win": 0xFFEB, "meta": 0xFFEB,
	"capslock": 0xFFE5, "numlock": 0xFF7F,
}

func keysymForName(name string) (uint32, error) {
	lower := strings.ToLower(name)
	if ks, ok := namedKeysyms[lower]; ok {
		return ks, nil
	}
	if strings.HasPrefix(lower, "f") && len(lower) <= 3 {
		if n, err := strconv.Atoi(lower[1:]); err == nil && n >= 1 && n <= 12 {
			return 0xFFBE + uint32(n-1), nil
		}
	}
	r := []rune(name)
	if len(r) == 1 {
		return keysymForRune(r[0]), nil
	}
	return 0, fmt.Errorf("unknown key name %q", name)
}

func keysymForRune(r rune) uint32 {
	switch r {
	case '\n', '\r':
		return 0xFF0D
	case '\t':
		return 0xFF09
	case '\b':
		return 0xFF08
	case 0x7F:
		return 0xFFFF
	}
	if r < 0x20 {
		return 0 // control characters handled by the caller via chords
	}
	if r <= 0xFF {
		return uint32(r) // Latin-1 keysyms are the codepoints
	}
	return 0x01000000 | uint32(r) // Unicode keysym convention
}

// pressKeysym presses (and releases) one keysym, remapping a spare
// keycode temporarily when the layout has no mapping for it.
func (d *linuxDriver) pressKeysym(keysym uint32) error {
	xc, err := d.connect()
	if err != nil {
		return err
	}
	if err := xc.loadKeymap(); err != nil {
		return err
	}
	code, shift, ok := xc.keycodeFor(keysym)
	if !ok {
		spare, err := xc.spareKeycode()
		if err != nil {
			return err
		}
		syms := make([]uint32, xc.keyPer)
		syms[0] = keysym
		if err := xc.changeKeyMapping(spare, syms, xc.keyPer); err != nil {
			return fmt.Errorf("remap spare keycode: %w", err)
		}
		code, shift = spare, false
		defer func() {
			zeros := make([]uint32, xc.keyPer)
			_ = xc.changeKeyMapping(spare, zeros, xc.keyPer)
		}()
	}
	var shiftCode byte
	if shift {
		sc, _, ok := xc.keycodeFor(0xFFE1)
		if !ok {
			return fmt.Errorf("layout has no shift key")
		}
		shiftCode = sc
		if err := xc.fakeInput(evKeyPress, shiftCode, 0, 0, false); err != nil {
			return err
		}
	}
	if err := xc.fakeInput(evKeyPress, code, 0, 0, false); err != nil {
		return err
	}
	time.Sleep(8 * time.Millisecond)
	if err := xc.fakeInput(evKeyRelease, code, 0, 0, false); err != nil {
		return err
	}
	if shift {
		return xc.fakeInput(evKeyRelease, shiftCode, 0, 0, false)
	}
	return nil
}

func (d *linuxDriver) Key(chord string) error {
	mods, key, err := parseChord(chord)
	if err != nil {
		return err
	}
	xc, err := d.connect()
	if err != nil {
		return err
	}
	if err := xc.loadKeymap(); err != nil {
		return err
	}
	// Press modifiers first, main key next, unwind in reverse.
	var pressed []byte
	press := func(keysym uint32) error {
		code, _, ok := xc.keycodeFor(keysym)
		if !ok {
			return fmt.Errorf("layout has no key for keysym %#x", keysym)
		}
		if err := xc.fakeInput(evKeyPress, code, 0, 0, false); err != nil {
			return err
		}
		pressed = append(pressed, code)
		return nil
	}
	for _, m := range mods {
		ks, err := keysymForName(m)
		if err != nil {
			return err
		}
		if err := press(ks); err != nil {
			return err
		}
	}
	ks, err := keysymForName(key)
	if err != nil {
		return err
	}
	err = press(ks)
	for i := len(pressed) - 1; i >= 0; i-- {
		_ = xc.fakeInput(evKeyRelease, pressed[i], 0, 0, false)
	}
	return err
}

func (d *linuxDriver) Type(text string) (int, error) {
	n := 0
	for _, r := range text {
		if r < 0x20 && r != '\n' && r != '\t' && r != '\r' {
			return n, fmt.Errorf("type: control character %#U needs a key chord", r)
		}
		ks := keysymForRune(r)
		if ks == 0 {
			return n, fmt.Errorf("type: no keysym for %#U", r)
		}
		if err := d.pressKeysym(ks); err != nil {
			return n, err
		}
		n++
	}
	return n, nil
}

func (d *linuxDriver) Focus(window uint32) error {
	xc, err := d.connect()
	if err != nil {
		return err
	}
	// SetInputFocus (opcode 42): revert-to None, focus, CurrentTime.
	req := reqHead(42, 0, 3)
	req = append(req, u32(window)...)
	req = append(req, u32(0)...)
	if _, err := xc.do(req, false); err != nil {
		return err
	}
	// Also ask any window manager politely (_NET_ACTIVE_WINDOW).
	if atom, err := xc.internAtom("_NET_ACTIVE_WINDOW", true); err == nil && atom != 0 {
		ev := make([]byte, 32)
		ev[0] = 33 // ClientMessage
		ev[1] = 32
		binary.LittleEndian.PutUint32(ev[4:8], xc.root)
		binary.LittleEndian.PutUint32(ev[8:12], atom)
		binary.LittleEndian.PutUint32(ev[12:16], 1) // source: application
		req := reqHead(25, 0, 11)
		req = append(req, u32(xc.root)...)
		req = append(req, u32(0x80000|0x100000)...) // SubstructureRedirect|Notify
		req = append(req, ev...)
		_, _ = xc.do(req, false)
	}
	focused, err := xc.getInputFocus()
	if err != nil {
		return err
	}
	// The focus window may be the target or one of its descendants.
	cur := focused
	for depth := 0; depth < 32; depth++ {
		if cur == window {
			return nil
		}
		if cur == 0 || cur == xc.root {
			break
		}
		parent, _, err := xc.queryTree(cur)
		if err != nil || parent == 0 {
			break
		}
		cur = parent
	}
	return fmt.Errorf("focus did not land on window %d (focus is %d)", window, focused)
}

func (d *linuxDriver) Launch(argv []string, cwd string) (int, error) {
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Dir = cwd
	cmd.Stdout = nil
	cmd.Stderr = nil
	cmd.Stdin = nil
	if err := cmd.Start(); err != nil {
		return 0, fmt.Errorf("launch %q: %w", argv[0], err)
	}
	go func() { _ = cmd.Wait() }()
	return cmd.Process.Pid, nil
}

func (d *linuxDriver) Kill(pid int) error {
	proc, err := os.FindProcess(pid)
	if err != nil {
		return err
	}
	return proc.Kill()
}

func (d *linuxDriver) CloseWindow(window uint32) error {
	xc, err := d.connect()
	if err != nil {
		return err
	}
	wmProtocols, err := xc.internAtom("WM_PROTOCOLS", false)
	if err != nil {
		return err
	}
	wmDelete, err := xc.internAtom("WM_DELETE_WINDOW", false)
	if err != nil {
		return err
	}
	ev := make([]byte, 32)
	ev[0] = 33 // ClientMessage
	ev[1] = 32
	binary.LittleEndian.PutUint32(ev[4:8], window)
	binary.LittleEndian.PutUint32(ev[8:12], wmProtocols)
	binary.LittleEndian.PutUint32(ev[12:16], wmDelete)
	// data.l[1] = CurrentTime (0)
	req := reqHead(25, 0, 11)
	req = append(req, u32(window)...)
	req = append(req, u32(0)...) // event mask: NoEventMask, delivered to creator
	req = append(req, ev...)
	_, err = xc.do(req, false)
	return err
}

func (d *linuxDriver) Close() {
	if d.xc != nil && d.xc.conn != nil {
		_ = d.xc.conn.Close()
	}
}
