//go:build windows

package main

// Windows backend: raw Win32, no automation library.
//
// Screen capture is GDI BitBlt + GetDIBits, input is SendInput,
// window enumeration is EnumWindows — the same direct-OS posture
// as the Linux backend's hand-rolled X11 client. This bridge runs
// inside the owner's interactive session on the host (EC2), where
// the desktop actually lives.

import (
	"bytes"
	"fmt"
	"image"
	"image/png"
	"os"
	"os/exec"
	"strings"
	"syscall"
	"time"
	"unicode/utf16"
	"unsafe"
)

var (
	user32   = syscall.NewLazyDLL("user32.dll")
	gdi32    = syscall.NewLazyDLL("gdi32.dll")
	kernel32 = syscall.NewLazyDLL("kernel32.dll")

	procGetSystemMetrics       = user32.NewProc("GetSystemMetrics")
	procEnumWindows            = user32.NewProc("EnumWindows")
	procIsWindowVisible        = user32.NewProc("IsWindowVisible")
	procGetWindowTextW         = user32.NewProc("GetWindowTextW")
	procGetWindowTextLengthW   = user32.NewProc("GetWindowTextLengthW")
	procGetWindowRect          = user32.NewProc("GetWindowRect")
	procGetForegroundWindow    = user32.NewProc("GetForegroundWindow")
	procGetWindowThreadProcess = user32.NewProc("GetWindowThreadProcessId")
	procSetForegroundWindow    = user32.NewProc("SetForegroundWindow")
	procGetCursorPos           = user32.NewProc("GetCursorPos")
	procSetCursorPos           = user32.NewProc("SetCursorPos")
	procSendInput              = user32.NewProc("SendInput")
	procPostMessageW           = user32.NewProc("PostMessageW")
	procGetShellWindow         = user32.NewProc("GetShellWindow")
	procVkKeyScanW             = user32.NewProc("VkKeyScanW")
	procGetDC                  = user32.NewProc("GetDC")
	procReleaseDC              = user32.NewProc("ReleaseDC")

	procCreateCompatibleDC     = gdi32.NewProc("CreateCompatibleDC")
	procCreateCompatibleBitmap = gdi32.NewProc("CreateCompatibleBitmap")
	procSelectObject           = gdi32.NewProc("SelectObject")
	procBitBlt                 = gdi32.NewProc("BitBlt")
	procGetDIBits              = gdi32.NewProc("GetDIBits")
	procDeleteObject           = gdi32.NewProc("DeleteObject")
	procDeleteDC               = gdi32.NewProc("DeleteDC")
	procOpenProcess            = kernel32.NewProc("OpenProcess")
)

const (
	smCXScreen        = 0
	smCYScreen        = 1
	smXVirtualScreen  = 76
	smYVirtualScreen  = 77
	smCXVirtualScreen = 78
	smCYVirtualScreen = 79

	srcCopy = 0x00CC0020

	inputMouse    = 0
	inputKeyboard = 1

	mouseEventMove     = 0x0001
	mouseEventLeftDown = 0x0002
	mouseEventLeftUp   = 0x0004
	mouseEventRightDown = 0x0008
	mouseEventRightUp  = 0x0010
	mouseEventMiddleDown = 0x0020
	mouseEventMiddleUp = 0x0040
	mouseEventWheel    = 0x0800
	mouseEventHWheel   = 0x1000

	keyEventKeyUp   = 0x0002
	keyEventUnicode = 0x0004

	wheelDelta = 120
	wmClose    = 0x0010
)

type rect struct {
	Left, Top, Right, Bottom int32
}

type point struct {
	X, Y int32
}

func metric(i int) int {
	r, _, _ := procGetSystemMetrics.Call(uintptr(i))
	return int(r)
}

func virtualScreen() (x, y, w, h int) {
	w = metric(smCXVirtualScreen)
	h = metric(smCYVirtualScreen)
	if w <= 0 || h <= 0 {
		return 0, 0, metric(smCXScreen), metric(smCYScreen)
	}
	return metric(smXVirtualScreen), metric(smYVirtualScreen), w, h
}

type windowsDriver struct{}

func newDriver() Driver { return &windowsDriver{} }

func (d *windowsDriver) Probe() ProbeInfo {
	info := ProbeInfo{
		Protocol: 1, Platform: "windows", Backend: "win32 (own bridge)",
		Display: "console",
		Features: []string{"screenshot", "windows", "pointer", "mouse",
			"keyboard", "launch", "focus", "close_window"},
	}
	_, _, w, h := virtualScreen()
	if w <= 0 || h <= 0 {
		info.Available = false
		info.Reason = "GetSystemMetrics reports no display in this session"
		return info
	}
	info.Available = true
	info.Width = w
	info.Height = h
	return info
}

func (d *windowsDriver) Ping() error {
	if metric(smCXScreen) <= 0 {
		return fmt.Errorf("no display metrics in this session")
	}
	return nil
}

type bitmapInfoHeader struct {
	Size          uint32
	Width         int32
	Height        int32
	Planes        uint16
	BitCount      uint16
	Compression   uint32
	SizeImage     uint32
	XPelsPerMeter int32
	YPelsPerMeter int32
	ClrUsed       uint32
	ClrImportant  uint32
}

func (d *windowsDriver) Screenshot() ([]byte, int, int, error) {
	vx, vy, w, h := virtualScreen()
	if w <= 0 || h <= 0 {
		return nil, 0, 0, fmt.Errorf("no display in this session")
	}
	hdcScreen, _, _ := procGetDC.Call(0)
	if hdcScreen == 0 {
		return nil, 0, 0, fmt.Errorf("GetDC failed")
	}
	defer procReleaseDC.Call(0, hdcScreen)
	hdcMem, _, _ := procCreateCompatibleDC.Call(hdcScreen)
	if hdcMem == 0 {
		return nil, 0, 0, fmt.Errorf("CreateCompatibleDC failed")
	}
	defer procDeleteDC.Call(hdcMem)
	hbmp, _, _ := procCreateCompatibleBitmap.Call(hdcScreen, uintptr(w), uintptr(h))
	if hbmp == 0 {
		return nil, 0, 0, fmt.Errorf("CreateCompatibleBitmap failed")
	}
	defer procDeleteObject.Call(hbmp)
	old, _, _ := procSelectObject.Call(hdcMem, hbmp)
	defer procSelectObject.Call(hdcMem, old)
	ok, _, _ := procBitBlt.Call(hdcMem, 0, 0, uintptr(w), uintptr(h),
		hdcScreen, uintptr(vx), uintptr(vy), srcCopy)
	if ok == 0 {
		return nil, 0, 0, fmt.Errorf("BitBlt failed")
	}
	bmi := bitmapInfoHeader{
		Size: 40, Width: int32(w), Height: -int32(h), // top-down
		Planes: 1, BitCount: 32,
	}
	pixels := make([]byte, w*h*4)
	r, _, _ := procGetDIBits.Call(hdcMem, hbmp, 0, uintptr(h),
		uintptr(unsafe.Pointer(&pixels[0])),
		uintptr(unsafe.Pointer(&bmi)), 0)
	if r == 0 {
		return nil, 0, 0, fmt.Errorf("GetDIBits failed")
	}
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for i := 0; i < w*h; i++ {
		img.Pix[i*4+0] = pixels[i*4+2]
		img.Pix[i*4+1] = pixels[i*4+1]
		img.Pix[i*4+2] = pixels[i*4+0]
		img.Pix[i*4+3] = 255
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return nil, 0, 0, err
	}
	return buf.Bytes(), w, h, nil
}

var enumCollect []WindowInfo
var enumFocus uintptr

func (d *windowsDriver) Windows() ([]WindowInfo, uint32, error) {
	enumCollect = nil
	shell, _, _ := procGetShellWindow.Call()
	fg, _, _ := procGetForegroundWindow.Call()
	enumFocus = fg
	cb := syscall.NewCallback(func(hwnd uintptr, _ uintptr) uintptr {
		if hwnd == shell {
			return 1
		}
		vis, _, _ := procIsWindowVisible.Call(hwnd)
		if vis == 0 {
			return 1
		}
		var r rect
		ok, _, _ := procGetWindowRect.Call(hwnd, uintptr(unsafe.Pointer(&r)))
		if ok == 0 || r.Right-r.Left <= 0 || r.Bottom-r.Top <= 0 {
			return 1
		}
		title := ""
		n, _, _ := procGetWindowTextLengthW.Call(hwnd)
		if n > 0 {
			buf := make([]uint16, n+1)
			procGetWindowTextW.Call(hwnd, uintptr(unsafe.Pointer(&buf[0])), uintptr(n+1))
			title = syscall.UTF16ToString(buf)
		}
		var pid uint32
		procGetWindowThreadProcess.Call(hwnd, uintptr(unsafe.Pointer(&pid)))
		enumCollect = append(enumCollect, WindowInfo{
			ID: uint32(hwnd), Title: title,
			X: int(r.Left), Y: int(r.Top),
			Width: int(r.Right - r.Left), Height: int(r.Bottom - r.Top),
			Focused: hwnd == enumFocus, PID: int(pid),
		})
		return 1
	})
	procEnumWindows.Call(cb, 0)
	return enumCollect, uint32(fg), nil
}

func (d *windowsDriver) Pointer() (int, int, error) {
	var p point
	ok, _, err := procGetCursorPos.Call(uintptr(unsafe.Pointer(&p)))
	if ok == 0 {
		return 0, 0, fmt.Errorf("GetCursorPos: %v", err)
	}
	return int(p.X), int(p.Y), nil
}

func (d *windowsDriver) MouseMove(x, y int) error {
	ok, _, err := procSetCursorPos.Call(uintptr(x), uintptr(y))
	if ok == 0 {
		return fmt.Errorf("SetCursorPos: %v", err)
	}
	return nil
}

// inputRecord builds one 40-byte INPUT struct (amd64 layout):
// type u32, pad u32, then a 32-byte union. Keyboard data lives at
// union+0, mouse data at union+0.
func inputRecord(typ uint32, fill func(u []byte)) []byte {
	b := make([]byte, 40)
	b[0] = byte(typ)
	fill(b[8:])
	return b
}

func sendInputs(records [][]byte) error {
	if len(records) == 0 {
		return nil
	}
	flat := make([]byte, 0, len(records)*40)
	for _, r := range records {
		flat = append(flat, r...)
	}
	n, _, err := procSendInput.Call(uintptr(len(records)),
		uintptr(unsafe.Pointer(&flat[0])), uintptr(40))
	if int(n) != len(records) {
		return fmt.Errorf("SendInput delivered %d of %d inputs: %v", n, len(records), err)
	}
	return nil
}

func mouseRecord(flags uint32, data uint32) []byte {
	return inputRecord(inputMouse, func(u []byte) {
		// MOUSEINPUT: dx, dy int32; mouseData, dwFlags, time u32; extra uintptr
		u[8] = byte(data)
		u[9] = byte(data >> 8)
		u[10] = byte(data >> 16)
		u[11] = byte(data >> 24)
		u[12] = byte(flags)
		u[13] = byte(flags >> 8)
		u[14] = byte(flags >> 16)
		u[15] = byte(flags >> 24)
	})
}

func buttonFlags(button string) (down, up uint32, err error) {
	switch strings.ToLower(button) {
	case "left", "":
		return mouseEventLeftDown, mouseEventLeftUp, nil
	case "middle":
		return mouseEventMiddleDown, mouseEventMiddleUp, nil
	case "right":
		return mouseEventRightDown, mouseEventRightUp, nil
	}
	return 0, 0, fmt.Errorf("unknown button %q", button)
}

func (d *windowsDriver) MouseClick(x, y int, button string, clicks int) error {
	down, up, err := buttonFlags(button)
	if err != nil {
		return err
	}
	if err := d.MouseMove(x, y); err != nil {
		return err
	}
	for i := 0; i < clicks; i++ {
		if err := sendInputs([][]byte{mouseRecord(down, 0)}); err != nil {
			return err
		}
		time.Sleep(15 * time.Millisecond)
		if err := sendInputs([][]byte{mouseRecord(up, 0)}); err != nil {
			return err
		}
		time.Sleep(15 * time.Millisecond)
	}
	return nil
}

func (d *windowsDriver) MouseDrag(x1, y1, x2, y2 int, button string) error {
	down, up, err := buttonFlags(button)
	if err != nil {
		return err
	}
	if err := d.MouseMove(x1, y1); err != nil {
		return err
	}
	if err := sendInputs([][]byte{mouseRecord(down, 0)}); err != nil {
		return err
	}
	steps := 12
	for i := 1; i <= steps; i++ {
		_ = d.MouseMove(x1+(x2-x1)*i/steps, y1+(y2-y1)*i/steps)
		time.Sleep(10 * time.Millisecond)
	}
	return sendInputs([][]byte{mouseRecord(up, 0)})
}

func (d *windowsDriver) MouseScroll(x, y, dx, dy int) error {
	if err := d.MouseMove(x, y); err != nil {
		return err
	}
	scroll := func(flags uint32, amount int) error {
		if amount == 0 {
			return nil
		}
		data := uint32(int32(amount * wheelDelta))
		return sendInputs([][]byte{mouseRecord(flags, data)})
	}
	if err := scroll(mouseEventWheel, dy); err != nil {
		return err
	}
	return scroll(mouseEventHWheel, dx)
}

func keyRecord(vk uint16, scan uint16, flags uint32) []byte {
	return inputRecord(inputKeyboard, func(u []byte) {
		// KEYBDINPUT: wVk u16, wScan u16, dwFlags u32, time u32, extra uintptr
		u[0] = byte(vk)
		u[1] = byte(vk >> 8)
		u[2] = byte(scan)
		u[3] = byte(scan >> 8)
		u[4] = byte(flags)
		u[5] = byte(flags >> 8)
		u[6] = byte(flags >> 16)
		u[7] = byte(flags >> 24)
	})
}

var namedVK = map[string]uint16{
	"return": 0x0D, "enter": 0x0D, "escape": 0x1B, "esc": 0x1B,
	"tab": 0x09, "backspace": 0x08, "delete": 0x2E, "del": 0x2E,
	"insert": 0x2D, "home": 0x24, "end": 0x23,
	"pageup": 0x21, "page_up": 0x21, "prior": 0x21,
	"pagedown": 0x22, "page_down": 0x22, "next": 0x22,
	"left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
	"space": 0x20, "shift": 0x10, "shift_l": 0xA0, "shift_r": 0xA1,
	"ctrl": 0x11, "control": 0x11, "control_l": 0xA2, "control_r": 0xA3,
	"alt": 0x12, "alt_l": 0xA4, "alt_r": 0xA5,
	"win": 0x5B, "super": 0x5B, "meta": 0x5B,
	"capslock": 0x14, "numlock": 0x90,
}

func vkForName(name string) (uint16, bool, error) {
	lower := strings.ToLower(name)
	if vk, ok := namedVK[lower]; ok {
		return vk, false, nil
	}
	if strings.HasPrefix(lower, "f") && len(lower) <= 3 {
		var n int
		if _, err := fmt.Sscanf(lower, "f%d", &n); err == nil && n >= 1 && n <= 12 {
			return uint16(0x70 + n - 1), false, nil
		}
	}
	r := []rune(name)
	if len(r) == 1 {
		res, _, _ := procVkKeyScanW.Call(uintptr(r[0]))
		vk := uint16(res & 0xFF)
		if vk == 0xFF {
			return 0, false, fmt.Errorf("no virtual key for %q on this layout", name)
		}
		shift := (res>>8)&1 == 1
		return vk, shift, nil
	}
	return 0, false, fmt.Errorf("unknown key name %q", name)
}

func modifierVK(mod string) (uint16, error) {
	switch strings.ToLower(mod) {
	case "ctrl", "control":
		return 0x11, nil
	case "alt":
		return 0x12, nil
	case "shift":
		return 0x10, nil
	case "win", "super", "meta":
		return 0x5B, nil
	}
	return 0, fmt.Errorf("unknown modifier %q", mod)
}

func (d *windowsDriver) Key(chord string) error {
	mods, key, err := parseChord(chord)
	if err != nil {
		return err
	}
	var down [][]byte
	var up [][]byte
	for _, m := range mods {
		vk, err := modifierVK(m)
		if err != nil {
			return err
		}
		down = append(down, keyRecord(vk, 0, 0))
		up = append([][]byte{keyRecord(vk, 0, keyEventKeyUp)}, up...)
	}
	vk, needShift, err := vkForName(key)
	if err != nil {
		return err
	}
	if needShift {
		down = append(down, keyRecord(0x10, 0, 0))
		up = append([][]byte{keyRecord(0x10, 0, keyEventKeyUp)}, up...)
	}
	down = append(down, keyRecord(vk, 0, 0))
	up = append([][]byte{keyRecord(vk, 0, keyEventKeyUp)}, up...)
	if err := sendInputs(down); err != nil {
		return err
	}
	time.Sleep(10 * time.Millisecond)
	return sendInputs(up)
}

func (d *windowsDriver) Type(text string) (int, error) {
	n := 0
	for _, r := range text {
		units := utf16.Encode([]rune{r})
		for _, u := range units {
			if err := sendInputs([][]byte{keyRecord(0, u, keyEventUnicode)}); err != nil {
				return n, err
			}
			if err := sendInputs([][]byte{keyRecord(0, u, keyEventUnicode|keyEventKeyUp)}); err != nil {
				return n, err
			}
		}
		n++
	}
	return n, nil
}

func (d *windowsDriver) Focus(window uint32) error {
	hwnd := uintptr(window)
	ok, _, err := procSetForegroundWindow.Call(hwnd)
	if ok == 0 {
		return fmt.Errorf("SetForegroundWindow: %v", err)
	}
	return nil
}

func (d *windowsDriver) Launch(argv []string, cwd string) (int, error) {
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Dir = cwd
	if err := cmd.Start(); err != nil {
		return 0, fmt.Errorf("launch %q: %w", argv[0], err)
	}
	go func() { _ = cmd.Wait() }()
	return cmd.Process.Pid, nil
}

func (d *windowsDriver) Kill(pid int) error {
	proc, err := os.FindProcess(pid)
	if err != nil {
		return err
	}
	return proc.Kill()
}

func (d *windowsDriver) CloseWindow(window uint32) error {
	ok, _, err := procPostMessageW.Call(uintptr(window), wmClose, 0, 0)
	if ok == 0 {
		return fmt.Errorf("PostMessageW(WM_CLOSE): %v", err)
	}
	return nil
}

func (d *windowsDriver) Close() {}
