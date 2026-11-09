//go:build !linux && !windows

package main

import (
	"fmt"
	"runtime"
)

// Fallback backend for hosts with no desktop driver yet. Probe
// reports the honest truth; every action errors. The Python organ
// gates on probe availability, so this backend is never silently
// driven.
type noneDriver struct{}

func newDriver() Driver { return &noneDriver{} }

func (d *noneDriver) Probe() ProbeInfo {
	return ProbeInfo{
		Protocol: 1, Platform: runtime.GOOS, Backend: "none",
		Available: false,
		Reason:    fmt.Sprintf("no desktop backend is implemented for %s", runtime.GOOS),
		Features:  []string{},
	}
}

func (d *noneDriver) unavailable() error {
	return fmt.Errorf("desktop backend unavailable on %s", runtime.GOOS)
}

func (d *noneDriver) Ping() error                          { return d.unavailable() }
func (d *noneDriver) Screenshot() ([]byte, int, int, error) { return nil, 0, 0, d.unavailable() }
func (d *noneDriver) Windows() ([]WindowInfo, uint32, error) {
	return nil, 0, d.unavailable()
}
func (d *noneDriver) Pointer() (int, int, error)          { return 0, 0, d.unavailable() }
func (d *noneDriver) MouseMove(x, y int) error            { return d.unavailable() }
func (d *noneDriver) MouseClick(x, y int, b string, c int) error {
	return d.unavailable()
}
func (d *noneDriver) MouseDrag(x1, y1, x2, y2 int, b string) error {
	return d.unavailable()
}
func (d *noneDriver) MouseScroll(x, y, dx, dy int) error { return d.unavailable() }
func (d *noneDriver) Key(chord string) error             { return d.unavailable() }
func (d *noneDriver) Type(text string) (int, error)      { return 0, d.unavailable() }
func (d *noneDriver) Focus(window uint32) error          { return d.unavailable() }
func (d *noneDriver) Launch(argv []string, cwd string) (int, error) {
	return 0, d.unavailable()
}
func (d *noneDriver) Kill(pid int) error            { return d.unavailable() }
func (d *noneDriver) CloseWindow(window uint32) error { return d.unavailable() }
func (d *noneDriver) Close()                        {}
