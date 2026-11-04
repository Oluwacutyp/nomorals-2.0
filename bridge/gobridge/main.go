// devon-whatsapp-bridge — WhatsApp Web session holder for Devon's
// WhatsAppAdapter, written in Go on whatsmeow (go.mau.fi/whatsmeow).
//
// This binary REPLACES the old Node/Baileys bridge
// (bridge/whatsapp-bridge.mjs et al.) wholesale: one self-contained
// executable, no Node, no npm, no node_modules. It speaks the exact
// same contracts as the bridge it replaces:
//
//  1. The supervisor contract (nomorals/social/pairing.py): argv
//     flags --port/--creds/--media, env WA_PAIRING_CODE /
//     WA_PAIRING_NUMBER for pairing launches, and stdout log lines
//     the supervisor's drain scans —
//     PAIRING CODE for +<digits>: <code>
//     PAIRING REFUSED: <reason>
//     PAIRING LINK FAILED: <reason>
//     connected as <jid>
//     A pairing code is printed ONLY after whatsmeow's PairPhone
//     returns one without error — PairPhone awaits the server's
//     answer, so a code on this line is a code WhatsApp actually
//     registered. Phantom codes are impossible by construction.
//
//  2. The adapter contract (nomorals/social/chat/whatsapp.py):
//     a JSON-lines protocol over TCP on 127.0.0.1:<port>, the same
//     event and command shapes the Baileys bridge served.
//
// Session state lives in a whatsmeow sqlstore (pure-Go modernc
// sqlite — CGO_ENABLED=0 builds) inside the creds dir the
// supervisor passes. Old Baileys multi-file creds in the same dir
// are ignored; they are a different format and do not migrate.
//
// Reconnects: whatsmeow's own auto-reconnect owns the socket; this
// bridge owns the OBSERVABLE policy — permanent failures (logged
// out, stream replaced, client outdated, temp-banned, connect
// failures) park the bridge and say so in the status payload,
// exactly as the old bridge's policy table did. The status payload
// keeps the old rich shape; the Python side reads state/user and
// ignores the rest.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"net"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"go.mau.fi/whatsmeow"
	waE2E "go.mau.fi/whatsmeow/proto/waE2E"
	"go.mau.fi/whatsmeow/store/sqlstore"
	"go.mau.fi/whatsmeow/types"
	"go.mau.fi/whatsmeow/types/events"
	waLog "go.mau.fi/whatsmeow/util/log"

	_ "modernc.org/sqlite"
)

// Build identity, surfaced by --version (the supervisor's probe).
const (
	bridgeVersion  = "gobridge/1.0.0"
	whatsmeowPin   = "v0.0.0-20261102152715-54038813c47f"
	reconnectFile  = "reconnect-state.json"
	reconnectReset = 15 * time.Minute
	sessionDBName  = "whatsmeow.db"
)

// logf writes one supervisor-visible log line to stdout in the
// shape the drain has always cleaned: "<ISO> [bridge] <msg>".
func logf(format string, args ...any) {
	msg := fmt.Sprintf(format, args...)
	fmt.Printf("%s [bridge] %s\n", time.Now().UTC().Format("2006-01-02T15:04:05.000Z"), msg)
}

type bridge struct {
	cli       *whatsmeow.Client
	container *sqlstore.Container

	port          int
	credsDir      string
	mediaDir      string
	pairingNumber string // digits only; "" = no pairing launch
	pairingMode   bool   // WA_PAIRING_CODE was set

	// TCP clients
	clientsMu sync.Mutex
	clients   map[*clientConn]struct{}

	// observable connection state (the old bridge's status payload)
	stateMu             sync.Mutex
	connState           string // closed | open | reconnecting | parked
	meJID               string
	meLID               string // the account's LID (<lid>@lid), "" until known
	lastQR              string // latest login QR — replayed to new clients
	lastDisconnectCode  *int
	lastDisconnectLabel string
	terminalReason      string

	// pairing (guarded by pairMu)
	pairMu          sync.Mutex
	pairRequested   bool // PairPhone already called for this process
	pairOutstanding bool // code issued, link not yet confirmed

	// message memory (mirrors the .mjs ring buffers)
	memMu     sync.Mutex
	recent    map[string][]*remembered
	titles    map[string]string
	msgCache  map[string]*cachedMsg
	cacheKeys []string
}

type clientConn struct {
	conn net.Conn
	mu   sync.Mutex // serializes writes to this socket
}

func (c *clientConn) writeLine(line []byte) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	_, err := c.conn.Write(line)
	return err == nil
}

type remembered struct {
	incoming bool
	text     string
	sender   string // display name (history rows)
	media    []map[string]any
	replyTo  string
	tsMs     int64
	// raw material for the `read` command (MarkRead needs the key parts)
	id        types.MessageID
	chatJID   types.JID
	senderJID types.JID
}

type cachedMsg struct {
	msg    *waE2E.Message
	sender types.JID
}

func main() {
	if len(os.Args) > 1 && os.Args[1] == "--version" {
		fmt.Printf("devon-whatsapp-bridge %s (whatsmeow %s)\n", bridgeVersion, whatsmeowPin)
		return
	}
	var (
		port    = flag.Int("port", 0, "TCP port to serve the adapter protocol on")
		creds   = flag.String("creds", "", "session creds directory")
		media   = flag.String("media", "", "inbound media directory")
		showVer = flag.Bool("version", false, "print version and exit")
	)
	flag.Parse()
	if *showVer {
		fmt.Printf("devon-whatsapp-bridge %s (whatsmeow %s)\n", bridgeVersion, whatsmeowPin)
		return
	}
	if *port == 0 {
		if v := os.Getenv("WA_BRIDGE_PORT"); v != "" {
			fmt.Sscanf(v, "%d", port)
		}
		if *port == 0 {
			*port = 8787
		}
	}
	if *creds == "" {
		*creds = os.Getenv("WA_CREDS_DIR")
	}
	if *creds == "" {
		*creds = "./.creds"
	}
	if *media == "" {
		*media = os.Getenv("WA_MEDIA_DIR")
	}
	if *media == "" {
		*media = "./media"
	}
	credsAbs, err := filepath.Abs(*creds)
	if err != nil {
		credsAbs = *creds
	}
	mediaAbs, err := filepath.Abs(*media)
	if err != nil {
		mediaAbs = *media
	}
	if err := os.MkdirAll(mediaAbs, 0o755); err != nil {
		logf("cannot create media dir %s: %v", mediaAbs, err)
		os.Exit(1)
	}
	if err := os.MkdirAll(credsAbs, 0o755); err != nil {
		logf("cannot create creds dir %s: %v", credsAbs, err)
		os.Exit(1)
	}

	pairingNumber := digitsOnly(os.Getenv("WA_PAIRING_NUMBER"))
	pairingMode := os.Getenv("WA_PAIRING_CODE") == "1" || os.Getenv("WA_PAIRING_CODE") == "true"

	ctx := context.Background()
	dbPath := filepath.ToSlash(filepath.Join(credsAbs, sessionDBName))
	container, err := sqlstore.New(ctx, "sqlite",
		"file:"+dbPath+"?_pragma=foreign_keys(1)", nil)
	if err != nil {
		logf("failed to open session store: %v", err)
		os.Exit(1)
	}
	device, err := container.GetFirstDevice(ctx)
	if err != nil {
		logf("failed to load session device: %v", err)
		os.Exit(1)
	}

	b := &bridge{
		container:     container,
		port:          *port,
		credsDir:      credsAbs,
		mediaDir:      mediaAbs,
		pairingNumber: pairingNumber,
		pairingMode:   pairingMode,
		clients:       make(map[*clientConn]struct{}),
		connState:     "closed",
		recent:        make(map[string][]*remembered),
		titles:        make(map[string]string),
		msgCache:      make(map[string]*cachedMsg),
	}

	// whatsmeow's own logger stays OFF stdout: the supervisor drain
	// owns that stream, and library chatter would pollute the death
	// lines it reads. This bridge logs every decision itself.
	cli := whatsmeow.NewClient(device, waLog.Noop)
	cli.QRClientType = whatsmeow.PairClientChrome
	b.cli = cli
	cli.AddEventHandler(b.handleEvent)

	ln, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", b.port))
	if err != nil {
		logf("cannot listen on 127.0.0.1:%d: %v", b.port, err)
		os.Exit(1)
	}
	logf("listening on 127.0.0.1:%d; creds=%s; media=%s", b.port, credsAbs, mediaAbs)
	go b.acceptLoop(ln)

	parked := b.loadReconnectState()
	if parked {
		logf("PARKED: %s (owner action required)", b.terminalReason)
		b.setConnState("parked")
		b.broadcastStatus()
	} else {
		go b.connectLoop()
	}

	// Run until killed (the supervisor owns our lifecycle).
	select {}
}

// ── connection state / status payload ─────────────────────────────

func (b *bridge) setConnState(state string) {
	b.stateMu.Lock()
	b.connState = state
	b.stateMu.Unlock()
}

// myLID returns the account's own LID in non-AD form
// ("<lid>@lid"), or "" when the account has none (not migrated)
// or it isn't known yet. Accounts migrated to LID addressing
// deliver the "Message yourself" chat under this JID, not the
// phone JID in "user" — the adapter needs both identities to
// recognise the self-chat. Read live from the device store so a
// LID learned after connect (migration) is picked up on the next
// status broadcast; meLID is the cached copy from connect time.
func (b *bridge) myLID() string {
	if b.cli != nil && b.cli.Store != nil {
		if lid := b.cli.Store.GetLID(); !lid.IsEmpty() {
			return lid.ToNonAD().String()
		}
	}
	// Unlocked read, like meJID's uses: statusPayload calls this
	// while holding stateMu, and Go mutexes are not reentrant.
	return b.meLID
}

func (b *bridge) statusPayload() map[string]any {
	b.stateMu.Lock()
	defer b.stateMu.Unlock()
	var code any
	if b.lastDisconnectCode != nil {
		code = *b.lastDisconnectCode
	}
	return map[string]any{
		"type":                  "status",
		"state":                 b.connState,
		"user":                  b.meJID,
		"lid":                   b.myLID(),
		"reconnect_attempt":     b.cli.AutoReconnectErrors,
		"next_retry_in_s":       0,
		"last_disconnect_code":  code,
		"last_disconnect_label": nilOr(b.lastDisconnectLabel),
		"terminal_reason":       nilOr(b.terminalReason),
	}
}

func nilOr(s string) any {
	if s == "" {
		return nil
	}
	return s
}

func (b *bridge) broadcastStatus() {
	b.broadcast(b.statusPayload())
}

func (b *bridge) park(reason, label string, code *int) {
	// A pairing that was outstanding dies with the park — the
	// supervisor hears it in the shape it maps to last_error.
	b.pairMu.Lock()
	outstanding := b.pairOutstanding
	b.pairOutstanding = false
	b.pairMu.Unlock()
	if outstanding {
		logf("PAIRING LINK FAILED: %s", reason)
	}
	b.stateMu.Lock()
	b.terminalReason = reason
	b.lastDisconnectLabel = label
	b.lastDisconnectCode = code
	b.connState = "parked"
	b.stateMu.Unlock()
	b.saveReconnectState()
	logf("PARKED: %s (owner action required)", reason)
	b.broadcastStatus()
}

func (b *bridge) isParked() bool {
	b.stateMu.Lock()
	defer b.stateMu.Unlock()
	return b.terminalReason != ""
}

// ── initial connect loop ──────────────────────────────────────────
// whatsmeow background-retries only the connect errors it
// classifies as retryable; a hard dial failure (e.g. a TLS
// timeout its classifier misses) would otherwise leave this
// process serving TCP with a dead session forever. The loop is
// the old bridge's safeConnect: jittered exponential backoff
// (5s → 120s cap) under a 15-attempt budget, then park. Once a
// Connect succeeds, whatsmeow's auto-reconnect owns later drops
// (and onDisconnected applies the same budget to its counter).
const (
	reconnectBaseS    = 5.0
	reconnectCapS     = 120.0
	reconnectMaxTries = 15
)

func (b *bridge) connectLoop() {
	attempt := 0
	for {
		if b.isParked() || b.cli.IsConnected() {
			return
		}
		err := b.cli.Connect()
		if err == nil || errors.Is(err, whatsmeow.ErrAlreadyConnected) {
			// Connected, or whatsmeow's background auto-reconnect
			// owns the retry from here.
			return
		}
		attempt++
		logf("connect failed: %v", err)
		if attempt >= reconnectMaxTries {
			b.park("retry-budget-exhausted — stopped after 15 attempts: inspect the session dir and network, then restart the bridge",
				"retry-budget-exhausted", nil)
			return
		}
		exp := reconnectBaseS * math.Pow(2, float64(attempt-1))
		if exp > reconnectCapS {
			exp = reconnectCapS
		}
		delay := exp * (0.7 + rand.Float64()*0.6) // ±~30% jitter
		b.stateMu.Lock()
		b.lastDisconnectLabel = "connect-failed"
		b.stateMu.Unlock()
		b.setConnState("reconnecting")
		b.broadcastStatus()
		logf("reconnecting in %.0fs (attempt %d/%d)", delay, attempt, reconnectMaxTries)
		time.Sleep(time.Duration(delay) * time.Second)
	}
}

// ── reconnect state persistence (same file/keys as the old bridge) ─

type reconnectState struct {
	Attempt             int    `json:"attempt"`
	LastDisconnectCode  *int   `json:"lastDisconnectCode"`
	LastDisconnectLabel string `json:"lastDisconnectLabel"`
	LastDisconnectAt    int64  `json:"lastDisconnectAt"`
	TerminalReason      string `json:"terminalReason"`
	SavedAt             int64  `json:"savedAt"`
}

func (b *bridge) saveReconnectState() {
	b.stateMu.Lock()
	st := reconnectState{
		Attempt:             b.cli.AutoReconnectErrors,
		LastDisconnectCode:  b.lastDisconnectCode,
		LastDisconnectLabel: b.lastDisconnectLabel,
		TerminalReason:      b.terminalReason,
		SavedAt:             time.Now().UnixMilli(),
	}
	if b.lastDisconnectCode != nil {
		st.LastDisconnectAt = st.SavedAt
	}
	b.stateMu.Unlock()
	data, err := json.MarshalIndent(st, "", "  ")
	if err != nil {
		return
	}
	if err := os.WriteFile(filepath.Join(b.credsDir, reconnectFile), data, 0o600); err != nil {
		logf("reconnect-state save failed: %v", err)
	}
}

// loadReconnectState restores a recent outage's context. Returns
// true when a terminal (parked) state was restored — the caller
// then does not connect, exactly as the old bridge behaved.
func (b *bridge) loadReconnectState() bool {
	raw, err := os.ReadFile(filepath.Join(b.credsDir, reconnectFile))
	if err != nil {
		return false
	}
	var st reconnectState
	if err := json.Unmarshal(raw, &st); err != nil {
		return false
	}
	age := time.Since(time.UnixMilli(st.LastDisconnectAt))
	if st.LastDisconnectAt > 0 && age > reconnectReset {
		logf("reconnect-state: last outage is old, starting a fresh attempt ladder")
		return false
	}
	b.stateMu.Lock()
	b.lastDisconnectCode = st.LastDisconnectCode
	b.lastDisconnectLabel = st.LastDisconnectLabel
	b.terminalReason = st.TerminalReason
	parked := st.TerminalReason != ""
	b.stateMu.Unlock()
	if parked || st.Attempt > 0 {
		logf("reconnect-state restored: attempt=%d last=%s terminal=%s",
			st.Attempt, st.LastDisconnectLabel, st.TerminalReason)
	}
	return parked
}

// ── whatsmeow event handling: connection lifecycle + pairing ──────

func (b *bridge) handleEvent(evt any) {
	switch v := evt.(type) {
	case *events.Connected:
		b.onConnected()
	case *events.Disconnected:
		b.onDisconnected()
	case *events.LoggedOut:
		code := int(v.Reason)
		b.onPermanent(fmt.Sprintf("logged-out — re-auth required (%s)", v.Reason), "logged-out", &code)
	case *events.StreamReplaced:
		code := 440
		b.onPermanent("connection-replaced", "connection-replaced", &code)
	case *events.ClientOutdated:
		code := 405
		b.onPermanent("client-outdated", "client-outdated", &code)
	case *events.TemporaryBan:
		code := 402
		b.onPermanent(fmt.Sprintf("temporarily-banned (%s)", v.String()), "temporarily-banned", &code)
	case *events.CATRefreshError:
		b.onPermanent("cat-refresh-failed", "cat-refresh-failed", nil)
	case *events.ConnectFailure:
		code := int(v.Reason)
		b.onPermanent(fmt.Sprintf("connect-failure (%s)", v.Reason), "connect-failure", &code)
	case *events.PairSuccess:
		b.onPairSuccess(v)
	case *events.PairError:
		b.onPairError(v)
	case *events.QR:
		b.onQR(v)
	case *events.Message:
		b.onMessage(v)
	case *events.Receipt:
		b.onReceipt(v)
	}
}

func (b *bridge) onConnected() {
	me := ""
	lid := ""
	if b.cli.Store.ID != nil {
		me = b.cli.Store.ID.ToNonAD().String()
	}
	if l := b.cli.Store.GetLID(); !l.IsEmpty() {
		lid = l.ToNonAD().String()
	}
	b.stateMu.Lock()
	b.meJID = me
	b.meLID = lid
	b.connState = "open"
	b.lastQR = ""
	wasParked := b.terminalReason
	b.terminalReason = ""
	b.lastDisconnectCode = nil
	b.lastDisconnectLabel = ""
	b.stateMu.Unlock()
	b.pairMu.Lock()
	if me != "" {
		b.pairOutstanding = false
	}
	b.pairMu.Unlock()
	if wasParked != "" {
		logf("connected after park — clearing terminal state: %s", wasParked)
	}
	b.saveReconnectState()
	logf("connected as %s", me)
	b.broadcastStatus()
}

func (b *bridge) onDisconnected() {
	b.stateMu.Lock()
	if b.connState == "parked" {
		b.stateMu.Unlock()
		return
	}
	b.connState = "closed"
	b.stateMu.Unlock()
	b.broadcastStatus()
	// The old policy's retry budget, applied to whatsmeow's own
	// auto-reconnect counter: endless blind retries help nobody.
	if b.cli.AutoReconnectErrors >= reconnectMaxTries {
		b.park("retry-budget-exhausted — stopped after 15 attempts: inspect the session dir and network, then restart the bridge",
			"retry-budget-exhausted", nil)
		return
	}
	// whatsmeow's auto-reconnect owns the retry; the observable
	// state follows it into "reconnecting" while a session exists.
	if b.cli.Store.ID != nil {
		b.setConnState("reconnecting")
		b.broadcastStatus()
	}
}

// onPermanent handles the failures whatsmeow classifies as
// permanent: no blind retry — park and notify, as the old policy
// table did. park() itself reports a dead outstanding pairing.
func (b *bridge) onPermanent(reason, label string, code *int) {
	b.stateMu.Lock()
	b.connState = "closed"
	b.stateMu.Unlock()
	b.broadcastStatus()
	b.park(reason, label, code)
}

func (b *bridge) onPairSuccess(v *events.PairSuccess) {
	b.pairMu.Lock()
	b.pairOutstanding = false
	b.pairMu.Unlock()
	logf("pairing succeeded — linked as %s", v.ID.ToNonAD().String())
}

func (b *bridge) onPairError(v *events.PairError) {
	b.pairMu.Lock()
	outstanding := b.pairOutstanding
	b.pairOutstanding = false
	b.pairMu.Unlock()
	if outstanding {
		logf("PAIRING LINK FAILED: %v", v.Error)
	} else {
		logf("pairing failed: %v", v.Error)
	}
}

// onQR is the pairing trigger: whatsmeow emits *events.QR when the
// login websocket is established and a code is available — the
// documented precondition for PairPhone. In code mode the FIRST
// event requests the code (once per process; a fresh /pair starts
// a fresh process). In QR mode the latest code rides to the
// adapter as the {"type":"qr"} event, as before.
func (b *bridge) onQR(v *events.QR) {
	if len(v.Codes) == 0 || b.cli.Store.ID != nil {
		return
	}
	latest := v.Codes[len(v.Codes)-1]
	if b.pairingMode && b.pairingNumber != "" {
		b.requestPairCodeOnce()
		return
	}
	b.stateMu.Lock()
	b.lastQR = latest
	b.stateMu.Unlock()
	logf("QR received — scan with WhatsApp (Linked Devices).")
	b.broadcast(map[string]any{"type": "qr", "data": latest})
}

// requestPairCodeOnce asks WhatsApp for the pairing code exactly
// once. PairPhone awaits the server's IQ answer: an error here is
// the server's refusal, reported verbatim — and NO code is printed,
// so Devon can never display a code WhatsApp never registered.
func (b *bridge) requestPairCodeOnce() {
	b.pairMu.Lock()
	if b.pairRequested {
		b.pairMu.Unlock()
		return
	}
	b.pairRequested = true
	b.pairMu.Unlock()
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()
		code, err := b.cli.PairPhone(ctx, b.pairingNumber, true,
			whatsmeow.PairClientChrome, "Chrome (Windows)")
		if err != nil {
			logf("PAIRING REFUSED: %s", err.Error())
			return
		}
		b.pairMu.Lock()
		b.pairOutstanding = true
		b.pairMu.Unlock()
		logf("PAIRING CODE for +%s: %s (type into WhatsApp > Linked Devices)", b.pairingNumber, code)
		b.broadcast(map[string]any{"type": "pairing_code", "data": code})
	}()
}

// ── TCP plumbing ──────────────────────────────────────────────────

func (b *bridge) acceptLoop(ln net.Listener) {
	for {
		conn, err := ln.Accept()
		if err != nil {
			logf("accept failed: %v", err)
			return
		}
		cc := &clientConn{conn: conn}
		b.clientsMu.Lock()
		b.clients[cc] = struct{}{}
		b.clientsMu.Unlock()
		// A fresh client learns where we stand immediately —
		// including the latest QR while logged out (the old
		// bridge replayed its cached code the same way).
		if line, err := json.Marshal(b.statusPayload()); err == nil {
			cc.writeLine(append(line, '\n'))
		}
		b.stateMu.Lock()
		qr := b.lastQR
		b.stateMu.Unlock()
		if qr != "" && b.cli.Store.ID == nil {
			if line, err := json.Marshal(map[string]any{
				"type": "qr", "data": qr,
			}); err == nil {
				cc.writeLine(append(line, '\n'))
			}
		}
		go b.serveClient(cc)
	}
}

func (b *bridge) broadcast(obj map[string]any) {
	line, err := json.Marshal(obj)
	if err != nil {
		return
	}
	line = append(line, '\n')
	b.clientsMu.Lock()
	clients := make([]*clientConn, 0, len(b.clients))
	for cc := range b.clients {
		clients = append(clients, cc)
	}
	b.clientsMu.Unlock()
	for _, cc := range clients {
		if !cc.writeLine(line) {
			b.dropClient(cc)
		}
	}
}

func (b *bridge) dropClient(cc *clientConn) {
	b.clientsMu.Lock()
	delete(b.clients, cc)
	b.clientsMu.Unlock()
	cc.conn.Close()
}

func (b *bridge) reply(cc *clientConn, id string, obj map[string]any) {
	out := make(map[string]any, len(obj)+1)
	for k, v := range obj {
		out[k] = v
	}
	out["id"] = id
	line, err := json.Marshal(out)
	if err != nil {
		return
	}
	cc.writeLine(append(line, '\n'))
}

// ── small shared helpers ──────────────────────────────────────────

func digitsOnly(s string) string {
	var sb strings.Builder
	for _, r := range s {
		if r >= '0' && r <= '9' {
			sb.WriteRune(r)
		}
	}
	return sb.String()
}

func kindOf(jid types.JID) string {
	if jid.Server == types.GroupServer {
		return "group"
	}
	return "dm"
}

// parseJID normalizes a wire JID the way Baileys' jidNormalizedUser
// did: device parts stripped, empty input rejected by callers.
func parseJID(raw string) (types.JID, error) {
	jid, err := types.ParseJID(strings.TrimSpace(raw))
	if err != nil {
		return types.JID{}, err
	}
	return jid.ToNonAD(), nil
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}
