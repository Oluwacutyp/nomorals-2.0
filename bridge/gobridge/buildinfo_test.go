package main

// Bridge build identity pins (2026-11-12 view-once diagnostic).
//
// The owner's second live view-once failure: a view-once photo
// arrived, the toggle sat ON, the journal held zero rows.  The
// chain on main is complete on paper, so the live break is
// either the RUNNING bridge binary predating the 2026-11-08
// unwrapInbound fix — or the arrival never reaching the
// observer.  Neither was provable from the Python side: the
// bridge's version string is a build-invariant constant
// ("gobridge/1.0.0" in every build) and the supervisor's
// orphan-adoption check keys on that same constant, so a
// pre-fix binary verifies exactly like a current one.
//
// The status payload therefore carries a build block — commit
// + built time stamped at build time via -ldflags, and the
// unwrap_inbound capability flag naming the fix itself — so
// the next live demo can prove which binary is running.

import "testing"

func TestStatusPayloadCarriesBuildIdentity(t *testing.T) {
	b := &bridge{port: 8787, connState: "open",
		meJID: "2348065574087@s.whatsapp.net"}
	payload := b.statusPayload()
	build, ok := payload["build"].(map[string]any)
	if !ok {
		t.Fatalf("status payload has no build block: %#v", payload)
	}
	if build["program"] != bridgeVersion {
		t.Errorf("build.program = %v, want %v",
			build["program"], bridgeVersion)
	}
	if build["unwrap_inbound"] != true {
		t.Errorf("build.unwrap_inbound = %v, want true",
			build["unwrap_inbound"])
	}
	if build["commit"] == nil || build["commit"] == "" {
		t.Errorf("build.commit is empty: %#v", build)
	}
	if build["built"] == nil || build["built"] == "" {
		t.Errorf("build.built is empty: %#v", build)
	}
}

func TestVersionStringNamesBuild(t *testing.T) {
	s := versionString()
	if !contains(s, bridgeVersion) {
		t.Errorf("version string lacks program: %q", s)
	}
	if !contains(s, "commit") {
		t.Errorf("version string lacks commit: %q", s)
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
