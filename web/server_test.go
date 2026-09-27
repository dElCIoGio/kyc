package main

import (
	"bytes"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"net/http/cookiejar"
	"net/http/httptest"
	"net/url"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

const testWebhookSecret = "test-webhook-secret-that-is-long-enough-0123456789"
const testAPIKey = "test-key-that-stays-on-the-server"

type fakeKYCAPI struct {
	mu         sync.Mutex
	sequence   int
	sessions   map[string]upstreamSession
	issued     map[string]int
	deleted    []string
	seenAPIKey bool
}

func newFakeKYCAPI() *fakeKYCAPI {
	return &fakeKYCAPI{sessions: make(map[string]upstreamSession), issued: make(map[string]int)}
}
func (fake *fakeKYCAPI) ServeHTTP(writer http.ResponseWriter, request *http.Request) {
	fake.mu.Lock()
	defer fake.mu.Unlock()
	if request.Header.Get("X-API-Key") != testAPIKey {
		writeJSON(writer, http.StatusUnauthorized, map[string]any{"error": map[string]string{"code": "UNAUTHORIZED", "message": "Authentication failed"}})
		return
	}
	fake.seenAPIKey = true
	path := request.URL.Path
	if request.Method == http.MethodPost && path == "/v1/sessions" {
		fake.sequence++
		id := "vs_test_" + string(rune('0'+fake.sequence))
		session := fakeSession(id)
		fake.sessions[id] = session
		writeJSON(writer, http.StatusCreated, session)
		return
	}
	for id, session := range fake.sessions {
		if path == "/v1/sessions/"+id+"/browser-token" && request.Method == http.MethodPost {
			fake.issued[id]++
			writeJSON(writer, http.StatusOK, map[string]any{"verification_url": "https://kyc.example/verify/" + id + "#bt_test_credential_" + string(rune('0'+fake.issued[id])) + "_abcdefghijklmnop", "expires_at": futureTime()})
			return
		}
		if path == "/v1/sessions/"+id && request.Method == http.MethodGet {
			writeJSON(writer, http.StatusOK, session)
			return
		}
		if path == "/v1/sessions/"+id+"/result" && request.Method == http.MethodGet {
			writeJSON(writer, http.StatusOK, map[string]any{"session_id": id, "document": map[string]any{"status": "completed", "document_type": "ao_id_card", "fields": map[string]any{"full_name": map[string]any{"status": "valid", "value": "Amina Example", "raw_ocr_candidates": "must-not-render"}}, "issues": []any{}}, "liveness": map[string]any{"status": "passed"}, "face_comparison": map[string]any{"status": "completed"}, "raw_qr_payload": "must-not-render"})
			return
		}
		if path == "/v1/sessions/"+id && request.Method == http.MethodDelete {
			delete(fake.sessions, id)
			fake.deleted = append(fake.deleted, id)
			writer.WriteHeader(http.StatusNoContent)
			return
		}
	}
	writeJSON(writer, http.StatusNotFound, map[string]any{"error": map[string]string{"code": "SESSION_NOT_FOUND", "message": "Session was not found"}})
}
func fakeSession(id string) upstreamSession {
	action := "submit_document_front"
	return upstreamSession{SessionID: id, Status: "in_progress", CreatedAt: time.Now().UTC().Add(-time.Minute), ExpiresAt: time.Now().UTC().Add(time.Hour), NextAction: &action, Document: documentState{Status: "awaiting_capture", FrontCapture: "missing", BackCapture: "missing"}, Liveness: checkState{Status: "not_started"}, FaceComparison: checkState{Status: "not_available"}}
}
func futureTime() string { return time.Now().UTC().Add(time.Hour).Format(time.RFC3339) }

func newSandbox(t *testing.T, fake *fakeKYCAPI) (*httptest.Server, *http.Client, *server) {
	t.Helper()
	upstream := httptest.NewServer(fake)
	t.Cleanup(upstream.Close)
	parsed, err := url.Parse(upstream.URL)
	if err != nil {
		t.Fatal(err)
	}
	config := Config{Address: ":0", APIURL: parsed, APIKey: testAPIKey, WebhookSecret: testWebhookSecret}
	app, err := newServer(config, newKYCClient(config, &http.Client{Timeout: 2 * time.Second}), newSessionStore())
	if err != nil {
		t.Fatal(err)
	}
	console := httptest.NewServer(app.routes())
	t.Cleanup(console.Close)
	jar, err := cookiejar.New(nil)
	if err != nil {
		t.Fatal(err)
	}
	return console, &http.Client{Jar: jar, Timeout: 2 * time.Second}, app
}
func htmxRequest(t *testing.T, client *http.Client, method, target string) *http.Response {
	t.Helper()
	request, err := http.NewRequest(method, target, nil)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("HX-Request", "true")
	response, err := client.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	return response
}
func responseBody(t *testing.T, response *http.Response) string {
	t.Helper()
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	if err != nil {
		t.Fatal(err)
	}
	if response.StatusCode != http.StatusOK {
		t.Fatalf("unexpected HTTP %d: %s", response.StatusCode, body)
	}
	return string(body)
}
func writeJSON(writer http.ResponseWriter, status int, payload any) {
	writer.Header().Set("Content-Type", "application/json")
	writer.WriteHeader(status)
	_ = json.NewEncoder(writer).Encode(payload)
}

func TestCreateIssuesHostedCredentialAndKeepsAPIKeyServerSide(t *testing.T) {
	fake := newFakeKYCAPI()
	console, client, _ := newSandbox(t, fake)
	body := responseBody(t, htmxRequest(t, client, http.MethodPost, console.URL+"/sandbox/sessions"))
	for _, expected := range []string{"vs_test_1", "https://kyc.example/verify/vs_test_1#bt_test_credential_1_abcdefghijklmnop", "Open verification"} {
		if !strings.Contains(body, expected) {
			t.Fatalf("missing %q: %s", expected, body)
		}
	}
	if strings.Contains(body, testAPIKey) {
		t.Fatalf("API key was rendered: %s", body)
	}
	if !fake.seenAPIKey || fake.issued["vs_test_1"] != 1 {
		t.Fatalf("expected server-side create and issue, got %+v", fake)
	}
}

func TestRotateFetchResultAndDelete(t *testing.T) {
	fake := newFakeKYCAPI()
	console, client, _ := newSandbox(t, fake)
	_ = responseBody(t, htmxRequest(t, client, http.MethodPost, console.URL+"/sandbox/sessions"))
	body := responseBody(t, htmxRequest(t, client, http.MethodPost, console.URL+"/sandbox/sessions/vs_test_1/browser-token"))
	if !strings.Contains(body, "bt_test_credential_2_abcdefghijklmnop") || strings.Contains(body, "bt_test_credential_1_abcdefghijklmnop") {
		t.Fatalf("rotation did not replace link: %s", body)
	}
	fake.mu.Lock()
	session := fake.sessions["vs_test_1"]
	session.Status = "completed"
	session.NextAction = nil
	session.Document = documentState{Status: "completed", FrontCapture: "accepted", BackCapture: "accepted", ResultAvailable: true}
	session.Liveness = checkState{Status: "passed"}
	session.FaceComparison = checkState{Status: "completed"}
	fake.sessions["vs_test_1"] = session
	fake.mu.Unlock()
	_ = responseBody(t, htmxRequest(t, client, http.MethodGet, console.URL+"/sandbox/sessions/vs_test_1/status"))
	body = responseBody(t, htmxRequest(t, client, http.MethodPost, console.URL+"/sandbox/sessions/vs_test_1/result"))
	for _, expected := range []string{"Amina Example", "Verification processing completed"} {
		if !strings.Contains(body, expected) {
			t.Fatalf("missing %q: %s", expected, body)
		}
	}
	for _, hidden := range []string{"must-not-render", "raw_ocr_candidates", "raw_qr_payload"} {
		if strings.Contains(body, hidden) {
			t.Fatalf("unsafe result field rendered: %s", body)
		}
	}
	body = responseBody(t, htmxRequest(t, client, http.MethodPost, console.URL+"/sandbox/sessions/vs_test_1/delete"))
	if len(fake.deleted) != 1 || strings.Contains(body, "vs_test_1") {
		t.Fatalf("delete did not clear sandbox state: %s", body)
	}
}

func webhookBody(t *testing.T, id, sessionID, eventType string, sequence int) []byte {
	t.Helper()
	return webhookEvent(t, id, sessionID, eventType, sequence, time.Now().UTC(), "in_progress", false)
}
func webhookEvent(t *testing.T, id, sessionID, eventType string, sequence int, createdAt time.Time, status string, resultAvailable bool) []byte {
	t.Helper()
	body, err := json.Marshal(map[string]any{"schema_version": "1", "id": id, "type": eventType, "created_at": createdAt.Format(time.RFC3339), "data": map[string]any{"session_id": sessionID, "sequence": sequence, "status": status, "next_action": "wait", "result_available": resultAvailable}})
	if err != nil {
		t.Fatal(err)
	}
	return body
}
func signature(timestamp string, body []byte) string {
	mac := hmac.New(sha256.New, []byte(testWebhookSecret))
	_, _ = mac.Write([]byte(timestamp))
	_, _ = mac.Write([]byte("."))
	_, _ = mac.Write(body)
	return "v1=" + hex.EncodeToString(mac.Sum(nil))
}
func sendWebhook(t *testing.T, target, id string, body []byte, valid bool) *http.Response {
	t.Helper()
	return sendWebhookAt(t, target, id, body, valid, time.Now())
}
func sendWebhookAt(t *testing.T, target, id string, body []byte, valid bool, sentAt time.Time) *http.Response {
	t.Helper()
	request, err := http.NewRequest(http.MethodPost, target+"/webhooks", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	timestamp := strconv.FormatInt(sentAt.Unix(), 10)
	request.Header.Set("X-KYC-Webhook-ID", id)
	request.Header.Set("X-KYC-Webhook-Timestamp", timestamp)
	if valid {
		request.Header.Set("X-KYC-Webhook-Signature", signature(timestamp, body))
	}
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	return response
}
func expectStatus(t *testing.T, response *http.Response, want int) {
	t.Helper()
	defer response.Body.Close()
	if response.StatusCode != want {
		t.Fatalf("webhook HTTP %d, want %d", response.StatusCode, want)
	}
}
func trackedFor(t *testing.T, app *server, client *http.Client, target string, sessionID string) trackedSession {
	t.Helper()
	parsed, err := url.Parse(target)
	if err != nil {
		t.Fatal(err)
	}
	cookies := client.Jar.Cookies(parsed)
	if len(cookies) == 0 {
		t.Fatalf("no sandbox cookie was issued to this browser")
	}
	tracked, ok := app.sessions.get(cookies[0].Value, sessionID)
	if !ok {
		t.Fatalf("session %s is not tracked by this browser", sessionID)
	}
	return tracked
}
func TestCurrentWebhookEnvelopeValidatesDeduplicatesAndOrders(t *testing.T) {
	fake := newFakeKYCAPI()
	console, client, app := newSandbox(t, fake)
	_ = responseBody(t, htmxRequest(t, client, http.MethodPost, console.URL+"/sandbox/sessions"))
	first := webhookBody(t, "evt-processing", "vs_test_1", "verification.processing.failed", 4)
	second := webhookBody(t, "evt-document", "vs_test_1", "verification.document.failed", 4)
	for _, call := range []struct {
		id   string
		body []byte
	}{{"evt-processing", first}, {"evt-document", second}, {"evt-document", second}} {
		response := sendWebhook(t, console.URL, call.id, call.body, true)
		if response.StatusCode != http.StatusNoContent {
			t.Fatalf("webhook HTTP %d", response.StatusCode)
		}
		response.Body.Close()
	}
	invalid := sendWebhook(t, console.URL, "evt-invalid", webhookBody(t, "evt-invalid", "vs_test_1", "verification.processing.completed", 5), false)
	if invalid.StatusCode != http.StatusUnauthorized {
		t.Fatalf("invalid webhook HTTP %d", invalid.StatusCode)
	}
	invalid.Body.Close()
	parsed, _ := url.Parse(console.URL)
	cookie := client.Jar.Cookies(parsed)[0]
	tracked, ok := app.sessions.selected(cookie.Value)
	if !ok || len(tracked.Webhooks) != 2 {
		t.Fatalf("stored webhooks: %#v", tracked.Webhooks)
	}
	if tracked.Webhooks[0].Type != "verification.document.failed" || tracked.Webhooks[1].Type != "verification.processing.failed" {
		t.Fatalf("webhook order was not deterministic: %#v", tracked.Webhooks)
	}
}

func TestAPIKeyStaysServerSideAndIsAbsentFromHTMLAndAssets(t *testing.T) {
	fake := newFakeKYCAPI()
	console, client, _ := newSandbox(t, fake)
	_ = responseBody(t, htmxRequest(t, client, http.MethodPost, console.URL+"/sandbox/sessions"))
	page, err := client.Get(console.URL + "/")
	if err != nil {
		t.Fatal(err)
	}
	if page.StatusCode != http.StatusOK {
		t.Fatalf("index HTTP %d", page.StatusCode)
	}
	bodies := map[string]string{"/": responseBody(t, page)}
	for _, asset := range []string{"/assets/app.css", "/assets/app.js", "/assets/htmx.min.js"} {
		response, err := client.Get(console.URL + asset)
		if err != nil {
			t.Fatal(err)
		}
		if response.StatusCode != http.StatusOK {
			t.Fatalf("asset %s HTTP %d", asset, response.StatusCode)
		}
		bodies[asset] = responseBody(t, response)
	}
	for asset, body := range bodies {
		if strings.Contains(body, testAPIKey) {
			t.Fatalf("API key leaked into %s", asset)
		}
		if strings.Contains(body, testWebhookSecret) {
			t.Fatalf("webhook secret leaked into %s", asset)
		}
	}
	if !fake.seenAPIKey || fake.issued["vs_test_1"] != 1 {
		t.Fatalf("expected server-side create and issue, got %+v", fake)
	}
}

func TestWebhookRejectsStaleOrMalformedEnvelopesAndIgnoresUnknownSessions(t *testing.T) {
	fake := newFakeKYCAPI()
	console, client, app := newSandbox(t, fake)
	_ = responseBody(t, htmxRequest(t, client, http.MethodPost, console.URL+"/sandbox/sessions"))

	// A correctly signed event outside the timestamp window is unauthorized.
	expectStatus(t, sendWebhookAt(t, console.URL, "evt-stale", webhookBody(t, "evt-stale", "vs_test_1", "verification.document.completed", 2), true, time.Now().Add(-time.Hour)), http.StatusUnauthorized)
	// A bad signature is unauthorized even with a fresh timestamp.
	expectStatus(t, sendWebhook(t, console.URL, "evt-unsigned", webhookBody(t, "evt-unsigned", "vs_test_1", "verification.document.completed", 2), false), http.StatusUnauthorized)

	rejected := []struct {
		name   string
		header string
		body   []byte
	}{
		{"header id mismatch", "evt-header", webhookBody(t, "evt-payload", "vs_test_1", "verification.document.completed", 2)},
		{"unknown event type", "evt-type", webhookBody(t, "evt-type", "vs_test_1", "verification.review.requested", 2)},
		{"zero sequence", "evt-sequence", webhookBody(t, "evt-sequence", "vs_test_1", "verification.document.completed", 0)},
	}
	for _, call := range rejected {
		expectStatus(t, sendWebhook(t, console.URL, call.header, call.body, true), http.StatusBadRequest)
	}
	malformed := map[string]any{"schema_version": "2", "id": "evt-schema", "type": "verification.document.completed", "created_at": time.Now().UTC().Format(time.RFC3339), "data": map[string]any{"session_id": "vs_test_1", "sequence": 2, "status": "in_progress", "next_action": "wait", "result_available": false}}
	encoded, err := json.Marshal(malformed)
	if err != nil {
		t.Fatal(err)
	}
	expectStatus(t, sendWebhook(t, console.URL, "evt-schema", encoded, true), http.StatusBadRequest)

	// A well-formed event for a session this sandbox does not track is
	// acknowledged but never attached to a browser session.
	expectStatus(t, sendWebhook(t, console.URL, "evt-unknown-session", webhookBody(t, "evt-unknown-session", "vs_test_9", "verification.document.completed", 2), true), http.StatusNoContent)

	tracked := trackedFor(t, app, client, console.URL, "vs_test_1")
	if len(tracked.Webhooks) != 0 || len(tracked.Milestones) != 2 {
		t.Fatalf("rejected or unassociated events were recorded: %#v", tracked)
	}
}

func TestTerminalWebhooksRecordMilestonesAndDurations(t *testing.T) {
	fake := newFakeKYCAPI()
	console, client, app := newSandbox(t, fake)
	_ = responseBody(t, htmxRequest(t, client, http.MethodPost, console.URL+"/sandbox/sessions"))

	base := time.Now().UTC().Add(-90 * time.Second)
	events := []struct {
		id, eventType, milestone string
		sequence                 int
		offset                   time.Duration
		status                   string
	}{
		{"evt-doc", "verification.document.completed", "Document Terminal", 2, 10 * time.Second, "in_progress"},
		{"evt-live", "verification.liveness.passed", "Liveness Terminal", 3, 40 * time.Second, "in_progress"},
		{"evt-proc", "verification.processing.completed", "Processing Terminal", 4, 80 * time.Second, "completed"},
	}
	for _, event := range events {
		created := base.Add(event.offset)
		expectStatus(t, sendWebhook(t, console.URL, event.id, webhookEvent(t, event.id, "vs_test_1", event.eventType, event.sequence, created, event.status, true), true), http.StatusNoContent)
	}
	tracked := trackedFor(t, app, client, console.URL, "vs_test_1")
	milestones := append([]localMilestone(nil), tracked.Milestones...)
	sort.Slice(milestones, func(i, j int) bool { return milestones[i].ObservedAt.Before(milestones[j].ObservedAt) })
	if len(milestones) != 5 {
		t.Fatalf("expected 2 local and 3 webhook milestones, got %#v", milestones)
	}
	want := []string{"Session Created", "Browser Credential Issued", "Document Terminal", "Liveness Terminal", "Processing Terminal"}
	for index, milestone := range milestones {
		if got := label(milestone.Type); got != want[index] {
			t.Fatalf("milestone %d is %q, want %q", index, got, want[index])
		}
	}
	for _, milestone := range milestones[2:] {
		if milestone.APICreatedAt.IsZero() {
			t.Fatalf("webhook milestone %q has no API-created time", milestone.Type)
		}
	}
	if !milestones[0].APICreatedAt.IsZero() {
		t.Fatalf("locally issued milestone must not carry an API-created time")
	}

	body := responseBody(t, htmxRequest(t, client, http.MethodGet, console.URL+"/sandbox/sessions/vs_test_1/status"))
	for _, expected := range []string{"Document Terminal", "Liveness Terminal", "Processing Terminal", "Session creation → document terminal", "Document terminal → liveness terminal", "Liveness terminal → processing terminal", "Session creation → processing terminal", "Processing event → Go receipt"} {
		if !strings.Contains(body, expected) {
			t.Fatalf("missing %q: %s", expected, body)
		}
	}
}

func TestRecentSessionSwitcherIsScopedToTheVisitorCookie(t *testing.T) {
	fake := newFakeKYCAPI()
	console, client, _ := newSandbox(t, fake)
	_ = responseBody(t, htmxRequest(t, client, http.MethodPost, console.URL+"/sandbox/sessions"))
	body := responseBody(t, htmxRequest(t, client, http.MethodPost, console.URL+"/sandbox/sessions"))
	if !strings.Contains(body, "vs_test_2") {
		t.Fatalf("newest session was not tracked: %s", body)
	}

	body = responseBody(t, htmxRequest(t, client, http.MethodPost, console.URL+"/sandbox/sessions/vs_test_1/select"))
	if !strings.Contains(body, "vs_test_1") {
		t.Fatalf("switching to the older session failed: %s", body)
	}
	// Selecting an untracked session reports the miss but must not change the
	// session the switcher has selected.
	body = responseBody(t, htmxRequest(t, client, http.MethodPost, console.URL+"/sandbox/sessions/vs_test_9/select"))
	if !strings.Contains(body, "Unknown sandbox session") || !strings.Contains(body, `data-session-id="vs_test_1"`) {
		t.Fatalf("an untracked selection changed the active session: %s", body)
	}

	// A separate browser with its own cookie sees none of these sessions.
	jar, err := cookiejar.New(nil)
	if err != nil {
		t.Fatal(err)
	}
	other := &http.Client{Jar: jar, Timeout: 2 * time.Second}
	otherPage, err := other.Get(console.URL + "/")
	if err != nil {
		t.Fatal(err)
	}
	otherBody := responseBody(t, otherPage)
	for _, sessionID := range []string{"vs_test_1", "vs_test_2"} {
		if strings.Contains(otherBody, sessionID) {
			t.Fatalf("session %s leaked into another browser's sandbox", sessionID)
		}
	}
}

func TestTerminalWordingNeverClaimsIdentityApproval(t *testing.T) {
	fake := newFakeKYCAPI()
	console, client, _ := newSandbox(t, fake)
	_ = responseBody(t, htmxRequest(t, client, http.MethodPost, console.URL+"/sandbox/sessions"))
	fake.mu.Lock()
	session := fake.sessions["vs_test_1"]
	session.Status = "completed"
	session.NextAction = nil
	session.Document = documentState{Status: "completed", FrontCapture: "accepted", BackCapture: "accepted", ResultAvailable: true}
	fake.sessions["vs_test_1"] = session
	fake.mu.Unlock()
	body := responseBody(t, htmxRequest(t, client, http.MethodGet, console.URL+"/sandbox/sessions/vs_test_1/status"))
	if !strings.Contains(body, "Verification processing completed") {
		t.Fatalf("terminal wording missing: %s", body)
	}
	for _, forbidden := range []string{"Approved", "approved", "Verified identity", "identity approved"} {
		if strings.Contains(body, forbidden) {
			t.Fatalf("terminal state claimed identity approval via %q: %s", forbidden, body)
		}
	}
}
