package main

import (
	"bytes"
	"encoding/json"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/cookiejar"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

const testSessionID = "upstream-session-that-must-not-be-rendered"

type fakeKYCAPI struct {
	mu              sync.Mutex
	status          string
	sides           []string
	resultAvailable bool
	created         bool
	processCalled   bool
	deleted         bool
	notFound        bool
	rateLimited     bool
	uploads         map[string]fakeUpload
	result          map[string]any
}

type fakeUpload struct {
	Filename    string
	ContentType string
	Content     []byte
}

func newFakeKYCAPI() *fakeKYCAPI {
	return &fakeKYCAPI{
		status:  "created",
		uploads: make(map[string]fakeUpload),
		result: map[string]any{
			"status": "success",
			"front": map[string]any{
				"status":        "success",
				"document_type": "ao_id_card",
				"profile_id":    "ao_id_card/front/v1",
				"detection":     map[string]any{"confidence": 0.98, "orientation_degrees": 0},
				"fields": map[string]any{
					"full_name": map[string]any{"status": "valid", "raw_value": "Amina Raw", "normalized_value": "Amina Example", "confidence": 0.98, "warnings": []string{}},
				},
				"issues": []any{},
			},
			"back": map[string]any{
				"status":        "success",
				"document_type": "ao_id_card",
				"profile_id":    "ao_id_card/back/v1",
				"fields":        map[string]any{},
				"qr_code": map[string]any{
					"status":      "decoded",
					"raw_payload": "diagnostic-secret-payload",
					"data":        map[string]any{"id_number": "AO-12345"},
					"warnings":    []string{},
				},
				"issues": []any{},
			},
			"issues": []any{},
		},
	}
}

func (fake *fakeKYCAPI) ServeHTTP(writer http.ResponseWriter, request *http.Request) {
	if request.Header.Get("X-API-Key") != "test-key-that-stays-on-the-server" {
		writeAPIError(writer, http.StatusUnauthorized, "UNAUTHORIZED", "Missing API key")
		return
	}
	fake.mu.Lock()
	defer fake.mu.Unlock()
	if fake.rateLimited {
		writer.Header().Set("Retry-After", "3")
		writeAPIError(writer, http.StatusTooManyRequests, "RATE_LIMITED", "Too many requests")
		return
	}
	if fake.notFound {
		writeAPIError(writer, http.StatusNotFound, "SESSION_NOT_FOUND", "Session was not found")
		return
	}
	switch {
	case request.Method == http.MethodPost && request.URL.Path == "/v1/sessions":
		fake.created = true
		fake.writeSession(writer)
	case request.Method == http.MethodPost && strings.HasPrefix(request.URL.Path, "/v1/sessions/"+testSessionID+"/images/"):
		side := strings.TrimPrefix(request.URL.Path, "/v1/sessions/"+testSessionID+"/images/")
		reader, err := request.MultipartReader()
		if err != nil {
			writeAPIError(writer, http.StatusBadRequest, "INVALID_REQUEST", "Bad upload")
			return
		}
		part, err := reader.NextPart()
		if err != nil {
			writeAPIError(writer, http.StatusBadRequest, "INVALID_REQUEST", "Missing image")
			return
		}
		content, _ := io.ReadAll(part)
		fake.uploads[side] = fakeUpload{Filename: part.FileName(), ContentType: part.Header.Get("Content-Type"), Content: content}
		if !contains(fake.sides, side) {
			fake.sides = append(fake.sides, side)
		}
		fake.status = "uploading"
		writeJSON(writer, http.StatusOK, map[string]any{
			"session_id": testSessionID, "status": fake.status, "side": side,
			"uploaded_sides": fake.sides, "expires_at": futureTime(),
		})
	case request.Method == http.MethodGet && request.URL.Path == "/v1/sessions/"+testSessionID:
		fake.writeSession(writer)
	case request.Method == http.MethodPost && request.URL.Path == "/v1/sessions/"+testSessionID+"/process":
		fake.processCalled = true
		fake.status = "queued"
		writeJSON(writer, http.StatusAccepted, map[string]any{"session_id": testSessionID, "job_id": "job-id-hidden", "status": "queued"})
	case request.Method == http.MethodGet && request.URL.Path == "/v1/sessions/"+testSessionID+"/result":
		writeJSON(writer, http.StatusOK, fake.result)
	case request.Method == http.MethodDelete && request.URL.Path == "/v1/sessions/"+testSessionID:
		fake.deleted = true
		writeJSON(writer, http.StatusOK, map[string]any{"deleted": true})
	default:
		writeAPIError(writer, http.StatusNotFound, "SESSION_NOT_FOUND", "Session was not found")
	}
}

func (fake *fakeKYCAPI) writeSession(writer http.ResponseWriter) {
	writeJSON(writer, http.StatusOK, map[string]any{
		"session_id": testSessionID, "status": fake.status, "created_at": pastTime(), "expires_at": futureTime(),
		"job_id": "job-id-hidden", "uploaded_sides": fake.sides, "result_available": fake.resultAvailable,
	})
}

func newConsole(t *testing.T, fake *fakeKYCAPI, maxUploadBytes int64) (*httptest.Server, *http.Client) {
	t.Helper()
	upstream := httptest.NewServer(fake)
	t.Cleanup(upstream.Close)
	parsed, err := loadConfig(func(key string) (string, bool) {
		values := map[string]string{
			"KYC_WEB_API_URL":          upstream.URL,
			"KYC_API_KEY":              "test-key-that-stays-on-the-server",
			"KYC_WEB_MAX_UPLOAD_BYTES": "15728640",
			"KYC_WEB_COOKIE_SECURE":    "false",
			"KYC_WEBHOOK_SECRET":       testWebhookSecret,
		}
		if maxUploadBytes > 0 {
			values["KYC_WEB_MAX_UPLOAD_BYTES"] = stringNumber(maxUploadBytes)
		}
		value, ok := values[key]
		return value, ok
	})
	if err != nil {
		t.Fatal(err)
	}
	app, err := newServer(parsed, newKYCClient(parsed, &http.Client{Timeout: 2 * time.Second}), newSessionStore())
	if err != nil {
		t.Fatal(err)
	}
	console := httptest.NewServer(app.routes())
	t.Cleanup(console.Close)
	jar, err := cookiejar.New(nil)
	if err != nil {
		t.Fatal(err)
	}
	return console, &http.Client{Jar: jar, Timeout: 2 * time.Second}
}

func TestUploadsAreProxiedWithoutExposingSecretsOrFilenames(t *testing.T) {
	fake := newFakeKYCAPI()
	console, client := newConsole(t, fake, 0)

	response := uploadRequest(t, client, console.URL, "front", "private-amina-id.jpeg", "image/jpeg", []byte{0xff, 0xd8, 0xff, 0x01})
	body := responseBody(t, response)
	if !fake.created {
		t.Fatal("expected the BFF to create an upstream session")
	}
	upload := fake.uploads["front"]
	if upload.Filename != "front.jpg" || upload.ContentType != "image/jpeg" || !bytes.Equal(upload.Content, []byte{0xff, 0xd8, 0xff, 0x01}) {
		t.Fatalf("unexpected proxied upload: %#v", upload)
	}
	for _, secret := range []string{"private-amina-id.jpeg", testSessionID, "test-key-that-stays-on-the-server"} {
		if strings.Contains(body, secret) {
			t.Fatalf("sensitive value %q was rendered", secret)
		}
	}
	if !strings.Contains(body, "One side uploaded") {
		t.Fatalf("upload state was not rendered: %s", body)
	}
}

func TestProcessRequiresBothSides(t *testing.T) {
	fake := newFakeKYCAPI()
	console, client := newConsole(t, fake, 0)
	_ = responseBody(t, uploadRequest(t, client, console.URL, "front", "front.png", "image/png", []byte("front")))

	response := postHTMX(t, client, console.URL+"/checks/process")
	body := responseBody(t, response)
	if fake.processCalled {
		t.Fatal("the BFF started verification with only one uploaded side")
	}
	if !strings.Contains(body, "Both document sides are required") {
		t.Fatalf("missing both-side guidance: %s", body)
	}
}

func TestStatusRendersSafeReviewSummaryAndStopsPollingAtCompletion(t *testing.T) {
	fake := newFakeKYCAPI()
	console, client := newConsole(t, fake, 0)
	_ = responseBody(t, uploadRequest(t, client, console.URL, "front", "front.jpg", "image/jpeg", []byte("front")))
	_ = responseBody(t, uploadRequest(t, client, console.URL, "back", "back.png", "image/png", []byte("back")))
	queued := responseBody(t, postHTMX(t, client, console.URL+"/checks/process"))
	if !strings.Contains(queued, "every 2s") || !strings.Contains(queued, "Verification queued") {
		t.Fatalf("active verification should poll: %s", queued)
	}

	fake.mu.Lock()
	fake.status = "success"
	fake.resultAvailable = true
	fake.mu.Unlock()
	response := getHTMX(t, client, console.URL+"/checks/status")
	body := responseBody(t, response)
	for _, expected := range []string{"Verification complete", "Front side", "Full Name", "Amina Example", "QR data", "AO-12345"} {
		if !strings.Contains(body, expected) {
			t.Fatalf("expected %q in review summary: %s", expected, body)
		}
	}
	for _, omitted := range []string{"diagnostic-secret-payload", testSessionID, "job-id-hidden", "raw_payload", "alternatives"} {
		if strings.Contains(body, omitted) {
			t.Fatalf("diagnostic value %q was rendered", omitted)
		}
	}
	if strings.Contains(body, "every 2s") {
		t.Fatalf("terminal state must stop polling: %s", body)
	}
}

func TestResetDeletesUpstreamSessionAndClearsState(t *testing.T) {
	fake := newFakeKYCAPI()
	console, client := newConsole(t, fake, 0)
	_ = responseBody(t, uploadRequest(t, client, console.URL, "front", "front.jpg", "image/jpeg", []byte("front")))

	body := responseBody(t, postHTMX(t, client, console.URL+"/checks/reset"))
	if !fake.deleted {
		t.Fatal("expected reset to delete the upstream session")
	}
	if !strings.Contains(body, "Ready for documents") || strings.Contains(body, "One side uploaded") {
		t.Fatalf("reset did not clear the rendered state: %s", body)
	}
}

func TestExpiredUpstreamSessionClearsTheBrowserMapping(t *testing.T) {
	fake := newFakeKYCAPI()
	console, client := newConsole(t, fake, 0)
	_ = responseBody(t, uploadRequest(t, client, console.URL, "front", "front.jpg", "image/jpeg", []byte("front")))

	fake.mu.Lock()
	fake.notFound = true
	fake.mu.Unlock()
	response := getHTMX(t, client, console.URL+"/checks/status")
	body := responseBody(t, response)
	if !strings.Contains(body, "This check has expired") {
		t.Fatalf("missing expiry guidance: %s", body)
	}
	if !strings.Contains(response.Header.Get("Set-Cookie"), "Max-Age=0") {
		t.Fatalf("expired session did not clear its cookie: %s", response.Header.Get("Set-Cookie"))
	}

	body = responseBody(t, getHTMX(t, client, console.URL+"/checks/status"))
	if !strings.Contains(body, "Ready for documents") {
		t.Fatalf("cleared session should not keep polling the API: %s", body)
	}
}

func TestPartialResultIsPresentedAsReviewRequired(t *testing.T) {
	fake := newFakeKYCAPI()
	console, client := newConsole(t, fake, 0)
	_ = responseBody(t, uploadRequest(t, client, console.URL, "front", "front.jpg", "image/jpeg", []byte("front")))
	_ = responseBody(t, uploadRequest(t, client, console.URL, "back", "back.jpg", "image/jpeg", []byte("back")))

	fake.mu.Lock()
	fake.status = "partial"
	fake.resultAvailable = true
	fake.result["status"] = "partial"
	fake.mu.Unlock()
	body := responseBody(t, getHTMX(t, client, console.URL+"/checks/status"))
	if !strings.Contains(body, "Review required") || !strings.Contains(body, "Some document details need operator review") {
		t.Fatalf("partial result was not presented for review: %s", body)
	}
}

func TestOversizedUploadIsRejectedBeforeTheUpstreamCall(t *testing.T) {
	fake := newFakeKYCAPI()
	console, client := newConsole(t, fake, 4)
	body := responseBody(t, uploadRequest(t, client, console.URL, "front", "front.png", "image/png", []byte("12345")))
	if fake.created || len(fake.uploads) != 0 {
		t.Fatal("oversized content must not create or upload to an upstream session")
	}
	if !strings.Contains(body, "image must be 4 bytes or smaller") {
		t.Fatalf("missing size guidance: %s", body)
	}
}

func TestRateLimitResponseIncludesTheRetryDelay(t *testing.T) {
	fake := newFakeKYCAPI()
	fake.rateLimited = true
	console, client := newConsole(t, fake, 0)
	body := responseBody(t, uploadRequest(t, client, console.URL, "front", "front.jpg", "image/jpeg", []byte("front")))
	if !strings.Contains(body, "Too many requests") || !strings.Contains(body, "Try again in 3 seconds.") {
		t.Fatalf("rate-limit guidance was not rendered: %s", body)
	}
}

func uploadRequest(t *testing.T, client *http.Client, baseURL string, side string, filename string, contentType string, content []byte) *http.Response {
	t.Helper()
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreatePart(textPartHeader("image", filename, contentType))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := part.Write(content); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	request, err := http.NewRequest(http.MethodPost, baseURL+"/checks/images/"+side, &body)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Content-Type", writer.FormDataContentType())
	request.Header.Set("HX-Request", "true")
	response, err := client.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	return response
}

func postHTMX(t *testing.T, client *http.Client, target string) *http.Response {
	t.Helper()
	request, err := http.NewRequest(http.MethodPost, target, nil)
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

func getHTMX(t *testing.T, client *http.Client, target string) *http.Response {
	t.Helper()
	request, err := http.NewRequest(http.MethodGet, target, nil)
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
	if response.StatusCode != http.StatusOK {
		payload, _ := io.ReadAll(response.Body)
		t.Fatalf("unexpected HTTP %d: %s", response.StatusCode, payload)
	}
	payload, err := io.ReadAll(response.Body)
	if err != nil {
		t.Fatal(err)
	}
	return string(payload)
}

func writeAPIError(writer http.ResponseWriter, status int, code string, message string) {
	writeJSON(writer, status, map[string]any{"error": map[string]string{"code": code, "message": message}})
}

func writeJSON(writer http.ResponseWriter, status int, payload any) {
	writer.Header().Set("Content-Type", "application/json")
	writer.WriteHeader(status)
	_ = json.NewEncoder(writer).Encode(payload)
}

func pastTime() string   { return time.Now().UTC().Add(-time.Minute).Format(time.RFC3339) }
func futureTime() string { return time.Now().UTC().Add(time.Hour).Format(time.RFC3339) }

func contains(values []string, wanted string) bool {
	for _, value := range values {
		if value == wanted {
			return true
		}
	}
	return false
}

func stringNumber(value int64) string {
	return strconv.FormatInt(value, 10)
}
