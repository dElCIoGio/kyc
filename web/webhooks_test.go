package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
	"testing"
	"time"
)

const testWebhookSecret = "test-webhook-secret-that-is-long-enough-0123456789"

func webhookBodyBytes(t *testing.T, id string, sessionID string, sequence int, eventType string) []byte {
	t.Helper()
	body, err := json.Marshal(map[string]any{
		"version":     "1",
		"id":          id,
		"type":        eventType,
		"occurred_at": time.Now().UTC().Format(time.RFC3339),
		"data": map[string]any{
			"session_id":       sessionID,
			"job_id":           "job-id-hidden",
			"sequence":         sequence,
			"status":           "running",
			"result_available": false,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	return body
}

func signWebhook(secret string, timestamp string, body []byte) string {
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write([]byte(timestamp))
	mac.Write([]byte("."))
	mac.Write(body)
	return "v1=" + hex.EncodeToString(mac.Sum(nil))
}

func webhookHeaders(id string, timestamp string, signature string) map[string]string {
	return map[string]string{
		"Content-Type":            "application/json",
		"X-KYC-Webhook-ID":        id,
		"X-KYC-Webhook-Timestamp": timestamp,
		"X-KYC-Webhook-Signature": signature,
	}
}

func postWebhook(t *testing.T, baseURL string, body []byte, headers map[string]string) *http.Response {
	t.Helper()
	request, err := http.NewRequest(http.MethodPost, baseURL+"/webhooks", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	for key, value := range headers {
		request.Header.Set(key, value)
	}
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	return response
}

func wantWebhookStatus(t *testing.T, response *http.Response, want int) {
	t.Helper()
	defer response.Body.Close()
	if response.StatusCode != want {
		t.Fatalf("webhook response was %d, want %d", response.StatusCode, want)
	}
}

func TestWebhookSignatureValidation(t *testing.T) {
	console, _ := newConsole(t, newFakeKYCAPI(), 0)
	body := webhookBodyBytes(t, "event-signature", testSessionID, 1, "kyc.session.updated")
	now := strconv.FormatInt(time.Now().Unix(), 10)
	stale := strconv.FormatInt(time.Now().Unix()-1000, 10)

	cases := []struct {
		name       string
		headers    map[string]string
		wantStatus int
	}{
		{
			name:       "valid signature",
			headers:    webhookHeaders("event-signature", now, signWebhook(testWebhookSecret, now, body)),
			wantStatus: http.StatusNoContent,
		},
		{
			name:       "wrong secret",
			headers:    webhookHeaders("event-signature", now, signWebhook("wrong-secret-that-is-long-enough-0123456789", now, body)),
			wantStatus: http.StatusUnauthorized,
		},
		{
			name:       "missing signature",
			headers:    webhookHeaders("event-signature", now, ""),
			wantStatus: http.StatusUnauthorized,
		},
		{
			name:       "stale timestamp",
			headers:    webhookHeaders("event-signature", stale, signWebhook(testWebhookSecret, stale, body)),
			wantStatus: http.StatusUnauthorized,
		},
		{
			name:       "tampered body",
			headers:    webhookHeaders("event-signature", now, signWebhook(testWebhookSecret, now, webhookBodyBytes(t, "event-signature", testSessionID, 2, "kyc.session.updated"))),
			wantStatus: http.StatusUnauthorized,
		},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			wantWebhookStatus(t, postWebhook(t, console.URL, body, testCase.headers), testCase.wantStatus)
		})
	}
}

func TestWebhookRejectsInvalidPayloads(t *testing.T) {
	console, _ := newConsole(t, newFakeKYCAPI(), 0)
	now := strconv.FormatInt(time.Now().Unix(), 10)

	cases := []struct {
		name       string
		id         string
		body       []byte
		wantStatus int
	}{
		{
			name:       "unexpected type",
			id:         "event-type",
			body:       webhookBodyBytes(t, "event-type", testSessionID, 1, "kyc.session.other"),
			wantStatus: http.StatusBadRequest,
		},
		{
			name:       "missing session id",
			id:         "event-session",
			body:       webhookBodyBytes(t, "event-session", "", 1, "kyc.session.updated"),
			wantStatus: http.StatusBadRequest,
		},
		{
			name:       "sequence below one",
			id:         "event-sequence",
			body:       webhookBodyBytes(t, "event-sequence", testSessionID, 0, "kyc.session.updated"),
			wantStatus: http.StatusBadRequest,
		},
		{
			name:       "id header mismatch",
			id:         "event-other",
			body:       webhookBodyBytes(t, "event-mismatch", testSessionID, 1, "kyc.session.updated"),
			wantStatus: http.StatusBadRequest,
		},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			headers := webhookHeaders(testCase.id, now, signWebhook(testWebhookSecret, now, testCase.body))
			wantWebhookStatus(t, postWebhook(t, console.URL, testCase.body, headers), testCase.wantStatus)
		})
	}
}

func sseLines(ctx context.Context, reader *bufio.Reader) <-chan string {
	lines := make(chan string, 16)
	go func() {
		defer close(lines)
		for {
			line, err := reader.ReadString('\n')
			if err != nil {
				return
			}
			select {
			case lines <- line:
			case <-ctx.Done():
				return
			}
		}
	}()
	return lines
}

func waitForSSE(lines <-chan string, marker string, timeout time.Duration) (string, bool) {
	var accumulated strings.Builder
	deadline := time.After(timeout)
	for {
		select {
		case <-deadline:
			return accumulated.String(), false
		case line, open := <-lines:
			if !open {
				return accumulated.String(), false
			}
			accumulated.WriteString(line)
			if strings.Contains(accumulated.String(), marker) {
				return accumulated.String(), true
			}
		}
	}
}

func TestWebhookPublishesToBrowserEventStream(t *testing.T) {
	fake := newFakeKYCAPI()
	console, client := newConsole(t, fake, 0)
	_ = responseBody(t, uploadRequest(t, client, console.URL, "front", "front.jpg", "image/jpeg", []byte("front")))

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	streamRequest, err := http.NewRequestWithContext(ctx, http.MethodGet, console.URL+"/events/stream", nil)
	if err != nil {
		t.Fatal(err)
	}
	sseClient := &http.Client{Jar: client.Jar}
	stream, err := sseClient.Do(streamRequest)
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Body.Close()
	if stream.StatusCode != http.StatusOK {
		t.Fatalf("event stream response was %d, want %d", stream.StatusCode, http.StatusOK)
	}
	if contentType := stream.Header.Get("Content-Type"); !strings.HasPrefix(contentType, "text/event-stream") {
		t.Fatalf("event stream content type was %q", contentType)
	}

	lines := sseLines(ctx, bufio.NewReader(stream.Body))
	if _, ok := waitForSSE(lines, ": connected", 3*time.Second); !ok {
		t.Fatal("event stream did not acknowledge the connection")
	}

	body := webhookBodyBytes(t, "event-stream-1", testSessionID, 1, "kyc.session.updated")
	now := strconv.FormatInt(time.Now().Unix(), 10)
	headers := webhookHeaders("event-stream-1", now, signWebhook(testWebhookSecret, now, body))
	wantWebhookStatus(t, postWebhook(t, console.URL, body, headers), http.StatusNoContent)

	received, ok := waitForSSE(lines, "event: kyc.session.updated", 3*time.Second)
	if !ok {
		t.Fatalf("browser stream did not receive the webhook event: %s", received)
	}

	wantWebhookStatus(t, postWebhook(t, console.URL, body, headers), http.StatusNoContent)
	if received, ok := waitForSSE(lines, "event: kyc.session.updated", 750*time.Millisecond); ok {
		t.Fatalf("duplicate webhook id reached the browser stream: %s", received)
	}
}
