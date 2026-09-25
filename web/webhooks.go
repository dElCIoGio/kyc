package main

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"strconv"
	"time"
)

const webhookBodyLimit = 64 << 10

type inboundWebhook struct {
	Version string `json:"version"`
	ID string `json:"id"`
	Type string `json:"type"`
	Data struct {
		SessionID string `json:"session_id"`
		Sequence int `json:"sequence"`
		Status string `json:"status"`
	} `json:"data"`
}

func (server *server) webhook(writer http.ResponseWriter, request *http.Request) {
	request.Body = http.MaxBytesReader(writer, request.Body, webhookBodyLimit)
	body, err := io.ReadAll(request.Body)
	if err != nil || len(body) == 0 { http.Error(writer, "invalid webhook", http.StatusBadRequest); return }
	id := request.Header.Get("X-KYC-Webhook-ID")
	timestamp := request.Header.Get("X-KYC-Webhook-Timestamp")
	signature := request.Header.Get("X-KYC-Webhook-Signature")
	seconds, parseErr := strconv.ParseInt(timestamp, 10, 64)
	if id == "" || parseErr != nil || abs(time.Now().Unix()-seconds) > 300 || !server.verifyWebhook(timestamp, body, signature) {
		http.Error(writer, "unauthorized", http.StatusUnauthorized); return
	}
	var event inboundWebhook
	if json.Unmarshal(body, &event) != nil || event.Version != "1" || event.Type != "kyc.session.updated" || event.ID != id || event.Data.SessionID == "" || event.Data.Sequence < 1 {
		http.Error(writer, "invalid webhook", http.StatusBadRequest); return
	}
	if !server.deduper.first(id) { writer.WriteHeader(http.StatusNoContent); return }
	for _, token := range server.sessions.tokensForUpstream(event.Data.SessionID) { server.events.publish(token) }
	writer.WriteHeader(http.StatusNoContent)
}

func (server *server) verifyWebhook(timestamp string, body []byte, signature string) bool {
	const prefix = "v1="
	if len(signature) != len(prefix)+64 || signature[:len(prefix)] != prefix { return false }
	expected := hmac.New(sha256.New, []byte(server.config.WebhookSecret))
	_, _ = expected.Write([]byte(timestamp))
	_, _ = expected.Write([]byte("."))
	_, _ = expected.Write(body)
	provided, err := hex.DecodeString(signature[len(prefix):])
	return err == nil && hmac.Equal(expected.Sum(nil), provided)
}

func (server *server) streamEvents(writer http.ResponseWriter, request *http.Request) {
	token, _, ok := server.sessionFromRequest(request)
	if !ok { writer.WriteHeader(http.StatusNoContent); return }
	flusher, ok := writer.(http.Flusher)
	if !ok { http.Error(writer, "streaming unavailable", http.StatusInternalServerError); return }
	writer.Header().Set("Content-Type", "text/event-stream")
	writer.Header().Set("Cache-Control", "no-cache")
	writer.Header().Set("X-Accel-Buffering", "no")
	channel, cancel := server.events.subscribe(token)
	defer cancel()
	_, _ = writer.Write([]byte(": connected\n\n")); flusher.Flush()
	heartbeat := time.NewTicker(25 * time.Second); defer heartbeat.Stop()
	for {
		select {
		case <-request.Context().Done(): return
		case <-channel:
			_, _ = writer.Write([]byte("event: kyc.session.updated\ndata: {}\n\n")); flusher.Flush()
		case <-heartbeat.C:
			_, _ = writer.Write([]byte(": keepalive\n\n")); flusher.Flush()
		}
	}
}

func abs(value int64) int64 { if value < 0 { return -value }; return value }
