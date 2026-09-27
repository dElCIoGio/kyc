package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"path"
	"strconv"
	"strings"
	"time"
)

// upstreamSession deliberately models only the stable, safe public session
// projection. The integration sandbox never asks the API for images or engine
// diagnostics.
type upstreamSession struct {
	SessionID      string        `json:"session_id"`
	Status         string        `json:"status"`
	CreatedAt      time.Time     `json:"created_at"`
	ExpiresAt      time.Time     `json:"expires_at"`
	NextAction     *string       `json:"next_action"`
	Document       documentState `json:"document"`
	Liveness       checkState    `json:"liveness"`
	FaceComparison checkState    `json:"face_comparison"`
}

type documentState struct {
	Status          string `json:"status"`
	FrontCapture    string `json:"front_capture"`
	BackCapture     string `json:"back_capture"`
	ResultAvailable bool   `json:"result_available"`
}

type checkState struct {
	Status string `json:"status"`
}

type browserTokenResponse struct {
	VerificationURL string    `json:"verification_url"`
	ExpiresAt       time.Time `json:"expires_at"`
}

type kycClient struct {
	baseURL, apiKey string
	http            *http.Client
}

type upstreamError struct {
	StatusCode    int
	Code, Message string
	RetryAfter    time.Duration
}

func (err *upstreamError) Error() string {
	if err.Code != "" {
		return err.Code
	}
	return fmt.Sprintf("upstream request failed with status %d", err.StatusCode)
}

func newKYCClient(config Config, httpClient *http.Client) *kycClient {
	return &kycClient{baseURL: strings.TrimRight(config.APIURL.String(), "/"), apiKey: config.APIKey, http: httpClient}
}

func (client *kycClient) createSession(ctx context.Context) (upstreamSession, error) {
	var response upstreamSession
	return response, client.doJSON(ctx, http.MethodPost, "/v1/sessions", nil, nil, &response)
}

func (client *kycClient) session(ctx context.Context, sessionID string) (upstreamSession, error) {
	var response upstreamSession
	return response, client.doJSON(ctx, http.MethodGet, "/v1/sessions/"+url.PathEscape(sessionID), nil, nil, &response)
}

func (client *kycClient) issueBrowserToken(ctx context.Context, sessionID string) (browserTokenResponse, error) {
	var response browserTokenResponse
	return response, client.doJSON(ctx, http.MethodPost, "/v1/sessions/"+url.PathEscape(sessionID)+"/browser-token", nil, nil, &response)
}

func (client *kycClient) result(ctx context.Context, sessionID string) (normalizedResult, error) {
	var response normalizedResult
	return response, client.doJSON(ctx, http.MethodGet, "/v1/sessions/"+url.PathEscape(sessionID)+"/result", nil, nil, &response)
}

func (client *kycClient) deleteSession(ctx context.Context, sessionID string) error {
	return client.doJSON(ctx, http.MethodDelete, "/v1/sessions/"+url.PathEscape(sessionID), nil, nil, nil)
}

func (client *kycClient) doJSON(ctx context.Context, method, endpoint string, body io.Reader, headers map[string]string, target any) error {
	request, err := http.NewRequestWithContext(ctx, method, client.baseURL+path.Clean("/"+endpoint), body)
	if err != nil {
		return err
	}
	request.Header.Set("Accept", "application/json")
	request.Header.Set("X-API-Key", client.apiKey)
	for key, value := range headers {
		request.Header.Set(key, value)
	}
	response, err := client.http.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return decodeUpstreamError(response)
	}
	if target == nil {
		return nil
	}
	if err := json.NewDecoder(io.LimitReader(response.Body, 4<<20)).Decode(target); err != nil {
		return fmt.Errorf("decode KYC API response: %w", err)
	}
	return nil
}

func decodeUpstreamError(response *http.Response) *upstreamError {
	var payload struct {
		Error struct {
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	_ = json.NewDecoder(io.LimitReader(response.Body, 64<<10)).Decode(&payload)
	retryAfter := time.Duration(0)
	if seconds, err := strconv.Atoi(response.Header.Get("Retry-After")); err == nil && seconds > 0 {
		retryAfter = time.Duration(seconds) * time.Second
	}
	return &upstreamError{StatusCode: response.StatusCode, Code: payload.Error.Code, Message: payload.Error.Message, RetryAfter: retryAfter}
}
