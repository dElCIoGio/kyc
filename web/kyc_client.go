package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"net/url"
	"path"
	"strconv"
	"strings"
	"time"
)

type kycClient struct {
	baseURL string
	apiKey  string
	http    *http.Client
}

func textPartHeader(fieldName string, filename string, contentType string) textproto.MIMEHeader {
	header := make(textproto.MIMEHeader)
	header.Set("Content-Disposition", fmt.Sprintf(`form-data; name=%q; filename=%q`, fieldName, filename))
	header.Set("Content-Type", contentType)
	return header
}

type upstreamSession struct {
	SessionID       string    `json:"session_id"`
	Status          string    `json:"status"`
	CreatedAt       time.Time `json:"created_at"`
	ExpiresAt       time.Time `json:"expires_at"`
	JobID           string    `json:"job_id"`
	UploadedSides   []string  `json:"uploaded_sides"`
	ResultAvailable bool      `json:"result_available"`
}

type uploadResponse struct {
	SessionID     string    `json:"session_id"`
	Status        string    `json:"status"`
	Side          string    `json:"side"`
	UploadedSides []string  `json:"uploaded_sides"`
	ExpiresAt     time.Time `json:"expires_at"`
}

type processResponse struct {
	SessionID string `json:"session_id"`
	JobID     string `json:"job_id"`
	Status    string `json:"status"`
}

type upstreamError struct {
	StatusCode int
	Code       string
	Message    string
	RetryAfter time.Duration
}

func (err *upstreamError) Error() string {
	if err.Code != "" {
		return err.Code
	}
	return fmt.Sprintf("upstream request failed with status %d", err.StatusCode)
}

func newKYCClient(config Config, httpClient *http.Client) *kycClient {
	return &kycClient{
		baseURL: strings.TrimRight(config.APIURL.String(), "/"),
		apiKey:  config.APIKey,
		http:    httpClient,
	}
}

func (client *kycClient) createSession(ctx context.Context) (upstreamSession, error) {
	var response upstreamSession
	return response, client.doJSON(ctx, http.MethodPost, "/v1/sessions", nil, nil, &response)
}

func (client *kycClient) session(ctx context.Context, sessionID string) (upstreamSession, error) {
	var response upstreamSession
	return response, client.doJSON(ctx, http.MethodGet, "/v1/sessions/"+url.PathEscape(sessionID), nil, nil, &response)
}

func (client *kycClient) upload(
	ctx context.Context,
	sessionID string,
	side string,
	contentType string,
	content []byte,
) (uploadResponse, error) {
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	extension := ".bin"
	if contentType == "image/jpeg" {
		extension = ".jpg"
	} else if contentType == "image/png" {
		extension = ".png"
	}
	part, err := writer.CreatePart(textPartHeader("image", side+extension, contentType))
	if err != nil {
		return uploadResponse{}, err
	}
	if _, err := part.Write(content); err != nil {
		return uploadResponse{}, err
	}
	if err := writer.Close(); err != nil {
		return uploadResponse{}, err
	}

	var response uploadResponse
	err = client.doJSON(
		ctx,
		http.MethodPost,
		"/v1/sessions/"+url.PathEscape(sessionID)+"/images/"+side,
		&body,
		map[string]string{"Content-Type": writer.FormDataContentType()},
		&response,
	)
	return response, err
}

func (client *kycClient) process(ctx context.Context, sessionID string) (processResponse, error) {
	var response processResponse
	return response, client.doJSON(
		ctx,
		http.MethodPost,
		"/v1/sessions/"+url.PathEscape(sessionID)+"/process",
		nil,
		nil,
		&response,
	)
}

func (client *kycClient) result(ctx context.Context, sessionID string) (documentResult, error) {
	var response documentResult
	return response, client.doJSON(
		ctx,
		http.MethodGet,
		"/v1/sessions/"+url.PathEscape(sessionID)+"/result",
		nil,
		nil,
		&response,
	)
}

func (client *kycClient) deleteSession(ctx context.Context, sessionID string) error {
	return client.doJSON(
		ctx,
		http.MethodDelete,
		"/v1/sessions/"+url.PathEscape(sessionID),
		nil,
		nil,
		nil,
	)
}

func (client *kycClient) doJSON(
	ctx context.Context,
	method string,
	endpoint string,
	body io.Reader,
	headers map[string]string,
	target any,
) error {
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
	return &upstreamError{
		StatusCode: response.StatusCode,
		Code:       payload.Error.Code,
		Message:    payload.Error.Message,
		RetryAfter: retryAfter,
	}
}
