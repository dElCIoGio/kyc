package main

import (
	"fmt"
	"net/url"
	"strconv"
	"strings"
	"time"
)

const defaultMaxUploadBytes int64 = 15 * 1024 * 1024

// Config contains the only settings needed by the browser-facing service.
// KYC_API_KEY is intentionally read here and never written into a template,
// response, or client-side asset.
type Config struct {
	Address        string
	APIURL         *url.URL
	APIKey         string
	MaxUploadBytes int64
	CookieSecure   bool
	WebhookSecret  string
}

func loadConfig(lookup func(string) (string, bool)) (Config, error) {
	address := valueOr(lookup, "KYC_WEB_ADDRESS", ":8080")
	apiURLText := valueOr(lookup, "KYC_WEB_API_URL", "http://api:8000")
	apiURL, err := url.Parse(apiURLText)
	if err != nil || apiURL.Scheme == "" || apiURL.Host == "" || (apiURL.Scheme != "http" && apiURL.Scheme != "https") {
		return Config{}, fmt.Errorf("KYC_WEB_API_URL must be an absolute HTTP(S) URL")
	}
	if apiURL.User != nil || apiURL.RawQuery != "" || apiURL.Fragment != "" {
		return Config{}, fmt.Errorf("KYC_WEB_API_URL must not contain credentials, a query, or a fragment")
	}

	apiKey, ok := lookup("KYC_API_KEY")
	if !ok || strings.TrimSpace(apiKey) == "" {
		return Config{}, fmt.Errorf("KYC_API_KEY is required")
	}
	webhookSecret, ok := lookup("KYC_WEBHOOK_SECRET")
	if !ok || len(webhookSecret) < 32 {
		return Config{}, fmt.Errorf("KYC_WEBHOOK_SECRET must contain at least 32 characters")
	}

	maxUploadBytes := defaultMaxUploadBytes
	if raw, ok := lookup("KYC_WEB_MAX_UPLOAD_BYTES"); ok && strings.TrimSpace(raw) != "" {
		maxUploadBytes, err = strconv.ParseInt(raw, 10, 64)
		if err != nil || maxUploadBytes <= 0 {
			return Config{}, fmt.Errorf("KYC_WEB_MAX_UPLOAD_BYTES must be a positive integer")
		}
	}

	cookieSecure := false
	if raw, ok := lookup("KYC_WEB_COOKIE_SECURE"); ok && strings.TrimSpace(raw) != "" {
		cookieSecure, err = strconv.ParseBool(raw)
		if err != nil {
			return Config{}, fmt.Errorf("KYC_WEB_COOKIE_SECURE must be true or false")
		}
	}

	return Config{
		Address:        address,
		APIURL:         apiURL,
		APIKey:         apiKey,
		MaxUploadBytes: maxUploadBytes,
		CookieSecure:   cookieSecure,
		WebhookSecret:  webhookSecret,
	}, nil
}

func valueOr(lookup func(string) (string, bool), key, fallback string) string {
	if value, ok := lookup(key); ok && strings.TrimSpace(value) != "" {
		return value
	}
	return fallback
}

func newHTTPClient() *httpClient {
	return &httpClient{timeout: 45 * time.Second}
}
