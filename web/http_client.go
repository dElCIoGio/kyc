package main

import (
	"net/http"
	"time"
)

// httpClient keeps the standard client construction small and replaceable in tests.
type httpClient struct {
	timeout time.Duration
}

func (client *httpClient) standard() *http.Client {
	return &http.Client{Timeout: client.timeout}
}
