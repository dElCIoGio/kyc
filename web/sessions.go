package main

import (
	"crypto/rand"
	"encoding/base64"
	"sync"
	"time"
)

const sessionCookieName = "kyc_console"

type webSession struct {
	upstreamID string
	expiresAt  time.Time
}

// sessionStore deliberately keeps only opaque upstream identifiers and expiry
// metadata in process memory. Document images and results are never stored here.
type sessionStore struct {
	mu       sync.Mutex
	sessions map[string]webSession
}

func newSessionStore() *sessionStore {
	return &sessionStore{sessions: make(map[string]webSession)}
}

func (store *sessionStore) get(token string) (webSession, bool) {
	store.mu.Lock()
	defer store.mu.Unlock()
	store.purgeLocked(time.Now())
	session, ok := store.sessions[token]
	return session, ok
}

func (store *sessionStore) create(upstreamID string, expiresAt time.Time) (string, error) {
	bytes := make([]byte, 32)
	if _, err := rand.Read(bytes); err != nil {
		return "", err
	}
	token := base64.RawURLEncoding.EncodeToString(bytes)
	store.mu.Lock()
	defer store.mu.Unlock()
	store.purgeLocked(time.Now())
	store.sessions[token] = webSession{upstreamID: upstreamID, expiresAt: expiresAt}
	return token, nil
}

func (store *sessionStore) update(token string, expiresAt time.Time) {
	store.mu.Lock()
	defer store.mu.Unlock()
	if session, ok := store.sessions[token]; ok {
		session.expiresAt = expiresAt
		store.sessions[token] = session
	}
}

func (store *sessionStore) delete(token string) {
	store.mu.Lock()
	defer store.mu.Unlock()
	delete(store.sessions, token)
}

func (store *sessionStore) tokensForUpstream(upstreamID string) []string {
	store.mu.Lock(); defer store.mu.Unlock()
	store.purgeLocked(time.Now())
	var tokens []string
	for token, session := range store.sessions { if session.upstreamID == upstreamID { tokens = append(tokens, token) } }
	return tokens
}

func (store *sessionStore) purgeLocked(now time.Time) {
	for token, session := range store.sessions {
		if !session.expiresAt.IsZero() && !session.expiresAt.After(now) {
			delete(store.sessions, token)
		}
	}
}
