package main

import (
	"crypto/rand"
	"encoding/base64"
	"sort"
	"sync"
	"time"
)

const sessionCookieName = "kyc_console"

// localMilestone is one sandbox-side timing observation. ObservedAt is always
// set locally; APICreatedAt is populated only when the milestone mirrors an API
// event so the dashboard can keep the two clocks visibly separate.
type localMilestone struct {
	Type         string
	ObservedAt   time.Time
	APICreatedAt time.Time
	Status       string
	NextAction   *string
}
type receivedWebhook struct {
	ID, Type, SessionID, Status string
	CreatedAt, ReceivedAt       time.Time
	Sequence                    int
	NextAction                  *string
	ResultAvailable             bool
}
type trackedSession struct {
	Session             upstreamSession
	VerificationURL     string
	CredentialExpiresAt time.Time
	Milestones          []localMilestone
	Webhooks            []receivedWebhook
	Result              *normalizedResult
	ResultFetchedAt     time.Time
}
type visitorSessions struct {
	Selected string
	Sessions map[string]trackedSession
}

// terminalWebhookMilestones maps each terminal lifecycle webhook to the local
// timing milestone recorded when this service observes it. Non-terminal events
// such as verification.session.created are already covered by the locally
// issued session and browser-credential milestones.
var terminalWebhookMilestones = map[string]string{
	"verification.document.completed":   "document.terminal",
	"verification.document.failed":      "document.terminal",
	"verification.liveness.passed":      "liveness.terminal",
	"verification.liveness.failed":      "liveness.terminal",
	"verification.processing.completed": "processing.terminal",
	"verification.processing.failed":    "processing.terminal",
}

// sessionStore is an intentionally process-local sandbox ledger. It contains
// no uploads or artifacts and is scoped by an opaque HttpOnly browser cookie.
type sessionStore struct {
	mu       sync.Mutex
	visitors map[string]visitorSessions
}

func newSessionStore() *sessionStore {
	return &sessionStore{visitors: make(map[string]visitorSessions)}
}

func randomToken() (string, error) {
	bytes := make([]byte, 32)
	if _, err := rand.Read(bytes); err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(bytes), nil
}
func (store *sessionStore) createVisitor() (string, error) {
	token, err := randomToken()
	if err != nil {
		return "", err
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	store.visitors[token] = visitorSessions{Sessions: make(map[string]trackedSession)}
	return token, nil
}
func (store *sessionStore) add(token string, session upstreamSession, credential browserTokenResponse, issuedAt time.Time) bool {
	store.mu.Lock()
	defer store.mu.Unlock()
	store.purgeLocked(time.Now())
	visitor, ok := store.visitors[token]
	if !ok {
		return false
	}
	visitor.Sessions[session.SessionID] = trackedSession{Session: session, VerificationURL: credential.VerificationURL, CredentialExpiresAt: credential.ExpiresAt, Milestones: []localMilestone{{Type: "session.created", ObservedAt: issuedAt, Status: session.Status, NextAction: session.NextAction}, {Type: "browser_credential.issued", ObservedAt: issuedAt, Status: session.Status, NextAction: session.NextAction}}}
	visitor.Selected = session.SessionID
	store.visitors[token] = visitor
	return true
}
func (store *sessionStore) addWithoutCredential(token string, session upstreamSession, observedAt time.Time) bool {
	store.mu.Lock()
	defer store.mu.Unlock()
	visitor, ok := store.visitors[token]
	if !ok {
		return false
	}
	visitor.Sessions[session.SessionID] = trackedSession{Session: session, Milestones: []localMilestone{{Type: "session.created", ObservedAt: observedAt, Status: session.Status, NextAction: session.NextAction}}}
	visitor.Selected = session.SessionID
	store.visitors[token] = visitor
	return true
}
func (store *sessionStore) get(token, sessionID string) (trackedSession, bool) {
	store.mu.Lock()
	defer store.mu.Unlock()
	store.purgeLocked(time.Now())
	visitor, ok := store.visitors[token]
	if !ok {
		return trackedSession{}, false
	}
	session, ok := visitor.Sessions[sessionID]
	return cloneTracked(session), ok
}
func (store *sessionStore) selected(token string) (trackedSession, bool) {
	store.mu.Lock()
	defer store.mu.Unlock()
	store.purgeLocked(time.Now())
	visitor, ok := store.visitors[token]
	if !ok || visitor.Selected == "" {
		return trackedSession{}, false
	}
	session, ok := visitor.Sessions[visitor.Selected]
	return cloneTracked(session), ok
}
func (store *sessionStore) list(token string) []trackedSession {
	store.mu.Lock()
	defer store.mu.Unlock()
	store.purgeLocked(time.Now())
	visitor, ok := store.visitors[token]
	if !ok {
		return nil
	}
	result := make([]trackedSession, 0, len(visitor.Sessions))
	for _, session := range visitor.Sessions {
		result = append(result, cloneTracked(session))
	}
	sort.Slice(result, func(i, j int) bool { return result[i].Session.CreatedAt.After(result[j].Session.CreatedAt) })
	return result
}
func (store *sessionStore) selectSession(token, sessionID string) bool {
	store.mu.Lock()
	defer store.mu.Unlock()
	visitor, ok := store.visitors[token]
	if !ok {
		return false
	}
	if _, ok := visitor.Sessions[sessionID]; !ok {
		return false
	}
	visitor.Selected = sessionID
	store.visitors[token] = visitor
	return true
}
func (store *sessionStore) updateSession(token, sessionID string, session upstreamSession) bool {
	store.mu.Lock()
	defer store.mu.Unlock()
	visitor, ok := store.visitors[token]
	if !ok {
		return false
	}
	tracked, ok := visitor.Sessions[sessionID]
	if !ok {
		return false
	}
	tracked.Session = session
	visitor.Sessions[sessionID] = tracked
	store.visitors[token] = visitor
	return true
}
func (store *sessionStore) rotateCredential(token, sessionID string, credential browserTokenResponse, observedAt time.Time) bool {
	store.mu.Lock()
	defer store.mu.Unlock()
	visitor, ok := store.visitors[token]
	if !ok {
		return false
	}
	tracked, ok := visitor.Sessions[sessionID]
	if !ok {
		return false
	}
	tracked.VerificationURL = credential.VerificationURL
	tracked.CredentialExpiresAt = credential.ExpiresAt
	tracked.Milestones = append(tracked.Milestones, localMilestone{Type: "browser_credential.rotated", ObservedAt: observedAt, Status: tracked.Session.Status, NextAction: tracked.Session.NextAction})
	visitor.Sessions[sessionID] = tracked
	store.visitors[token] = visitor
	return true
}
func (store *sessionStore) setResult(token, sessionID string, result normalizedResult, observedAt time.Time) bool {
	store.mu.Lock()
	defer store.mu.Unlock()
	visitor, ok := store.visitors[token]
	if !ok {
		return false
	}
	tracked, ok := visitor.Sessions[sessionID]
	if !ok {
		return false
	}
	tracked.Result = &result
	tracked.ResultFetchedAt = observedAt
	tracked.Milestones = append(tracked.Milestones, localMilestone{Type: "result.fetched", ObservedAt: observedAt, Status: tracked.Session.Status, NextAction: tracked.Session.NextAction})
	visitor.Sessions[sessionID] = tracked
	store.visitors[token] = visitor
	return true
}
func (store *sessionStore) delete(token, sessionID string) bool {
	store.mu.Lock()
	defer store.mu.Unlock()
	visitor, ok := store.visitors[token]
	if !ok {
		return false
	}
	if _, ok := visitor.Sessions[sessionID]; !ok {
		return false
	}
	delete(visitor.Sessions, sessionID)
	if visitor.Selected == sessionID {
		visitor.Selected = ""
		for id := range visitor.Sessions {
			visitor.Selected = id
			break
		}
	}
	store.visitors[token] = visitor
	return true
}
func (store *sessionStore) recordWebhook(event receivedWebhook) []string {
	store.mu.Lock()
	defer store.mu.Unlock()
	store.purgeLocked(time.Now())
	var visitors []string
	for token, visitor := range store.visitors {
		tracked, ok := visitor.Sessions[event.SessionID]
		if !ok {
			continue
		}
		tracked.Webhooks = append(tracked.Webhooks, event)
		sort.Slice(tracked.Webhooks, func(i, j int) bool {
			if tracked.Webhooks[i].Sequence != tracked.Webhooks[j].Sequence {
				return tracked.Webhooks[i].Sequence < tracked.Webhooks[j].Sequence
			}
			if tracked.Webhooks[i].Type != tracked.Webhooks[j].Type {
				return tracked.Webhooks[i].Type < tracked.Webhooks[j].Type
			}
			return tracked.Webhooks[i].ID < tracked.Webhooks[j].ID
		})
		if milestone, ok := terminalWebhookMilestones[event.Type]; ok {
			tracked.Milestones = append(tracked.Milestones, localMilestone{Type: milestone, ObservedAt: event.ReceivedAt, APICreatedAt: event.CreatedAt, Status: event.Status, NextAction: event.NextAction})
		}
		visitor.Sessions[event.SessionID] = tracked
		store.visitors[token] = visitor
		visitors = append(visitors, token)
	}
	return visitors
}
func (store *sessionStore) hasVisitor(token string) bool {
	store.mu.Lock()
	defer store.mu.Unlock()
	store.purgeLocked(time.Now())
	_, ok := store.visitors[token]
	return ok
}
func (store *sessionStore) purgeLocked(now time.Time) {
	for token, visitor := range store.visitors {
		for id, session := range visitor.Sessions {
			if !session.Session.ExpiresAt.IsZero() && !session.Session.ExpiresAt.After(now) {
				delete(visitor.Sessions, id)
				if visitor.Selected == id {
					visitor.Selected = ""
				}
			}
		}
		store.visitors[token] = visitor
	}
}
func cloneTracked(value trackedSession) trackedSession {
	value.Milestones = append([]localMilestone(nil), value.Milestones...)
	value.Webhooks = append([]receivedWebhook(nil), value.Webhooks...)
	return value
}
