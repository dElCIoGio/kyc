package main

import (
	"embed"
	"errors"
	"fmt"
	"html/template"
	"io/fs"
	"net/http"
	"sort"
	"strings"
	"time"
)

//go:embed templates/*.html static/*
var webAssets embed.FS

type server struct {
	config    Config
	client    *kycClient
	sessions  *sessionStore
	templates *template.Template
	events    *eventHub
	deduper   *eventDeduper
}
type alertView struct{ Title, Message string }
type screenView struct {
	Sessions []sessionListView
	Active   *sandboxView
	Error    *alertView
}
type sessionListView struct {
	ID, Status, Created string
	Active              bool
}
type sandboxView struct {
	ID, Status, Created, Expires, NextAction, VerificationURL, CredentialExpires string
	ResultAvailable                                                              bool
	Workflow                                                                     []workflowView
	Milestones                                                                   []milestoneView
	Webhooks                                                                     []webhookView
	Durations                                                                    []durationView
	Result                                                                       *resultView
}
type workflowView struct{ Label, Detail, State string }
type milestoneView struct{ Observed, APICreated, Type, Status, NextAction string }
type webhookView struct{ APICreated, Received, Type, Sequence, Status, NextAction, ResultAvailable string }
type durationView struct{ Label, Value string }

func newServer(config Config, client *kycClient, sessions *sessionStore) (*server, error) {
	templates, err := template.New("web").ParseFS(webAssets, "templates/*.html")
	if err != nil {
		return nil, err
	}
	return &server{config: config, client: client, sessions: sessions, templates: templates, events: newEventHub(), deduper: newEventDeduper()}, nil
}

func (server *server) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /", server.index)
	mux.HandleFunc("GET /healthz", server.health)
	mux.HandleFunc("POST /sandbox/sessions", server.createSession)
	mux.HandleFunc("POST /sandbox/sessions/{sessionID}/select", server.selectSession)
	mux.HandleFunc("GET /sandbox/sessions/{sessionID}/status", server.status)
	mux.HandleFunc("POST /sandbox/sessions/{sessionID}/browser-token", server.rotateBrowserToken)
	mux.HandleFunc("POST /sandbox/sessions/{sessionID}/result", server.fetchResult)
	mux.HandleFunc("POST /sandbox/sessions/{sessionID}/delete", server.deleteSession)
	mux.HandleFunc("POST /webhooks", server.webhook)
	mux.HandleFunc("GET /events/stream", server.streamEvents)
	staticFiles, err := fs.Sub(webAssets, "static")
	if err != nil {
		panic(err)
	}
	mux.Handle("GET /assets/", http.StripPrefix("/assets/", http.FileServerFS(staticFiles)))
	return server.securityHeaders(mux)
}

func (server *server) index(writer http.ResponseWriter, request *http.Request) {
	token, _ := server.visitorFromRequest(request)
	server.renderPage(writer, request, server.screen(token, nil))
}
func (server *server) health(writer http.ResponseWriter, _ *http.Request) {
	writer.Header().Set("Content-Type", "application/json; charset=utf-8")
	_, _ = writer.Write([]byte(`{"status":"ok","service":"kyc-integration-sandbox"}`))
}

func (server *server) createSession(writer http.ResponseWriter, request *http.Request) {
	token, created, err := server.ensureVisitor(writer, request)
	if err != nil {
		server.renderDashboard(writer, request, server.screen(token, &alertView{Title: "Sandbox session could not be created", Message: "Please try again."}))
		return
	}
	upstream, err := server.client.createSession(request.Context())
	if err != nil {
		server.renderDashboard(writer, request, server.screen(token, server.errorAlert("Verification session could not be created", err)))
		return
	}
	observed := time.Now().UTC()
	credential, err := server.client.issueBrowserToken(request.Context(), upstream.SessionID)
	if err != nil {
		server.sessions.addWithoutCredential(token, upstream, observed)
		if created {
			server.setSessionCookie(writer, token)
		}
		server.renderDashboard(writer, request, server.screen(token, server.errorAlert("Browser credential could not be issued", err)))
		return
	}
	if !server.sessions.add(token, upstream, credential, observed) {
		server.renderDashboard(writer, request, server.screen(token, &alertView{Title: "Sandbox state expired", Message: "Create a new verification session."}))
		return
	}
	if created {
		server.setSessionCookie(writer, token)
	}
	server.renderDashboard(writer, request, server.screen(token, nil))
}

func (server *server) selectSession(writer http.ResponseWriter, request *http.Request) {
	token, ok := server.visitorFromRequest(request)
	if !ok || !server.sessions.selectSession(token, request.PathValue("sessionID")) {
		server.renderDashboard(writer, request, server.screen(token, &alertView{Title: "Unknown sandbox session", Message: "Choose one of this browser's tracked sessions."}))
		return
	}
	server.renderDashboard(writer, request, server.screen(token, nil))
}

func (server *server) status(writer http.ResponseWriter, request *http.Request) {
	token, tracked, ok := server.ownedSession(request)
	if !ok {
		server.renderDashboard(writer, request, server.screen(token, &alertView{Title: "Unknown sandbox session", Message: "Choose a tracked session."}))
		return
	}
	upstream, err := server.client.session(request.Context(), tracked.Session.SessionID)
	if err != nil {
		if server.clearIfGone(token, tracked.Session.SessionID, err) {
			server.renderDashboard(writer, request, server.screen(token, &alertView{Title: "This verification session is no longer available", Message: "It expired or was deleted. Create another verification to continue."}))
			return
		}
		server.renderDashboard(writer, request, server.screen(token, server.errorAlert("Safe session state is temporarily unavailable", err)))
		return
	}
	server.sessions.updateSession(token, upstream.SessionID, upstream)
	server.renderDashboard(writer, request, server.screen(token, nil))
}

func (server *server) rotateBrowserToken(writer http.ResponseWriter, request *http.Request) {
	token, tracked, ok := server.ownedSession(request)
	if !ok {
		server.renderDashboard(writer, request, server.screen(token, &alertView{Title: "Unknown sandbox session", Message: "Choose a tracked session."}))
		return
	}
	credential, err := server.client.issueBrowserToken(request.Context(), tracked.Session.SessionID)
	if err != nil {
		server.renderDashboard(writer, request, server.screen(token, server.errorAlert("Browser credential could not be rotated", err)))
		return
	}
	server.sessions.rotateCredential(token, tracked.Session.SessionID, credential, time.Now().UTC())
	server.renderDashboard(writer, request, server.screen(token, nil))
}

func (server *server) fetchResult(writer http.ResponseWriter, request *http.Request) {
	token, tracked, ok := server.ownedSession(request)
	if !ok {
		server.renderDashboard(writer, request, server.screen(token, &alertView{Title: "Unknown sandbox session", Message: "Choose a tracked session."}))
		return
	}
	if !tracked.Session.Document.ResultAvailable {
		server.renderDashboard(writer, request, server.screen(token, &alertView{Title: "Result is not available", Message: "Wait until the API reports result_available before fetching the normalized result."}))
		return
	}
	result, err := server.client.result(request.Context(), tracked.Session.SessionID)
	if err != nil {
		server.renderDashboard(writer, request, server.screen(token, server.errorAlert("Normalized result could not be fetched", err)))
		return
	}
	server.sessions.setResult(token, tracked.Session.SessionID, result, time.Now().UTC())
	server.renderDashboard(writer, request, server.screen(token, nil))
}

func (server *server) deleteSession(writer http.ResponseWriter, request *http.Request) {
	token, tracked, ok := server.ownedSession(request)
	if !ok {
		server.renderDashboard(writer, request, server.screen(token, nil))
		return
	}
	err := server.client.deleteSession(request.Context(), tracked.Session.SessionID)
	var upstream *upstreamError
	if err != nil && (!errors.As(err, &upstream) || upstream.StatusCode != http.StatusNotFound) {
		server.renderDashboard(writer, request, server.screen(token, server.errorAlert("Verification session could not be deleted", err)))
		return
	}
	server.sessions.delete(token, tracked.Session.SessionID)
	server.renderDashboard(writer, request, server.screen(token, nil))
}

func (server *server) ownedSession(request *http.Request) (string, trackedSession, bool) {
	token, ok := server.visitorFromRequest(request)
	if !ok {
		return "", trackedSession{}, false
	}
	session, ok := server.sessions.get(token, request.PathValue("sessionID"))
	return token, session, ok
}
func (server *server) visitorFromRequest(request *http.Request) (string, bool) {
	cookie, err := request.Cookie(sessionCookieName)
	if err != nil || cookie.Value == "" || !server.sessions.hasVisitor(cookie.Value) {
		return "", false
	}
	return cookie.Value, true
}
func (server *server) ensureVisitor(writer http.ResponseWriter, request *http.Request) (string, bool, error) {
	if token, ok := server.visitorFromRequest(request); ok {
		return token, false, nil
	}
	token, err := server.sessions.createVisitor()
	if err != nil {
		return "", false, err
	}
	return token, true, nil
}

func (server *server) setSessionCookie(writer http.ResponseWriter, token string) {
	http.SetCookie(writer, &http.Cookie{Name: sessionCookieName, Value: token, Path: "/", HttpOnly: true, Secure: server.config.CookieSecure, SameSite: http.SameSiteStrictMode})
}
func (server *server) clearIfGone(token, sessionID string, err error) bool {
	var upstream *upstreamError
	if !errors.As(err, &upstream) || (upstream.StatusCode != http.StatusNotFound && upstream.StatusCode != http.StatusGone) {
		return false
	}
	server.sessions.delete(token, sessionID)
	return true
}

func (server *server) screen(token string, alert *alertView) screenView {
	view := screenView{Error: alert}
	for _, tracked := range server.sessions.list(token) {
		view.Sessions = append(view.Sessions, sessionListView{ID: tracked.Session.SessionID, Status: label(tracked.Session.Status), Created: displayTime(tracked.Session.CreatedAt), Active: false})
	}
	tracked, ok := server.sessions.selected(token)
	if !ok {
		return view
	}
	for index := range view.Sessions {
		if view.Sessions[index].ID == tracked.Session.SessionID {
			view.Sessions[index].Active = true
		}
	}
	active := makeSandboxView(tracked)
	view.Active = &active
	return view
}

func makeSandboxView(tracked trackedSession) sandboxView {
	session := tracked.Session
	view := sandboxView{ID: session.SessionID, Status: label(session.Status), Created: displayTime(session.CreatedAt), Expires: displayTime(session.ExpiresAt), NextAction: actionLabel(session.NextAction), VerificationURL: tracked.VerificationURL, CredentialExpires: displayTime(tracked.CredentialExpiresAt), ResultAvailable: session.Document.ResultAvailable, Workflow: workflowFor(session), Durations: durationsFor(session, tracked.Webhooks)}
	milestones := append([]localMilestone(nil), tracked.Milestones...)
	sort.Slice(milestones, func(i, j int) bool { return milestones[i].ObservedAt.Before(milestones[j].ObservedAt) })
	for _, event := range milestones {
		view.Milestones = append(view.Milestones, milestoneView{Observed: displayTime(event.ObservedAt), APICreated: displayTime(event.APICreatedAt), Type: label(event.Type), Status: label(event.Status), NextAction: actionLabel(event.NextAction)})
	}
	for _, event := range tracked.Webhooks {
		available := "false"
		if event.ResultAvailable {
			available = "true"
		}
		view.Webhooks = append(view.Webhooks, webhookView{APICreated: displayTime(event.CreatedAt), Received: displayTime(event.ReceivedAt), Type: event.Type, Sequence: fmt.Sprintf("%d", event.Sequence), Status: label(event.Status), NextAction: actionLabel(event.NextAction), ResultAvailable: available})
	}
	if tracked.Result != nil {
		result := makeResultView(*tracked.Result)
		view.Result = &result
	}
	return view
}

func workflowFor(session upstreamSession) []workflowView {
	front := workflowView{Label: "Document front", State: "pending", Detail: "Waiting for hosted verifier"}
	if session.Document.FrontCapture == "accepted" {
		front.State, front.Detail = "complete", "Accepted"
	} else if actionIs(session.NextAction, "submit_document_front") {
		front.State, front.Detail = "active", "Capture requested"
	}
	back := workflowView{Label: "Document back", State: "pending", Detail: "Waiting for front capture"}
	if session.Document.BackCapture == "accepted" {
		back.State, back.Detail = "complete", "Accepted"
	} else if actionIs(session.NextAction, "submit_document_back") {
		back.State, back.Detail = "active", "Capture requested"
	}
	document := stateWorkflow("Document processing", session.Document.Status, "")
	liveness := stateWorkflow("Liveness", session.Liveness.Status, "not_available")
	if actionIs(session.NextAction, "submit_liveness") && liveness.State == "pending" {
		liveness.State, liveness.Detail = "active", "Submission requested"
	}
	face := stateWorkflow("Face comparison", session.FaceComparison.Status, "not_available")
	terminal := workflowView{Label: "Terminal status", State: "pending", Detail: "Verification remains in progress"}
	if session.Status == "completed" {
		terminal.State, terminal.Detail = "complete", "Verification processing completed"
	} else if session.Status == "failed" {
		terminal.State, terminal.Detail = "failed", "Verification processing failed"
	}
	return []workflowView{{Label: "Session created", State: "complete", Detail: "API session created"}, front, back, document, liveness, face, terminal}
}
func stateWorkflow(name, status, unavailable string) workflowView {
	view := workflowView{Label: name, State: "pending", Detail: label(status)}
	switch status {
	case "completed", "passed":
		view.State = "complete"
	case "failed":
		view.State = "failed"
	case "processing":
		view.State = "active"
	case "awaiting_capture":
		view.Detail = "Waiting for capture"
	}
	if status != "" && status == unavailable {
		view.State, view.Detail = "unavailable", "Not configured"
	}
	return view
}
func durationsFor(session upstreamSession, events []receivedWebhook) []durationView {
	var result []durationView
	document := firstEvent(events, "verification.document.completed", "verification.document.failed")
	liveness := firstEvent(events, "verification.liveness.passed", "verification.liveness.failed")
	processing := firstEvent(events, "verification.processing.completed", "verification.processing.failed")
	if document != nil {
		result = append(result, durationView{"Session creation → document terminal", durationBetween(session.CreatedAt, document.CreatedAt)})
	}
	if document != nil && liveness != nil {
		result = append(result, durationView{"Document terminal → liveness terminal", durationBetween(document.CreatedAt, liveness.CreatedAt)})
	}
	if liveness != nil && processing != nil {
		result = append(result, durationView{"Liveness terminal → processing terminal", durationBetween(liveness.CreatedAt, processing.CreatedAt)})
	}
	if processing != nil {
		result = append(result, durationView{"Session creation → processing terminal", durationBetween(session.CreatedAt, processing.CreatedAt)}, durationView{"Processing event → Go receipt", durationBetween(processing.CreatedAt, processing.ReceivedAt)})
	}
	return result
}
func firstEvent(events []receivedWebhook, types ...string) *receivedWebhook {
	for _, event := range events {
		for _, wanted := range types {
			if event.Type == wanted {
				copy := event
				return &copy
			}
		}
	}
	return nil
}
func durationBetween(start, end time.Time) string {
	if start.IsZero() || end.IsZero() {
		return "—"
	}
	duration := end.Sub(start)
	if duration < 0 {
		return "Clock offset"
	}
	return duration.Round(time.Millisecond).String()
}
func actionIs(action *string, wanted string) bool { return action != nil && *action == wanted }
func actionLabel(action *string) string {
	if action == nil {
		return "None"
	}
	return label(*action)
}
func label(value string) string {
	if value == "" {
		return "—"
	}
	words := strings.Fields(strings.ReplaceAll(strings.ReplaceAll(value, ".", " "), "_", " "))
	for index, word := range words {
		words[index] = strings.ToUpper(word[:1]) + strings.ToLower(word[1:])
	}
	return strings.Join(words, " ")
}
func displayTime(value time.Time) string {
	if value.IsZero() {
		return "—"
	}
	return value.UTC().Format("2006-01-02 15:04:05 UTC")
}

func (server *server) errorAlert(title string, err error) *alertView {
	var upstream *upstreamError
	if errors.As(err, &upstream) {
		message := strings.TrimSpace(upstream.Message)
		if message == "" {
			message = "The KYC service could not complete this request."
		}
		if upstream.RetryAfter > 0 {
			message += fmt.Sprintf(" Try again in %d seconds.", int(upstream.RetryAfter.Seconds()))
		}
		return &alertView{Title: title, Message: message}
	}
	return &alertView{Title: title, Message: "The KYC service is temporarily unavailable. Please try again."}
}
func (server *server) renderPage(writer http.ResponseWriter, request *http.Request, view screenView) {
	server.htmlHeaders(writer)
	_ = server.templates.ExecuteTemplate(writer, "page", view)
}
func (server *server) renderDashboard(writer http.ResponseWriter, request *http.Request, view screenView) {
	server.htmlHeaders(writer)
	if !isHTMXRequest(request) {
		_ = server.templates.ExecuteTemplate(writer, "page", view)
		return
	}
	_ = server.templates.ExecuteTemplate(writer, "dashboard", view)
}
func (server *server) htmlHeaders(writer http.ResponseWriter) {
	writer.Header().Set("Content-Type", "text/html; charset=utf-8")
	writer.Header().Set("Cache-Control", "no-store")
}
func isHTMXRequest(request *http.Request) bool { return request.Header.Get("HX-Request") == "true" }
func (server *server) securityHeaders(next http.Handler) http.Handler {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set("Content-Security-Policy", "default-src 'self'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'; img-src 'self' data:; style-src 'self'; script-src 'self'")
		writer.Header().Set("Referrer-Policy", "no-referrer")
		writer.Header().Set("X-Content-Type-Options", "nosniff")
		writer.Header().Set("X-Frame-Options", "DENY")
		next.ServeHTTP(writer, request)
	})
}
