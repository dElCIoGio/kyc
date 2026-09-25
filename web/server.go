package main

import (
	"embed"
	"errors"
	"fmt"
	"html/template"
	"io"
	"io/fs"
	"net/http"
	"strings"
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

type screenView struct {
	Status        lifecycleView
	MaxUpload     string
	HasSession    bool
	FrontUploaded bool
	BackUploaded  bool
	CanUpload     bool
	CanProcess    bool
	CanReset      bool
	Error         *alertView
	Result        *resultView
}

type lifecycleView struct {
	Label       string
	Description string
	Tone        string
	Busy        bool
	Poll        bool
}

type alertView struct {
	Title   string
	Message string
}

func newServer(config Config, client *kycClient, sessions *sessionStore) (*server, error) {
	templates, err := template.New("web").Funcs(template.FuncMap{
		"dict": func(values ...any) (map[string]any, error) {
			if len(values)%2 != 0 {
				return nil, fmt.Errorf("dict requires key/value pairs")
			}
			result := make(map[string]any, len(values)/2)
			for index := 0; index < len(values); index += 2 {
				key, ok := values[index].(string)
				if !ok {
					return nil, fmt.Errorf("dict keys must be strings")
				}
				result[key] = values[index+1]
			}
			return result, nil
		},
	}).ParseFS(webAssets, "templates/*.html")
	if err != nil {
		return nil, err
	}
	return &server{
		config:    config,
		client:    client,
		sessions:  sessions,
		templates: templates,
		events:    newEventHub(),
		deduper:   newEventDeduper(),
	}, nil
}

func (server *server) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /", server.index)
	mux.HandleFunc("GET /healthz", server.health)
	mux.HandleFunc("POST /checks/images/{side}", server.upload)
	mux.HandleFunc("POST /checks/process", server.process)
	mux.HandleFunc("GET /checks/status", server.status)
	mux.HandleFunc("POST /checks/reset", server.reset)
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
	if _, _, exists := server.sessionFromRequest(request); exists {
		server.renderPage(writer, request, screenView{
			Status: lifecycleView{
				Label:       "Restoring your check",
				Description: "Checking the latest KYC verification status.",
				Tone:        "neutral",
				Busy:        true,
				Poll:        true,
			},
			MaxUpload:  formatBytes(server.config.MaxUploadBytes),
			HasSession: true,
			CanReset:   true,
		})
		return
	}
	server.renderPage(writer, request, server.emptyScreen(nil))
}

func (server *server) health(writer http.ResponseWriter, _ *http.Request) {
	writer.Header().Set("Content-Type", "application/json; charset=utf-8")
	_, _ = writer.Write([]byte(`{"status":"ok","service":"kyc-operator-console"}`))
}

func (server *server) upload(writer http.ResponseWriter, request *http.Request) {
	side := request.PathValue("side")
	if side != "front" && side != "back" {
		server.renderDashboard(writer, request, server.emptyScreen(&alertView{
			Title:   "Unsupported document side",
			Message: "Choose either the front or back document upload area.",
		}))
		return
	}
	contentType, content, err := server.readImage(request, writer)
	if err != nil {
		server.renderDashboard(writer, request, server.withCurrentError(request, "Upload could not start", err))
		return
	}

	token, session, exists := server.sessionFromRequest(request)
	if !exists {
		created, err := server.client.createSession(request.Context())
		if err != nil {
			server.renderDashboard(writer, request, server.emptyScreen(server.errorAlert("A KYC session could not be created", err)))
			return
		}
		token, err = server.sessions.create(created.SessionID, created.ExpiresAt)
		if err != nil {
			server.renderDashboard(writer, request, server.emptyScreen(&alertView{
				Title:   "A secure browser session could not be created",
				Message: "Please try the upload again.",
			}))
			return
		}
		session = webSession{upstreamID: created.SessionID, expiresAt: created.ExpiresAt}
		server.setSessionCookie(writer, token)
	}

	uploaded, err := server.client.upload(request.Context(), session.upstreamID, side, contentType, content)
	if err != nil {
		if server.clearIfNotFound(writer, token, err) {
			server.renderDashboard(writer, request, server.emptyScreen(&alertView{
				Title:   "This check is no longer available",
				Message: "Start another check and upload both document sides again.",
			}))
			return
		}
		server.renderDashboard(writer, request, server.withCurrentError(request, "Image could not be uploaded", err))
		return
	}
	server.sessions.update(token, uploaded.ExpiresAt)
	server.renderDashboard(writer, request, server.screenFromSession(upstreamSession{
		Status:        uploaded.Status,
		UploadedSides: uploaded.UploadedSides,
		ExpiresAt:     uploaded.ExpiresAt,
	}, nil, nil))
}

func (server *server) process(writer http.ResponseWriter, request *http.Request) {
	token, session, exists := server.sessionFromRequest(request)
	if !exists {
		server.renderDashboard(writer, request, server.emptyScreen(&alertView{
			Title:   "Upload both document sides first",
			Message: "A front and back image are required before verification can begin.",
		}))
		return
	}
	upstream, err := server.client.session(request.Context(), session.upstreamID)
	if err != nil {
		if server.clearIfNotFound(writer, token, err) {
			server.renderDashboard(writer, request, server.emptyScreen(&alertView{
				Title:   "This check is no longer available",
				Message: "Start another check and upload both document sides again.",
			}))
			return
		}
		server.renderDashboard(writer, request, server.withCurrentError(request, "Verification could not start", err))
		return
	}
	server.sessions.update(token, upstream.ExpiresAt)
	if !hasBothSides(upstream.UploadedSides) {
		server.renderDashboard(writer, request, server.screenFromSession(upstream, nil, &alertView{
			Title:   "Both document sides are required",
			Message: "Upload a front and back image before starting verification.",
		}))
		return
	}
	started, err := server.client.process(request.Context(), session.upstreamID)
	if err != nil {
		server.renderDashboard(writer, request, server.withCurrentError(request, "Verification could not start", err))
		return
	}
	upstream.Status = started.Status
	server.renderDashboard(writer, request, server.screenFromSession(upstream, nil, nil))
}

func (server *server) status(writer http.ResponseWriter, request *http.Request) {
	token, session, exists := server.sessionFromRequest(request)
	if !exists {
		server.renderDashboard(writer, request, server.emptyScreen(nil))
		return
	}
	upstream, err := server.client.session(request.Context(), session.upstreamID)
	if err != nil {
		if server.clearIfNotFound(writer, token, err) {
			server.renderDashboard(writer, request, server.emptyScreen(&alertView{
				Title:   "This check has expired",
				Message: "Start another check and upload both document sides again.",
			}))
			return
		}
		server.renderDashboard(writer, request, server.withCurrentError(request, "Status is temporarily unavailable", err))
		return
	}
	server.sessions.update(token, upstream.ExpiresAt)
	var result *documentResult
	if (upstream.Status == "success" || upstream.Status == "partial") && upstream.ResultAvailable {
		loaded, err := server.client.result(request.Context(), session.upstreamID)
		if err != nil {
			server.renderDashboard(writer, request, server.screenFromSession(upstream, nil, server.errorAlert("Results could not be loaded", err)))
			return
		}
		result = &loaded
	}
	server.renderDashboard(writer, request, server.screenFromSession(upstream, result, nil))
}

func (server *server) reset(writer http.ResponseWriter, request *http.Request) {
	token, session, exists := server.sessionFromRequest(request)
	if !exists {
		server.renderDashboard(writer, request, server.emptyScreen(nil))
		return
	}
	err := server.client.deleteSession(request.Context(), session.upstreamID)
	if err != nil {
		var upstream *upstreamError
		if !errors.As(err, &upstream) || upstream.StatusCode != http.StatusNotFound {
			server.renderDashboard(writer, request, server.withCurrentError(request, "This check could not be discarded", err))
			return
		}
	}
	server.sessions.delete(token)
	server.clearSessionCookie(writer)
	server.renderDashboard(writer, request, server.emptyScreen(nil))
}

func (server *server) readImage(request *http.Request, writer http.ResponseWriter) (string, []byte, error) {
	request.Body = http.MaxBytesReader(writer, request.Body, server.config.MaxUploadBytes+64*1024)
	reader, err := request.MultipartReader()
	if err != nil {
		return "", nil, clientInputError{"Choose a JPEG or PNG image to upload."}
	}
	for {
		part, err := reader.NextPart()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			if errors.Is(err, http.ErrBodyReadAfterClose) {
				return "", nil, clientInputError{"The upload was interrupted. Please try again."}
			}
			return "", nil, clientInputError{"The upload could not be read. Please choose the image again."}
		}
		if part.FormName() != "image" || part.FileName() == "" {
			continue
		}
		content, readErr := io.ReadAll(io.LimitReader(part, server.config.MaxUploadBytes+1))
		_ = part.Close()
		if readErr != nil {
			return "", nil, clientInputError{"The upload could not be read. Please try again."}
		}
		if int64(len(content)) > server.config.MaxUploadBytes {
			return "", nil, clientInputError{fmt.Sprintf("The image must be %s or smaller.", formatBytes(server.config.MaxUploadBytes))}
		}
		return part.Header.Get("Content-Type"), content, nil
	}
	return "", nil, clientInputError{"Choose a JPEG or PNG image to upload."}
}

type clientInputError struct {
	message string
}

func (err clientInputError) Error() string { return err.message }

func (server *server) sessionFromRequest(request *http.Request) (string, webSession, bool) {
	cookie, err := request.Cookie(sessionCookieName)
	if err != nil || cookie.Value == "" {
		return "", webSession{}, false
	}
	session, ok := server.sessions.get(cookie.Value)
	if !ok {
		return cookie.Value, webSession{}, false
	}
	return cookie.Value, session, true
}

func (server *server) setSessionCookie(writer http.ResponseWriter, token string) {
	http.SetCookie(writer, &http.Cookie{
		Name:     sessionCookieName,
		Value:    token,
		Path:     "/",
		HttpOnly: true,
		Secure:   server.config.CookieSecure,
		SameSite: http.SameSiteStrictMode,
	})
}

func (server *server) clearSessionCookie(writer http.ResponseWriter) {
	http.SetCookie(writer, &http.Cookie{
		Name:     sessionCookieName,
		Value:    "",
		Path:     "/",
		MaxAge:   -1,
		HttpOnly: true,
		Secure:   server.config.CookieSecure,
		SameSite: http.SameSiteStrictMode,
	})
}

func (server *server) clearIfNotFound(writer http.ResponseWriter, token string, err error) bool {
	var upstream *upstreamError
	if !errors.As(err, &upstream) || upstream.StatusCode != http.StatusNotFound {
		return false
	}
	server.sessions.delete(token)
	server.clearSessionCookie(writer)
	return true
}

func (server *server) withCurrentError(request *http.Request, title string, err error) screenView {
	_, session, exists := server.sessionFromRequest(request)
	if !exists {
		return server.emptyScreen(server.errorAlert(title, err))
	}
	upstream, getErr := server.client.session(request.Context(), session.upstreamID)
	if getErr != nil {
		return screenView{
			Status: lifecycleView{
				Label:       "Status temporarily unavailable",
				Description: "We could not reach the KYC service. This page will retry automatically.",
				Tone:        "warning",
				Busy:        true,
				Poll:        true,
			},
			MaxUpload:  formatBytes(server.config.MaxUploadBytes),
			HasSession: true,
			CanReset:   true,
			Error:      server.errorAlert(title, err),
		}
	}
	return server.screenFromSession(upstream, nil, server.errorAlert(title, err))
}

func (server *server) errorAlert(title string, err error) *alertView {
	var input clientInputError
	if errors.As(err, &input) {
		return &alertView{Title: title, Message: input.message}
	}
	var upstream *upstreamError
	if errors.As(err, &upstream) {
		message := strings.TrimSpace(upstream.Message)
		if message == "" {
			message = "The KYC service could not complete this request. Please try again."
		}
		if upstream.RetryAfter > 0 {
			message += fmt.Sprintf(" Try again in %d seconds.", int(upstream.RetryAfter.Seconds()))
		}
		return &alertView{Title: title, Message: message}
	}
	return &alertView{Title: title, Message: "The KYC service is temporarily unavailable. Please try again."}
}

func (server *server) emptyScreen(alert *alertView) screenView {
	return screenView{
		Status: lifecycleView{
			Label:       "Ready for documents",
			Description: "Upload clear images of the front and back of the identity card.",
			Tone:        "neutral",
		},
		CanUpload: true,
		MaxUpload: formatBytes(server.config.MaxUploadBytes),
		Error:     alert,
	}
}

func (server *server) screenFromSession(session upstreamSession, result *documentResult, alert *alertView) screenView {
	frontUploaded := hasSide(session.UploadedSides, "front")
	backUploaded := hasSide(session.UploadedSides, "back")
	status := lifecycleFor(session.Status, frontUploaded, backUploaded)
	terminal := session.Status == "success" || session.Status == "partial" || session.Status == "failed"
	view := screenView{
		Status:        status,
		MaxUpload:     formatBytes(server.config.MaxUploadBytes),
		HasSession:    true,
		FrontUploaded: frontUploaded,
		BackUploaded:  backUploaded,
		CanUpload:     !terminal && session.Status != "queued" && session.Status != "running",
		CanProcess:    frontUploaded && backUploaded && (session.Status == "created" || session.Status == "uploading"),
		CanReset:      true,
		Error:         alert,
	}
	if result != nil {
		view.Result = makeResultView(*result)
	}
	return view
}

func lifecycleFor(status string, frontUploaded bool, backUploaded bool) lifecycleView {
	switch status {
	case "queued":
		return lifecycleView{"Verification queued", "Your check is waiting to begin. This page will update automatically.", "neutral", true, true}
	case "running":
		return lifecycleView{"Verification in progress", "The KYC service is analysing the document. This can take up to 30 seconds.", "neutral", true, true}
	case "success":
		return lifecycleView{"Verification complete", "The extracted details are ready for review.", "success", false, false}
	case "partial":
		return lifecycleView{"Review required", "Some document details need operator review before this check can be completed.", "warning", false, false}
	case "failed":
		return lifecycleView{"Verification could not complete", "The document images and result have been cleared by the KYC service. Start a new check to try again.", "danger", false, false}
	default:
		if frontUploaded && backUploaded {
			return lifecycleView{"Ready to verify", "Both document sides are uploaded. Start verification when you are ready.", "success", false, false}
		}
		if frontUploaded || backUploaded {
			missing := "front"
			if frontUploaded {
				missing = "back"
			}
			return lifecycleView{"One side uploaded", "Upload the " + missing + " side to enable verification.", "neutral", false, false}
		}
		return lifecycleView{"Ready for documents", "Upload clear images of the front and back of the identity card.", "neutral", false, false}
	}
}

func hasBothSides(sides []string) bool {
	return hasSide(sides, "front") && hasSide(sides, "back")
}

func hasSide(sides []string, side string) bool {
	for _, candidate := range sides {
		if candidate == side {
			return true
		}
	}
	return false
}

func formatBytes(bytes int64) string {
	if bytes < 1024*1024 {
		return fmt.Sprintf("%d bytes", bytes)
	}
	return fmt.Sprintf("%.0f MB", float64(bytes)/(1024*1024))
}

func (server *server) renderPage(writer http.ResponseWriter, request *http.Request, view screenView) {
	server.htmlHeaders(writer)
	if err := server.templates.ExecuteTemplate(writer, "page", view); err != nil {
		return
	}
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

func isHTMXRequest(request *http.Request) bool {
	return request.Header.Get("HX-Request") == "true"
}

func (server *server) securityHeaders(next http.Handler) http.Handler {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set("Content-Security-Policy", "default-src 'self'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'; img-src 'self' data:; style-src 'self'; script-src 'self'")
		writer.Header().Set("Referrer-Policy", "no-referrer")
		writer.Header().Set("X-Content-Type-Options", "nosniff")
		writer.Header().Set("X-Frame-Options", "DENY")
		next.ServeHTTP(writer, request)
	})
}
