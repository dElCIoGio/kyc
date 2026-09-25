package main

import (
	"fmt"
	"sort"
	"strings"
)

// documentResult is intentionally limited to review data. The API may also
// return candidate provenance, raw QR data, and timings; none are modeled or
// rendered by the operator console.
type documentResult struct {
	Status string      `json:"status"`
	Front  *sideResult `json:"front"`
	Back   *sideResult `json:"back"`
	Issues []issue     `json:"issues"`
}

type sideResult struct {
	Status       string           `json:"status"`
	DocumentType string           `json:"document_type"`
	Side         string           `json:"side"`
	ProfileID    string           `json:"profile_id"`
	Detection    *detection       `json:"detection"`
	Fields       map[string]field `json:"fields"`
	QRCode       *qrCode          `json:"qr_code"`
	Issues       []issue          `json:"issues"`
}

type detection struct {
	Confidence         float64 `json:"confidence"`
	OrientationDegrees int     `json:"orientation_degrees"`
}

type field struct {
	Status          string   `json:"status"`
	RawValue        *string  `json:"raw_value"`
	NormalizedValue *string  `json:"normalized_value"`
	Confidence      float64  `json:"confidence"`
	Warnings        []string `json:"warnings"`
}

type qrCode struct {
	Status   string         `json:"status"`
	Data     map[string]any `json:"data"`
	Warnings []string       `json:"warnings"`
}

type issue struct {
	Stage     string `json:"stage"`
	Code      string `json:"code"`
	Severity  string `json:"severity"`
	Message   string `json:"message"`
	FieldName string `json:"field_name"`
}

type resultView struct {
	Status string
	Tone   string
	Front  *sideView
	Back   *sideView
	Issues []issueView
}

type sideView struct {
	Name         string
	Status       string
	Tone         string
	DocumentType string
	ProfileID    string
	Detection    *detectionView
	Fields       []fieldView
	QR           *qrView
	Issues       []issueView
}

type detectionView struct {
	Confidence  string
	Orientation string
}

type fieldView struct {
	Name       string
	Value      string
	Status     string
	Tone       string
	Confidence string
	Warnings   []string
}

type qrView struct {
	Status   string
	Tone     string
	Fields   []fieldView
	Warnings []string
}

type issueView struct {
	Stage    string
	Code     string
	Severity string
	Tone     string
	Message  string
	Field    string
}

func makeResultView(result documentResult) *resultView {
	return &resultView{
		Status: title(result.Status),
		Tone:   statusTone(result.Status),
		Front:  makeSideView("Front", result.Front),
		Back:   makeSideView("Back", result.Back),
		Issues: makeIssueViews(result.Issues),
	}
}

func makeSideView(name string, side *sideResult) *sideView {
	if side == nil {
		return nil
	}
	view := &sideView{
		Name:         name,
		Status:       title(side.Status),
		Tone:         statusTone(side.Status),
		DocumentType: title(side.DocumentType),
		ProfileID:    side.ProfileID,
		Issues:       makeIssueViews(side.Issues),
	}
	if side.Detection != nil {
		view.Detection = &detectionView{
			Confidence:  formatPercent(side.Detection.Confidence),
			Orientation: fmt.Sprintf("%d°", side.Detection.OrientationDegrees),
		}
	}
	view.Fields = makeFieldViews(side.Fields)
	if side.QRCode != nil {
		view.QR = makeQRView(side.QRCode)
	}
	return view
}

func makeFieldViews(fields map[string]field) []fieldView {
	names := make([]string, 0, len(fields))
	for name := range fields {
		names = append(names, name)
	}
	sort.Strings(names)
	views := make([]fieldView, 0, len(names))
	for _, name := range names {
		field := fields[name]
		value := "Not available"
		if field.NormalizedValue != nil && *field.NormalizedValue != "" {
			value = *field.NormalizedValue
		} else if field.RawValue != nil && *field.RawValue != "" {
			value = *field.RawValue
		}
		views = append(views, fieldView{
			Name:       title(name),
			Value:      value,
			Status:     title(field.Status),
			Tone:       statusTone(field.Status),
			Confidence: formatPercent(field.Confidence),
			Warnings:   field.Warnings,
		})
	}
	return views
}

func makeQRView(code *qrCode) *qrView {
	view := &qrView{
		Status:   title(code.Status),
		Tone:     statusTone(code.Status),
		Warnings: code.Warnings,
	}
	if code.Data == nil {
		return view
	}
	names := make([]string, 0, len(code.Data))
	for name := range code.Data {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		value, ok := code.Data[name].(string)
		if !ok || value == "" {
			continue
		}
		view.Fields = append(view.Fields, fieldView{
			Name:   title(name),
			Value:  value,
			Status: "Decoded",
			Tone:   "success",
		})
	}
	return view
}

func makeIssueViews(issues []issue) []issueView {
	views := make([]issueView, 0, len(issues))
	for _, issue := range issues {
		views = append(views, issueView{
			Stage:    title(issue.Stage),
			Code:     issue.Code,
			Severity: title(issue.Severity),
			Tone:     statusTone(issue.Severity),
			Message:  issue.Message,
			Field:    title(issue.FieldName),
		})
	}
	return views
}

func formatPercent(value float64) string {
	return fmt.Sprintf("%.0f%%", value*100)
}

func title(value string) string {
	if value == "" {
		return ""
	}
	words := strings.Fields(strings.ReplaceAll(value, "_", " "))
	for index, word := range words {
		words[index] = strings.ToUpper(word[:1]) + strings.ToLower(word[1:])
	}
	return strings.Join(words, " ")
}

func statusTone(status string) string {
	switch strings.ToLower(status) {
	case "success", "valid", "decoded":
		return "success"
	case "partial", "uncertain", "warning", "not_detected", "decode_failed", "parse_failed":
		return "warning"
	case "failed", "invalid", "missing", "error":
		return "danger"
	default:
		return "neutral"
	}
}
