package main

import "sort"

// normalizedResult matches the public result projection. Keeping the decoder
// narrow prevents accidental rendering of future diagnostic fields.
type normalizedResult struct {
	SessionID      string             `json:"session_id"`
	Document       normalizedDocument `json:"document"`
	Liveness       checkState         `json:"liveness"`
	FaceComparison checkState         `json:"face_comparison"`
}
type normalizedDocument struct {
	Status       string                     `json:"status"`
	DocumentType *string                    `json:"document_type"`
	Fields       map[string]normalizedField `json:"fields"`
	Issues       []normalizedIssue          `json:"issues"`
}
type normalizedField struct {
	Status string  `json:"status"`
	Value  *string `json:"value"`
}
type normalizedIssue struct {
	Code      string  `json:"code"`
	Message   string  `json:"message"`
	FieldName *string `json:"field_name"`
}

type resultView struct {
	DocumentStatus, DocumentType, LivenessStatus, FaceComparisonStatus string
	Fields                                                             []resultFieldView
	Issues                                                             []resultIssueView
}
type resultFieldView struct{ Name, Status, Value string }
type resultIssueView struct{ Code, Message, Field string }

func makeResultView(result normalizedResult) resultView {
	view := resultView{DocumentStatus: label(result.Document.Status), LivenessStatus: label(result.Liveness.Status), FaceComparisonStatus: label(result.FaceComparison.Status)}
	if result.Document.DocumentType != nil {
		view.DocumentType = label(*result.Document.DocumentType)
	}
	names := make([]string, 0, len(result.Document.Fields))
	for name := range result.Document.Fields {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		field := result.Document.Fields[name]
		value := "Not available"
		if field.Value != nil && *field.Value != "" {
			value = *field.Value
		}
		view.Fields = append(view.Fields, resultFieldView{Name: label(name), Status: label(field.Status), Value: value})
	}
	for _, issue := range result.Document.Issues {
		field := ""
		if issue.FieldName != nil {
			field = label(*issue.FieldName)
		}
		view.Issues = append(view.Issues, resultIssueView{Code: issue.Code, Message: issue.Message, Field: field})
	}
	return view
}
