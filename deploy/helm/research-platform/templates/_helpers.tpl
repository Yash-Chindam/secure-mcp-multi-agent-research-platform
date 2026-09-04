{{/*
Chart name and release-qualified fullname.
*/}}
{{- define "research-platform.name" -}}
{{- .Chart.Name -}}
{{- end -}}

{{- define "research-platform.fullname" -}}
{{- .Release.Name -}}
{{- end -}}

{{/*
Labels every pod this chart creates carries. `plane` (passed as $ = context, component =
"application" | "mcp") is what networkpolicy.yaml selects on to separate the
application, mcp, data and observability planes (section 15).
*/}}
{{- define "research-platform.labels" -}}
app.kubernetes.io/name: {{ include "research-platform.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "research-platform.selectorLabels" -}}
app.kubernetes.io/name: {{ include "research-platform.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
