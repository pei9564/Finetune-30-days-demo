{{- define "finetune-platform.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "finetune-platform.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := include "finetune-platform.name" . -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "finetune-platform.controlNamespace" -}}
{{- default "lora-system" .Values.namespace -}}
{{- end -}}

{{- define "finetune-platform.labels" -}}
app.kubernetes.io/name: {{ include "finetune-platform.name" . }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | trunc 63 | trimSuffix "-" }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "finetune-platform.selectorLabels" -}}
app.kubernetes.io/name: {{ include "finetune-platform.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "finetune-platform.service" -}}
{{- $ctx := .context -}}
{{- $namespace := default (include "finetune-platform.controlNamespace" $ctx) .namespace -}}
apiVersion: v1
kind: Service
metadata:
  name: {{ include "finetune-platform.fullname" $ctx }}-{{ .name }}
  namespace: {{ $namespace }}
  labels:
    {{- include "finetune-platform.labels" $ctx | nindent 4 }}
    app.kubernetes.io/component: {{ .name }}
spec:
  type: {{ .cfg.service.type }}
  selector:
    {{- include "finetune-platform.selectorLabels" $ctx | nindent 4 }}
    app.kubernetes.io/component: {{ .name }}
  ports:
    - name: {{ default "http" .cfg.service.name }}
      port: {{ .cfg.service.port }}
      targetPort: {{ .cfg.service.port }}
      protocol: TCP
{{- if and (eq .cfg.service.type "NodePort") .cfg.service.nodePort }}
      nodePort: {{ .cfg.service.nodePort }}
{{- end }}
{{- end -}}
