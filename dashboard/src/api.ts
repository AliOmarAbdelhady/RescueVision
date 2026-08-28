import axios from 'axios'

const api = axios.create({
  baseURL: '',
  timeout: 300000,
})

export interface DamageSummary {
  total_buildings: number
  no_damage: number
  minor_damage: number
  major_damage: number
  destroyed: number
}

export interface Urgency {
  score: number
  level: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'
}

export interface ArtifactUrls {
  building_mask?: string
  change_mask?: string
  damage_map?: string
  overlay_damage?: string
  overview?: string
  report_md?: string
  report_pdf?: string
}

export interface PredictionResult {
  case_id: string
  summary: DamageSummary
  urgency: Urgency
  artifact_urls: ArtifactUrls
  warnings: string[]
}

export interface CaseDetail {
  case_id: string
  summary: Record<string, unknown>
  artifacts: Record<string, string>
  report_markdown?: string
}

export interface ReportData {
  case_id: string
  report_markdown: string
  report_html?: string
  report_pdf?: string
}

export async function healthCheck() {
  const res = await api.get('/health')
  return res.data
}

export async function runPrediction(
  preImage: File,
  postImage: File,
  options?: { case_name?: string; enable_report?: boolean },
): Promise<PredictionResult> {
  const formData = new FormData()
  formData.append('pre_image', preImage)
  formData.append('post_image', postImage)
  if (options?.case_name) formData.append('case_name', options.case_name)
  if (options?.enable_report !== undefined)
    formData.append('enable_report', String(options.enable_report))

  const res = await api.post('/predict', formData)
  return res.data
}

export async function getCase(caseId: string): Promise<CaseDetail> {
  const res = await api.get(`/cases/${caseId}`)
  return res.data
}

export async function getReport(caseId: string): Promise<ReportData> {
  const res = await api.get(`/reports/${caseId}`)
  return res.data
}

export function getArtifactUrl(path: string | undefined): string | null {
  if (!path) return null
  return path
}
