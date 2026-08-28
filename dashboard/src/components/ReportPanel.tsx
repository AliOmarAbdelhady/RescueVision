import { useEffect, useState } from 'react'
import { getReport } from '../api'
import type { ReportData } from '../api'

interface Props {
  caseId: string
}

export default function ReportPanel({ caseId }: Props) {
  const [report, setReport] = useState<ReportData | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    getReport(caseId)
      .then((data) => {
        if (!cancelled) setReport(data)
      })
      .catch(() => {})
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => { cancelled = true }
  }, [caseId])

  if (loading) {
    return (
      <div className="bg-white rounded-xl shadow-sm border p-4">
        <h3 className="text-lg font-semibold text-gray-800 mb-3">Report</h3>
        <div className="animate-pulse space-y-3">
          <div className="h-4 bg-gray-200 rounded w-3/4" />
          <div className="h-4 bg-gray-200 rounded w-1/2" />
          <div className="h-4 bg-gray-200 rounded w-2/3" />
        </div>
      </div>
    )
  }

  if (!report) {
    return (
      <div className="bg-white rounded-xl shadow-sm border p-4">
        <h3 className="text-lg font-semibold text-gray-800 mb-3">Report</h3>
        <p className="text-gray-400">No report available</p>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-xl shadow-sm border p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-lg font-semibold text-gray-800">Assessment Report</h3>
        <div className="flex gap-2">
          {report.report_pdf && (
            <a
              href={`/reports/${caseId}/download`}
              className="px-3 py-1.5 text-sm bg-red-600 text-white rounded-lg hover:bg-red-700"
            >
              Download PDF
            </a>
          )}
          <a
            href={`/cases/${caseId}`}
            className="px-3 py-1.5 text-sm bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200"
          >
            JSON
          </a>
        </div>
      </div>

      <div className="prose prose-sm max-w-none bg-gray-50 rounded-lg p-4 max-h-[500px] overflow-y-auto">
        <pre className="whitespace-pre-wrap font-sans text-sm text-gray-700">
          {report.report_markdown || 'No report content available.'}
        </pre>
      </div>
    </div>
  )
}
