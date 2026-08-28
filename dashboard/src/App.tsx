import { useState } from 'react'
import UploadPanel from './components/UploadPanel'
import DamageSummaryCards from './components/DamageSummaryCards'
import ImageOverlayViewer from './components/ImageOverlayViewer'
import ReportPanel from './components/ReportPanel'
import { PredictionResult } from './api'

function App() {
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-slate-900 text-white shadow-lg">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-red-500 rounded-lg flex items-center justify-center text-xl font-bold">
              R
            </div>
            <div>
              <h1 className="text-xl font-bold">RescueVision</h1>
              <p className="text-sm text-gray-300">
                Disaster Damage Assessment Dashboard
              </p>
            </div>
          </div>
          <div className="text-xs text-gray-400">
            Research Prototype &mdash; Not for operational use
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-6 space-y-6">
        {/* Upload */}
        <UploadPanel
          onResult={setResult}
          onLoading={setLoading}
          onError={setError}
          loading={loading}
        />

        {/* Error */}
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-800 rounded-lg p-4">
            {error}
          </div>
        )}

        {/* Loading */}
        {loading && (
          <div className="flex items-center justify-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-red-500" />
            <span className="ml-4 text-gray-600">Running analysis...</span>
          </div>
        )}

        {/* Results */}
        {result && !loading && (
          <>
            <DamageSummaryCards summary={result.summary} urgency={result.urgency} />

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <ImageOverlayViewer artifacts={result.artifact_urls} />
              <ReportPanel caseId={result.case_id} />
            </div>

            {/* Warnings */}
            {result.warnings.length > 0 && (
              <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
                <h4 className="font-semibold text-amber-800 mb-2">Warnings</h4>
                <ul className="list-disc list-inside text-sm text-amber-700 space-y-1">
                  {result.warnings.map((w, i) => (
                    <li key={i}>{w}</li>
                  ))}
                </ul>
              </div>
            )}
          </>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-gray-100 border-t mt-12 py-4 text-center text-sm text-gray-500">
        RescueVision v0.1.0 &mdash; Research Prototype. Not a replacement for official emergency assessment.
      </footer>
    </div>
  )
}

export default App
