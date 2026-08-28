import { useRef, useState } from 'react'
import { runPrediction } from '../api'
import type { PredictionResult } from '../api'

interface Props {
  onResult: (result: PredictionResult) => void
  onLoading: (loading: boolean) => void
  onError: (error: string | null) => void
  loading: boolean
}

export default function UploadPanel({ onResult, onLoading, onError, loading }: Props) {
  const preRef = useRef<HTMLInputElement>(null)
  const postRef = useRef<HTMLInputElement>(null)
  const [preName, setPreName] = useState<string>('')
  const [postName, setPostName] = useState<string>('')

  const handleSubmit = async () => {
    const preFile = preRef.current?.files?.[0]
    const postFile = postRef.current?.files?.[0]

    if (!preFile || !postFile) {
      onError('Please upload both pre-disaster and post-disaster images.')
      return
    }

    onError(null)
    onLoading(true)

    try {
      const result = await runPrediction(preFile, postFile)
      onResult(result)
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Prediction failed'
      onError(msg)
    } finally {
      onLoading(false)
    }
  }

  return (
    <div className="bg-white rounded-xl shadow-sm border p-6">
      <h2 className="text-lg font-semibold text-gray-800 mb-4">Upload Images</h2>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        {/* Pre-disaster */}
        <div
          className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center cursor-pointer hover:border-blue-400 hover:bg-blue-50 transition-colors"
          onClick={() => preRef.current?.click()}
        >
          <input
            ref={preRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => setPreName(e.target.files?.[0]?.name || '')}
          />
          <div className="text-3xl mb-2">&#127758;</div>
          <p className="font-medium text-gray-700">Pre-Disaster Image</p>
          {preName && <p className="text-sm text-blue-600 mt-1">{preName}</p>}
        </div>

        {/* Post-disaster */}
        <div
          className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center cursor-pointer hover:border-red-400 hover:bg-red-50 transition-colors"
          onClick={() => postRef.current?.click()}
        >
          <input
            ref={postRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => setPostName(e.target.files?.[0]?.name || '')}
          />
          <div className="text-3xl mb-2">&#127755;</div>
          <p className="font-medium text-gray-700">Post-Disaster Image</p>
          {postName && <p className="text-sm text-red-600 mt-1">{postName}</p>}
        </div>
      </div>

      <button
        onClick={handleSubmit}
        disabled={loading || !preName || !postName}
        className="w-full bg-red-600 hover:bg-red-700 disabled:bg-gray-300 disabled:cursor-not-allowed text-white font-semibold py-3 px-6 rounded-lg transition-colors"
      >
        {loading ? 'Analyzing...' : 'Run Damage Assessment'}
      </button>
    </div>
  )
}
