interface Props {
  cases: Array<{
    tile_id: string
    similarity: number
    disaster_type?: string
    damage_distribution?: Record<string, number>
  }>
}

export default function SimilarCases({ cases }: Props) {
  if (cases.length === 0) return null

  return (
    <div className="bg-white rounded-xl shadow-sm border p-4">
      <h3 className="text-lg font-semibold text-gray-800 mb-3">Similar Disaster Cases</h3>
      <div className="space-y-3">
        {cases.map((c, i) => (
          <div key={c.tile_id} className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
            <span className="text-lg font-bold text-gray-400">#{i + 1}</span>
            <div className="flex-1">
              <p className="text-sm font-medium text-gray-800">{c.tile_id}</p>
              {c.disaster_type && (
                <p className="text-xs text-gray-500">{c.disaster_type}</p>
              )}
            </div>
            <div className="text-right">
              <p className="text-sm font-mono text-blue-600">
                {(c.similarity * 100).toFixed(1)}%
              </p>
              <p className="text-xs text-gray-400">similarity</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
