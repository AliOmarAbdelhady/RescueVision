import { useState } from 'react'
import type { ArtifactUrls } from '../api'

interface Props {
  artifacts: ArtifactUrls
}

type Layer = 'overview' | 'building_mask' | 'damage_map' | 'overlay_damage' | 'change_mask'

export default function ImageOverlayViewer({ artifacts }: Props) {
  const [activeLayer, setActiveLayer] = useState<Layer>('overview')

  const layers: { key: Layer; label: string; url: string | undefined }[] = [
    { key: 'overview', label: 'Overview', url: artifacts.overview },
    { key: 'building_mask', label: 'Buildings', url: artifacts.building_mask },
    { key: 'damage_map', label: 'Damage Map', url: artifacts.damage_map },
    { key: 'overlay_damage', label: 'Damage Overlay', url: artifacts.overlay_damage },
    { key: 'change_mask', label: 'Change Detection', url: artifacts.change_mask },
  ]

  const activeUrl = layers.find((l) => l.key === activeLayer)?.url

  return (
    <div className="bg-white rounded-xl shadow-sm border p-4">
      <h3 className="text-lg font-semibold text-gray-800 mb-3">Image Viewer</h3>

      {/* Layer tabs */}
      <div className="flex flex-wrap gap-2 mb-4">
        {layers.map((layer) => (
          <button
            key={layer.key}
            onClick={() => setActiveLayer(layer.key)}
            disabled={!layer.url}
            className={`px-3 py-1.5 text-sm rounded-full border transition-colors ${
              activeLayer === layer.key
                ? 'bg-slate-800 text-white border-slate-800'
                : 'bg-white text-gray-600 border-gray-300 hover:border-gray-400'
            } disabled:opacity-30 disabled:cursor-not-allowed`}
          >
            {layer.label}
          </button>
        ))}
      </div>

      {/* Image display */}
      <div className="bg-gray-100 rounded-lg overflow-hidden flex items-center justify-center min-h-[400px]">
        {activeUrl ? (
          <img
            src={activeUrl}
            alt={activeLayer}
            className="max-w-full max-h-[600px] object-contain"
          />
        ) : (
          <p className="text-gray-400">No image available for this layer</p>
        )}
      </div>
    </div>
  )
}
