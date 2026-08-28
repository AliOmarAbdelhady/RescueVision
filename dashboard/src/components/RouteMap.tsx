interface Props {
  routeImageUrl?: string
  safeRoutes: string[]
  blockedRoutes: string[]
}

export default function RouteMap({ routeImageUrl, safeRoutes, blockedRoutes }: Props) {
  return (
    <div className="bg-white rounded-xl shadow-sm border p-4">
      <h3 className="text-lg font-semibold text-gray-800 mb-3">Route Map</h3>

      {routeImageUrl && (
        <div className="mb-4 bg-gray-100 rounded-lg overflow-hidden">
          <img
            src={routeImageUrl}
            alt="Route map"
            className="max-w-full max-h-[400px] object-contain"
          />
        </div>
      )}

      <div className="grid grid-cols-2 gap-4">
        <div>
          <h4 className="text-sm font-medium text-green-700 mb-2">Safe Routes</h4>
          {safeRoutes.length > 0 ? (
            <ul className="text-sm text-gray-600 space-y-1">
              {safeRoutes.map((r) => (
                <li key={r} className="flex items-center gap-2">
                  <span className="w-2 h-2 bg-green-500 rounded-full" />
                  {r}
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-gray-400">No safe routes identified</p>
          )}
        </div>
        <div>
          <h4 className="text-sm font-medium text-red-700 mb-2">Blocked Routes</h4>
          {blockedRoutes.length > 0 ? (
            <ul className="text-sm text-gray-600 space-y-1">
              {blockedRoutes.map((r) => (
                <li key={r} className="flex items-center gap-2">
                  <span className="w-2 h-2 bg-red-500 rounded-full" />
                  {r}
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-gray-400">No blocked routes</p>
          )}
        </div>
      </div>
    </div>
  )
}
