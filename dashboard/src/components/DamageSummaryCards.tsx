import type { DamageSummary, Urgency } from '../api'

interface Props {
  summary: DamageSummary
  urgency: Urgency
}

const urgencyColors: Record<string, string> = {
  LOW: 'bg-green-100 text-green-800 border-green-300',
  MEDIUM: 'bg-yellow-100 text-yellow-800 border-yellow-300',
  HIGH: 'bg-orange-100 text-orange-800 border-orange-300',
  CRITICAL: 'bg-red-100 text-red-800 border-red-300',
}

export default function DamageSummaryCards({ summary, urgency }: Props) {
  const total = summary.total_buildings || 1
  const cards = [
    { label: 'Total Buildings', value: summary.total_buildings, color: 'bg-slate-100 text-slate-800' },
    { label: 'No Damage', value: summary.no_damage, pct: ((summary.no_damage / total) * 100).toFixed(1), color: 'bg-green-50 text-green-800' },
    { label: 'Minor Damage', value: summary.minor_damage, pct: ((summary.minor_damage / total) * 100).toFixed(1), color: 'bg-yellow-50 text-yellow-800' },
    { label: 'Major Damage', value: summary.major_damage, pct: ((summary.major_damage / total) * 100).toFixed(1), color: 'bg-orange-50 text-orange-800' },
    { label: 'Destroyed', value: summary.destroyed, pct: ((summary.destroyed / total) * 100).toFixed(1), color: 'bg-red-50 text-red-800' },
  ]

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        {cards.map((c) => (
          <div key={c.label} className={`${c.color} rounded-lg p-4 border`}>
            <p className="text-xs font-medium opacity-70">{c.label}</p>
            <p className="text-2xl font-bold mt-1">{c.value}</p>
            {c.pct && <p className="text-xs opacity-60">{c.pct}%</p>}
          </div>
        ))}
      </div>

      <div className={`${urgencyColors[urgency.level]} rounded-lg p-4 border flex items-center justify-between`}>
        <div>
          <p className="text-sm font-medium">Urgency Level</p>
          <p className="text-2xl font-bold">{urgency.level}</p>
        </div>
        <div className="text-right">
          <p className="text-sm opacity-70">Score</p>
          <p className="text-xl font-mono font-semibold">{urgency.score.toFixed(3)}</p>
        </div>
      </div>
    </div>
  )
}
