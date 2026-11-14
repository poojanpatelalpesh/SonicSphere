import type { ReactNode } from "react";

export function Panel({ title, icon, right, children, className = "" }: { title?: ReactNode; icon?: ReactNode; right?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`panel ${className}`}>
      {title && (
        <div className="panel-title">
          {icon}
          <span>{title}</span>
          <span className="ml-auto normal-case tracking-normal">{right}</span>
        </div>
      )}
      {children}
    </section>
  );
}

export function Seg<T extends string>({ value, options, onChange, label }: { value: T; options: { id: T; label: ReactNode; title?: string }[]; onChange: (v: T) => void; label: string }) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={o.id} type="button" aria-pressed={value === o.id} title={o.title} onClick={() => onChange(o.id)}>{o.label}</button>
      ))}
    </div>
  );
}

export function Slider({ label, value, min, max, step, onChange, format, id }: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; format?: (v: number) => string; id: string }) {
  return (
    <label htmlFor={id} className="block">
      <div className="mb-1 flex items-center justify-between text-xs">
        <span className="muted">{label}</span>
        <span className="font-mono tabular-nums">{format ? format(value) : value}</span>
      </div>
      <input id={id} type="range" className="w-full" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  );
}

export function Toggle({ label, checked, onChange, hint }: { label: string; checked: boolean; onChange: (v: boolean) => void; hint?: string }) {
  return (
    <label className="flex cursor-pointer items-center justify-between gap-3 text-sm" title={hint}>
      <span>{label}</span>
      <button type="button" role="switch" aria-checked={checked} onClick={() => onChange(!checked)}
              className="relative h-5 w-9 shrink-0 rounded-full border transition"
              style={{ background: checked ? "var(--accent)" : "var(--panel-2)", borderColor: checked ? "var(--accent)" : "var(--line)" }}>
        <span className="absolute top-0.5 h-3.5 w-3.5 rounded-full bg-white shadow transition-all" style={{ left: checked ? 18 : 2 }} />
      </button>
    </label>
  );
}

export function Meter({ value, color, label }: { value: number; color: string; label: string }) {
  const db = 20 * Math.log10(Math.max(value, 1e-5));
  const pct = Math.max(0, Math.min(100, ((db + 60) / 60) * 100));
  return (
    <div className="flex items-center gap-2 text-[11px]" aria-label={`${label} level ${db.toFixed(0)} dBFS`}>
      <span className="w-3 font-semibold" style={{ color }}>{label}</span>
      <div className="h-1.5 flex-1 overflow-hidden rounded-full" style={{ background: "var(--panel-2)" }}>
        <div className="h-full rounded-full transition-[width] duration-75" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="w-12 text-right font-mono tabular-nums muted">{value > 1e-5 ? `${db.toFixed(0)} dB` : "−∞"}</span>
    </div>
  );
}

export const fmt = {
  deg: (v: number | undefined | null, d = 1) => (v === undefined || v === null || !Number.isFinite(v) ? "–" : `${v.toFixed(d)}°`),
  num: (v: number | undefined | null, d = 2) => (v === undefined || v === null || !Number.isFinite(v) ? "–" : v.toFixed(d)),
  ms: (v: number | undefined | null, d = 2) => (v === undefined || v === null || !Number.isFinite(v) ? "n/a" : `${v.toFixed(d)} ms`),
  us: (v: number | undefined | null, d = 0) => (v === undefined || v === null || !Number.isFinite(v) ? "n/a" : `${v.toFixed(d)} µs`),
};
