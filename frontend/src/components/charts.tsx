import { useLayoutEffect, useMemo, useRef, useState } from "react";

export interface Series { name: string; color: string; x: ArrayLike<number>; y: ArrayLike<number>; dash?: string; width?: number }

function ticks(min: number, max: number, n = 5): number[] {
  const span = max - min || 1;
  const step0 = span / n;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) ?? 10 * mag;
  const out: number[] = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(Math.abs(v) < 1e-12 ? 0 : v);
  return out;
}

const fmtTick = (v: number) => (Math.abs(v) >= 1000 ? `${(v / 1000).toFixed(v % 1000 ? 1 : 0)}k` : Math.abs(v) < 0.01 && v !== 0 ? v.toExponential(0) : `${+v.toFixed(3)}`);

export function LineChart({ series, xLabel, yLabel, height = 190, xLog = false, yDomain, xDomain, title, legend = true, ariaLabel }: {
  series: Series[]; xLabel?: string; yLabel?: string; height?: number; xLog?: boolean; yDomain?: [number, number]; xDomain?: [number, number];
  title?: string; legend?: boolean; ariaLabel: string;
}) {
  // The SVG uses the container's pixel width as its coordinate width, so text keeps its size.
  const box = useRef<HTMLElement>(null);
  const [W, setW] = useState(560);
  useLayoutEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(280, Math.round(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const H = height, ml = 46, mr = 10, mt = title ? 22 : 10, mb = 34;
  const tx = (v: number) => (xLog ? Math.log10(Math.max(v, 1e-9)) : v);
  const { xmin, xmax, ymin, ymax } = useMemo(() => {
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    for (const s of series) for (let i = 0; i < s.x.length; i++) {
      const x = tx(s.x[i]), y = s.y[i];
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      xmin = Math.min(xmin, x); xmax = Math.max(xmax, x); ymin = Math.min(ymin, y); ymax = Math.max(ymax, y);
    }
    if (xDomain) { xmin = tx(xDomain[0]); xmax = tx(xDomain[1]); }
    if (yDomain) { ymin = yDomain[0]; ymax = yDomain[1]; } else { const pad = (ymax - ymin) * 0.08 || 1; ymin -= pad; ymax += pad; }
    return { xmin, xmax, ymin, ymax };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [series, xLog, yDomain, xDomain]);
  const sx = (v: number) => ml + ((tx(v) - xmin) / (xmax - xmin || 1)) * (W - ml - mr);
  const sy = (v: number) => mt + (1 - (v - ymin) / (ymax - ymin || 1)) * (H - mt - mb);
  const xt = xLog ? [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000].filter((v) => tx(v) >= xmin && tx(v) <= xmax) : ticks(xmin, xmax, 6);
  const yt = ticks(ymin, ymax, 4);
  if (!Number.isFinite(xmin)) return <figure ref={box} className="grid h-40 w-full place-items-center text-xs muted">no data</figure>;
  return (
    <figure className="w-full" ref={box}>
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} className="block max-w-full" role="img" aria-label={ariaLabel}>
        {title && <text x={ml} y={14} fontSize="11.5" fontWeight={600} fill="var(--text)">{title}</text>}
        {yt.map((v) => <g key={`y${v}`}><line x1={ml} x2={W - mr} y1={sy(v)} y2={sy(v)} stroke="var(--line)" strokeWidth={1} />
          <text x={ml - 6} y={sy(v) + 3.5} fontSize="10" textAnchor="end" fill="var(--muted)">{fmtTick(v)}</text></g>)}
        {xt.map((v) => <g key={`x${v}`}><line x1={sx(v)} x2={sx(v)} y1={mt} y2={H - mb} stroke="var(--line)" strokeWidth={0.6} />
          <text x={sx(v)} y={H - mb + 13} fontSize="10" textAnchor="middle" fill="var(--muted)">{fmtTick(v)}</text></g>)}
        {series.map((s) => {
          let d = "";
          for (let i = 0; i < s.x.length; i++) {
            const x = s.x[i], y = s.y[i];
            if (!Number.isFinite(y) || (xLog && x <= 0)) continue;
            d += `${d ? "L" : "M"}${sx(x).toFixed(1)},${sy(Math.max(ymin, Math.min(ymax, y))).toFixed(1)}`;
          }
          return <path key={s.name} d={d} fill="none" stroke={s.color} strokeWidth={s.width ?? 1.5} strokeDasharray={s.dash} strokeLinejoin="round" />;
        })}
        {xLabel && <text x={(ml + W - mr) / 2} y={H - 4} fontSize="10.5" textAnchor="middle" fill="var(--muted)">{xLabel}</text>}
        {yLabel && <text x={11} y={(mt + H - mb) / 2} fontSize="10.5" textAnchor="middle" fill="var(--muted)" transform={`rotate(-90 11 ${(mt + H - mb) / 2})`}>{yLabel}</text>}
      </svg>
      {legend && (
        <figcaption className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-[11px] muted">
          {series.map((s) => <span key={s.name} className="inline-flex items-center gap-1.5">
            <svg width="16" height="6"><line x1="0" x2="16" y1="3" y2="3" stroke={s.color} strokeWidth="2" strokeDasharray={s.dash} /></svg>{s.name}</span>)}
        </figcaption>
      )}
    </figure>
  );
}

