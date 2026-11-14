/**
 * Data access for the website.
 *
 * Static mode (default for production builds): everything is read from files exported by
 * `python -m sonicsphere.export_static` into public/data/, so the site needs no server.
 * API mode (default for `npm run dev`): the FastAPI backend, proxied at /api by Vite.
 * Override with VITE_DATA_MODE=static|api.
 */
const MODE = import.meta.env.VITE_DATA_MODE ?? (import.meta.env.PROD ? "static" : "api");
export const STATIC_DATA = MODE === "static";
const DATA = `${import.meta.env.BASE_URL}data/`;

export interface SubjectInfo { id: string; downloaded: boolean; note: string }
export interface DatasetSummary { id: string; name: string; default_subject: string; downloaded_subjects: string[]; listed_subjects: number }
export interface DatasetInfo {
  id: string; name: string; subject: string; homepage: string; download_url: string; reference: string;
  license_summary: string; license_text: string; sofa_conventions: string; sample_rate: number; hrir_length: number;
  num_directions: number; data_type: string; subjects: SubjectInfo[]; grid: Record<string, unknown>;
  coordinates: Record<string, string>; notes: string[]; sha256: string; attributes: Record<string, string>;
}
export interface SoundEntry {
  id: string; name: string; file: string; kind: string; loop: boolean; icon: string; description: string;
  source: string; license: string; duration_s: number; sample_rate: number; channels: number; url: string;
}
export interface Spectrum { freq_hz: number[]; left_db: number[]; right_db: number[] }
export interface HoldoutResult {
  dataset: string; subject: string; index: number; direction: { azimuth: number; elevation: number; lateral: number; polar: number };
  fs: number; measured: { left: number[]; right: number[]; spectrum: Spectrum };
  estimates: Record<string, { left: number[]; right: number[]; spectrum: Spectrum; status: string }>;
  metrics: Record<string, { lsd_db: number; nmse_db: number; itd_err_us: number }>;
}
export interface PrepareResult {
  dataset: string; subject: string; fs: number; filter_length: number; align_margin: number; toa_range_samples: number[];
  itd_range_us: number[]; num_triangles: number; num_gap_triangles: number; num_boundary_edges: number;
  max_valid_edge_deg: number; truncated_energy_db: number; align_roundtrip_nmse_db_median: number;
  align_roundtrip_nmse_db_max: number; gain: number; pack_bytes: number; etag: string; seconds: number;
}
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type Json = any;

async function req(path: string, init?: RequestInit): Promise<Response> {
  let r: Response;
  try {
    r = await fetch(path, init);
  } catch (e) {
    throw new Error(STATIC_DATA ? `Could not load ${path} (${(e as Error).message}). Check your connection and reload.`
                              : `Backend unreachable (${(e as Error).message}). Is it running on port 8000?`);
  }
  if (!r.ok) {
    let detail = `${r.status} ${r.statusText}`;
    try { const j = await r.json(); if (j.detail) detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch { /* not json */ }
    throw new Error(detail);
  }
  return r;
}

const json = async <T>(path: string, init?: RequestInit): Promise<T> => (await req(path, init)).json() as Promise<T>;

export const api = {
  health: () => json<{ status: string; version: string; datasets: Record<string, string[]> }>("/api/health"),
  config: () => json<Json>("/api/config"),
  datasets: () => json<DatasetSummary[]>(STATIC_DATA ? `${DATA}datasets.json` : "/api/datasets"),
  dataset: async (id: string, subject?: string) => {
    if (STATIC_DATA) {
      const list = await json<DatasetSummary[]>(`${DATA}datasets.json`);
      const sid = subject ?? list.find((d) => d.id === id)?.default_subject ?? "";
      return json<DatasetInfo>(`${DATA}datasets/${id}/${sid}.json`);
    }
    return json<DatasetInfo>(`/api/datasets/${encodeURIComponent(id)}${subject ? `?subject=${encodeURIComponent(subject)}` : ""}`);
  },
  filters: async (id: string, subject: string, fs?: number) => {
    if (STATIC_DATA) {
      const rate = Math.round(fs ?? 44100);
      try {
        return await (await req(`${DATA}filters/${id}_${subject}_${rate}.ssfp`)).arrayBuffer();
      } catch (e) {
        throw new Error(`No ear filters prepared for ${rate} Hz on this site (${(e as Error).message}).`);
      }
    }
    return (await req(`/api/datasets/${encodeURIComponent(id)}/subjects/${encodeURIComponent(subject)}/filters${fs ? `?fs=${fs}` : ""}`)).arrayBuffer();
  },
  prepare: (id: string, subject: string) =>
    json<PrepareResult>(`/api/datasets/${encodeURIComponent(id)}/subjects/${encodeURIComponent(subject)}/prepare`, { method: "POST" }),
  sounds: async () => {
    if (!STATIC_DATA) return json<SoundEntry[]>("/api/sounds");
    const list = await json<SoundEntry[]>(`${DATA}sounds.json`);
    return list.map((s) => ({ ...s, url: `${DATA}${s.url}` }));
  },
  soundData: async (url: string) => (await req(url)).arrayBuffer(),
  holdout: (dataset: string, subject: string, az: number, el: number) =>
    json<HoldoutResult>(`/api/analysis/holdout?dataset=${dataset}&subject=${subject}&azimuth=${az}&elevation=${el}`),
  evaluation: () => json<Json>("/api/evaluation/interpolation"),
  benchmarks: () => json<Json>("/api/benchmarks"),
  render: async (params: Record<string, unknown>, file?: File | Blob | null) => {
    const fd = new FormData();
    fd.append("params", JSON.stringify(params));
    if (file) fd.append("file", file, (file as File).name ?? "upload.wav");
    const r = await req("/api/render", { method: "POST", body: fd });
    const info = JSON.parse(r.headers.get("X-Render-Info") ?? "{}");
    return { blob: await r.blob(), info };
  },
};
