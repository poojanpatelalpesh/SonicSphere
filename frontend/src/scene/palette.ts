import type { Theme } from "../state/store";

export interface ScenePalette {
  bg: string; fog: string; grid: string; gridSection: string; floor: string;
  skin: string; hoodie: string; pants: string; shoe: string; chair: string; chairAccent: string; metal: string;
  cupLeft: string; cupRight: string; band: string;
  sphere: string; ring: string; source: string; sourceGlow: string; path: string; pathDraw: string; handle: string;
  label: string; labelBg: string; ray: string; front: string; ambient: number; key: number;
}

export const PALETTES: Record<Theme, ScenePalette> = {
  dark: {
    bg: "#060912", fog: "#060912", grid: "#1b2741", gridSection: "#26406b", floor: "#0a1020",
    skin: "#d9a988", hoodie: "#4b6390", pants: "#2f3d5e", shoe: "#e5e7eb", chair: "#252e44", chairAccent: "#22d3ee", metal: "#5b6b86",
    cupLeft: "#22d3ee", cupRight: "#f472b6", band: "#2b3550",
    sphere: "#3b82f6", ring: "#22d3ee", source: "#fbbf24", sourceGlow: "#f59e0b", path: "#a78bfa", pathDraw: "#f472b6", handle: "#e9d5ff",
    label: "#cbd5e1", labelBg: "rgba(10,15,28,.72)", ray: "#fbbf24", front: "#22d3ee", ambient: 0.55, key: 1.8,
  },
  light: {
    bg: "#eef2f8", fog: "#eef2f8", grid: "#cdd6e4", gridSection: "#9fb2d0", floor: "#e5ebf4",
    skin: "#d6a687", hoodie: "#4f6795", pants: "#334262", shoe: "#ffffff", chair: "#3a4358", chairAccent: "#0891b2", metal: "#8a97ad",
    cupLeft: "#0891b2", cupRight: "#db2777", band: "#3a4562",
    sphere: "#2563eb", ring: "#0891b2", source: "#d97706", sourceGlow: "#f59e0b", path: "#7c3aed", pathDraw: "#db2777", handle: "#5b21b6",
    label: "#1e293b", labelBg: "rgba(255,255,255,.82)", ray: "#d97706", front: "#0891b2", ambient: 0.65, key: 1.3,
  },
};
