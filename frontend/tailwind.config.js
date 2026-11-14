/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "Segoe UI", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "Cascadia Code", "Consolas", "monospace"],
      },
      colors: {
        ink: { 950: "#05070d", 900: "#0a0f1c", 850: "#0e1526", 800: "#131c31", 700: "#1d2944", 600: "#2a3a5e" },
        accent: { DEFAULT: "#22d3ee", soft: "#67e8f9", deep: "#0891b2" },
        ember: { DEFAULT: "#f472b6" },
      },
      boxShadow: { glow: "0 0 0 1px rgba(34,211,238,.25), 0 8px 30px -8px rgba(34,211,238,.35)" },
    },
  },
  plugins: [],
};
