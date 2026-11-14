import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./index.css";
import { getEngine, liveRef, useStore } from "./state/store";

// Development-only hook for automated end-to-end tests (stripped from production builds).
if (import.meta.env.DEV) {
  (window as unknown as Record<string, unknown>).__sonicsphere = { useStore, getEngine, liveRef };
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
