export interface Prediction {
  verdict: "ok" | "defect";
  highest: "ok" | "defect";
  borderline: boolean;
  probabilities: { ok: number; defect: number };
  threshold: number;
  warnings: string[];
  model: { name: string; version: string };
  inference_ms: number;
}

export interface ModelInfo {
  name: string;
  version: string;
  threshold: number;
}

const FRIENDLY: Record<number, string> = {
  413: "That file is too large. Upload an image under 8 MB.",
  422: "That file could not be read as an image. Upload a JPEG, PNG, WebP or BMP photo.",
  429: "Too many checks in a short time. Wait a minute and try again.",
};

export async function predict(file: File, signal?: AbortSignal): Promise<Prediction> {
  const body = new FormData();
  body.append("file", file);
  let res: Response;
  try {
    res = await fetch("/api/predict", { method: "POST", body, signal });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new Error("Could not reach the server. Check your connection and try again.");
  }
  if (!res.ok) {
    let detail = "";
    try {
      detail = (await res.json()).detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(FRIENDLY[res.status] ?? (detail || "The server hit a problem. Try again in a moment."));
  }
  return res.json();
}

export async function modelInfo(): Promise<ModelInfo | null> {
  try {
    const res = await fetch("/api/model");
    return res.ok ? res.json() : null;
  } catch {
    return null;
  }
}

export const pct = (p: number) => `${(p * 100).toFixed(p > 0.9995 || p < 0.0005 ? 2 : 1)}%`;
