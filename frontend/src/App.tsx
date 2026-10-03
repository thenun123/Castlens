import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useCallback, useEffect, useRef, useState } from "react";
import { modelInfo, pct, predict, type ModelInfo, type Prediction } from "./api";

type State =
  | { phase: "idle" }
  | { phase: "analysing"; url: string }
  | { phase: "done"; url: string; result: Prediction }
  | { phase: "error"; url?: string; message: string };

const ACCEPT = "image/jpeg,image/png,image/webp,image/bmp";

export default function App() {
  const [state, setState] = useState<State>({ phase: "idle" });
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [backendStatus, setBackendStatus] = useState<"checking" | "waking" | "ready" | "error">("checking");
  const [dragging, setDragging] = useState(false);
  const abort = useRef<AbortController | null>(null);
  const reduce = useReducedMotion();

  useEffect(() => {
    const timer = setTimeout(() => {
      setBackendStatus((prev) => (prev === "checking" ? "waking" : prev));
    }, 1500);

    modelInfo().then((res) => {
      if (res) {
        setInfo(res);
        setBackendStatus("ready");
      } else {
        setBackendStatus("error");
      }
    });

    return () => clearTimeout(timer);
  }, []);

  useEffect(() => () => abort.current?.abort(), []);

  const run = useCallback(async (file: File | undefined) => {
    if (!file) return;
    abort.current?.abort();
    const ctrl = new AbortController();
    abort.current = ctrl;
    const url = URL.createObjectURL(file);
    setState((s) => {
      if ("url" in s && s.url) URL.revokeObjectURL(s.url);
      return { phase: "analysing", url };
    });
    try {
      const result = await predict(file, ctrl.signal);
      setState({ phase: "done", url, result });
    } catch (e) {
      if ((e as Error).name !== "AbortError") setState({ phase: "error", url, message: (e as Error).message });
    }
  }, []);

  const url = "url" in state ? state.url : undefined;
  const analysing = state.phase === "analysing";

  return (
    <div className="shell">
      <header className="top">
        <div className="brand">
          <svg width="22" height="22" viewBox="0 0 32 32" aria-hidden="true">
            <circle cx="16" cy="16" r="13" fill="none" stroke="currentColor" strokeWidth="3" />
            <circle cx="16" cy="16" r="4" fill="currentColor" />
          </svg>
          CastLens
        </div>
        {info && (
          <span className="model" title={`Model build ${info.version}`}>
            {info.name} · build {info.version.slice(0, 7)}
          </span>
        )}
      </header>

      {backendStatus === "waking" && (
        <div className="banner">
          The backend is waking up (Render free tier). This usually takes about 50 seconds. Please hold on!
        </div>
      )}
      {backendStatus === "error" && (
        <div className="banner" style={{ background: "#fdf2f0", borderColor: "var(--fail)", color: "var(--fail)" }}>
          Cannot connect to the backend. It might be down.
        </div>
      )}

      <main className="bench">
        <section className="stage" aria-label="Part photo">
          <label
            className={`drop ${dragging ? "is-drag" : ""} ${url ? "has-image" : ""}`}
            onDragOver={(e) => (e.preventDefault(), setDragging(true))}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => (e.preventDefault(), setDragging(false), void run(e.dataTransfer.files[0]))}
          >
            <input type="file" accept={ACCEPT} onChange={(e) => (void run(e.target.files?.[0]), (e.target.value = ""))} />
            <AnimatePresence mode="wait" initial={false}>
              {url ? (
                <motion.div key="img" className="frame" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  <img src={url} alt="Uploaded casting" />
                  {analysing && !reduce && (
                    <motion.div
                      className="scan"
                      initial={{ top: "0%" }}
                      animate={{ top: ["0%", "100%", "0%"] }}
                      transition={{ duration: 1.8, repeat: Infinity, ease: "easeInOut" }}
                    />
                  )}
                </motion.div>
              ) : (
                <motion.div key="empty" className="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  <Reticle />
                  <strong>Drop a casting photo here</strong>
                  <span>or click to choose a file · JPEG, PNG, WebP, BMP · up to 8 MB</span>
                </motion.div>
              )}
            </AnimatePresence>
          </label>
          {url && (
            <p className="hint">Drop or choose another photo to check a different part.</p>
          )}
        </section>

        <section className="readout" aria-live="polite" aria-busy={analysing}>
          <AnimatePresence mode="wait" initial={false}>
            {state.phase === "idle" && (
              <Panel key="idle">
                <h1>Is this casting defective?</h1>
                <p className="lede">Upload one top-down photo. You get a verdict plus the probability of each outcome.</p>
              </Panel>
            )}
            {analysing && (
              <Panel key="busy">
                <h1>Checking the part…</h1>
                <p className="lede">This usually takes under a second.</p>
              </Panel>
            )}
            {state.phase === "error" && (
              <Panel key="err">
                <h1>Could not check this photo</h1>
                <p className="lede error" role="alert">{state.message}</p>
              </Panel>
            )}
            {state.phase === "done" && <Result key="done" r={state.result} reduce={!!reduce} />}
          </AnimatePresence>
        </section>
      </main>

      <footer className="foot">
        Photos are checked in memory and never stored. Trained on one casting type and camera setup: results for other parts or lighting are not validated.
      </footer>
    </div>
  );
}

function Panel({ children }: { children: React.ReactNode }) {
  return (
    <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.2 }}>
      {children}
    </motion.div>
  );
}

function Result({ r, reduce }: { r: Prediction; reduce: boolean }) {
  const defect = r.verdict === "defect";
  const t = pct(r.threshold);
  return (
    <Panel>
      <div className={`verdict ${defect ? "bad" : "good"}`}>
        {defect ? <Cross /> : <Tick />}
        <h1>{defect ? "Defect likely" : "Looks OK"}</h1>
      </div>
      <p className="lede">
        {defect
          ? `Defect probability ${pct(r.probabilities.defect)} is above the ${t} inspection threshold.`
          : `Defect probability ${pct(r.probabilities.defect)} is below the ${t} inspection threshold.`}
      </p>
      {r.borderline && (
        <p className="note">
          The model rates <b>{r.highest === "ok" ? "OK" : "defect"}</b> as more likely, but the inspection threshold is set low on
          purpose so that borderline parts are flagged for a second look instead of passing.
        </p>
      )}

      <Bar label="OK" value={r.probabilities.ok} higher={r.highest === "ok"} tone="good" reduce={reduce} />
      <Bar label="Defect" value={r.probabilities.defect} higher={r.highest === "defect"} tone="bad" marker={r.threshold} reduce={reduce} />

      {r.warnings.map((w) => (
        <p className="warn" key={w}>{w}</p>
      ))}
      <p className="meta">Checked in {Math.round(r.inference_ms)} ms · {r.model.name} build {r.model.version.slice(0, 7)}</p>
    </Panel>
  );
}

function Bar(p: { label: string; value: number; higher: boolean; tone: "good" | "bad"; marker?: number; reduce: boolean }) {
  return (
    <div className="bar">
      <div className="bar-head">
        <span>{p.label}</span>
        {p.higher && <span className="tag">Higher</span>}
        <span className="num">{pct(p.value)}</span>
      </div>
      <div className="track" role="img" aria-label={`${p.label} probability ${pct(p.value)}`}>
        <motion.div
          className={`fill ${p.tone}`}
          initial={{ width: 0 }}
          animate={{ width: `${p.value * 100}%` }}
          transition={p.reduce ? { duration: 0 } : { type: "spring", stiffness: 90, damping: 16, delay: 0.1 }}
        />
        {p.marker !== undefined && (
          <div className="marker" style={{ left: `${p.marker * 100}%` }} title={`Inspection threshold ${pct(p.marker)}`}>
            <span>{pct(p.marker)} threshold</span>
          </div>
        )}
      </div>
    </div>
  );
}

const Reticle = () => (
  <svg width="56" height="56" viewBox="0 0 56 56" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
    <circle cx="28" cy="28" r="17" />
    <circle cx="28" cy="28" r="4" />
    <path d="M28 4v10M28 42v10M4 28h10M42 28h10" />
  </svg>
);
const Tick = () => (
  <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M4 12.5l5 5L20 6.5" />
  </svg>
);
const Cross = () => (
  <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" aria-hidden="true">
    <path d="M5 5l14 14M19 5L5 19" />
  </svg>
);
