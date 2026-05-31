import React, { useState, useRef, useCallback, useEffect } from "react";

const API_URL = "http://127.0.0.1:8000/predict";

const META = {
  alzheimers: { color: "#f0568c", grad: "linear-gradient(135deg,#f0568c,#ff8fb0)", soft: "#fff2f6", desc: "Indicators consistent with Alzheimer's" },
  normal:     { color: "#12b5a5", grad: "linear-gradient(135deg,#12b5a5,#4fd6c6)", soft: "#eafaf8", desc: "No indicators of Alzheimer's detected" },
};

function CountUp({ end, dur = 1000, decimals = 0, suffix = "" }) {
  const [v, setV] = useState(0);
  useEffect(() => {
    let raf, start;
    const tick = (t) => { if (!start) start = t; const p = Math.min((t - start) / dur, 1);
      setV(end * (1 - Math.pow(1 - p, 3))); if (p < 1) raf = requestAnimationFrame(tick); };
    raf = requestAnimationFrame(tick); return () => cancelAnimationFrame(raf);
  }, [end, dur]);
  return <>{v.toFixed(decimals)}{suffix}</>;
}

function Ring({ pct, color }) {
  const r = 40, c = 2 * Math.PI * r;
  const [off, setOff] = useState(c);
  useEffect(() => { const t = setTimeout(() => setOff(c - (pct / 100) * c), 120); return () => clearTimeout(t); }, [pct, c]);
  return (
    <svg width="112" height="112" viewBox="0 0 112 112" style={{ flexShrink: 0 }}>
      <circle cx="56" cy="56" r={r} fill="none" stroke="#eef1f6" strokeWidth="10" />
      <circle cx="56" cy="56" r={r} fill="none" stroke={color} strokeWidth="10"
        strokeDasharray={c} strokeDashoffset={off} strokeLinecap="round" transform="rotate(-90 56 56)"
        style={{ transition: "stroke-dashoffset 1.2s cubic-bezier(.34,1.2,.4,1)" }} />
      <text x="56" y="54" textAnchor="middle" fontSize="24" fontWeight="800" fill="#1c1f3a">{pct.toFixed(0)}<tspan fontSize="14">%</tspan></text>
      <text x="56" y="72" textAnchor="middle" fontSize="9" fill="#9aa0b8" letterSpacing="1.5">CONFIDENCE</text>
    </svg>
  );
}

export default function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [fileName, setFileName] = useState("");
  const inputRef = useRef(null);

  const upload = useCallback(async (file) => {
    if (!file) return;
    setError(null); setLoading(true); setResult(null); setFileName(file.name);
    try {
      const fd = new FormData(); fd.append("file", file);
      const res = await fetch(API_URL, { method: "POST", body: fd });
      if (!res.ok) throw new Error(`${res.status}`);
      await new Promise(r => setTimeout(r, 400));
      setResult(await res.json());
    } catch { setError("Could not reach the model server. Is the FastAPI backend running on port 8000?"); }
    finally { setLoading(false); }
  }, []);

  const onDrop = (e) => { e.preventDefault(); setDragging(false); if (e.dataTransfer.files?.[0]) upload(e.dataTransfer.files[0]); };
  const reset = () => { setResult(null); setError(null); setFileName(""); };
  const meta = result ? META[result.prediction_key] : null;

  return (
    <div style={S.page}>
      <style>{CSS}</style>

      {/* top nav */}
      <nav style={S.nav}>
        <div style={S.navBrand}>
          <div style={S.navLogo}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2">
              <path d="M9 3a3 3 0 0 0-3 3 3 3 0 0 0-2 5 3 3 0 0 0 1 5 3 3 0 0 0 4 2 3 3 0 0 0 2 0M15 3a3 3 0 0 1 3 3 3 3 0 0 1 2 5 3 3 0 0 1-1 5 3 3 0 0 1-4 2 3 3 0 0 1-2 0M12 3v18" />
            </svg>
          </div>
          <span style={S.navName}>NeuroDetect</span>
        </div>
        <div style={S.navStats}>
          <span style={S.navStat}><b><CountUp end={76.5} decimals={1} suffix="%" /></b> acc</span>
          <span style={S.navStat}><b><CountUp end={0.81} decimals={2} /></b> auc</span>
          <span style={S.navBadge}>2D CNN + Grad-CAM</span>
        </div>
      </nav>

      <main style={S.main}>
        {!result && !loading && (
          <div style={S.hero} className="rise">
            <span style={S.pill}>Educational Deep Learning Demo</span>
            <h1 style={S.h1}>Brain MRI classification<br /><span style={S.h1grad}>with Grad-CAM explainability</span></h1>
            <p style={S.heroSub}>Upload an axial brain MRI slice to see the model's prediction, confidence, and a Grad-CAM heatmap of the regions it focused on.</p>
          </div>
        )}

        {!result && (
          <section
            style={{ ...S.drop, ...(dragging ? S.dropOn : {}) }}
            className="rise d1 dropHover"
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => inputRef.current?.click()}
          >
            <input ref={inputRef} type="file" accept="image/*" hidden onChange={(e) => upload(e.target.files?.[0])} />
            {loading ? (
              <><div className="spin" style={S.spinner} /><p style={S.dropText}>Analyzing&hellip;</p></>
            ) : (
              <>
                <div style={S.dropIcon} className="bob">
                  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#12b5a5" strokeWidth="1.7">
                    <path d="M12 16V4M12 4l-4 4M12 4l4 4M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
                  </svg>
                </div>
                <p style={S.dropText}>{fileName || "Drop a brain MRI slice here"}</p>
                <p style={S.dropHint}>or click to browse &middot; JPG / PNG</p>
              </>
            )}
          </section>
        )}

        {error && <div style={S.error} className="rise">{error}</div>}

        {result && meta && (
          <section style={S.dash} className="rise">
            {/* verdict banner */}
            <div style={{ ...S.banner, background: meta.grad }} className="pop">
              <div>
                <p style={S.bLabel}>PREDICTION</p>
                <h2 style={S.bClass}>{result.prediction}</h2>
                <p style={S.bDesc}>{meta.desc}</p>
              </div>
              <div style={S.bRing}><Ring pct={result.confidence * 100} color="#fff" /></div>
            </div>

            {/* grid: images + probabilities */}
            <div style={S.grid}>
              <div style={S.card} className="rise d1">
                <p style={S.cardTitle}>Scan Analysis</p>
                <div style={S.imgRow}>
                  <figure style={S.fig}>
                    <figcaption style={S.cap}>Input</figcaption>
                    <img src={result.input_image} alt="input" style={S.img} />
                  </figure>
                  <figure style={S.fig}>
                    <figcaption style={S.cap}>Grad-CAM</figcaption>
                    <img src={result.gradcam_image} alt="gradcam" style={S.img} />
                  </figure>
                </div>
              </div>

              <div style={S.card} className="rise d2">
                <p style={S.cardTitle}>Probabilities</p>
                {[...result.probabilities].sort((a, b) => b.value - a.value).map((p, i) => (
                  <div key={p.key} style={S.pRow}>
                    <div style={S.pTop}><span style={S.pName}>{p.label}</span><span style={{ ...S.pPct, color: META[p.key].color }}>{(p.value * 100).toFixed(1)}%</span></div>
                    <div style={S.pBg}><div style={{ ...S.pFill, width: `${p.value * 100}%`, background: META[p.key].grad, transitionDelay: `${i * 0.12}s` }} /></div>
                  </div>
                ))}
                <button style={S.btn} onClick={reset} className="btnHover">Analyze another scan</button>
              </div>
            </div>
          </section>
        )}
      </main>

      <footer style={S.footer}>Educational and research use only &middot; Not a medical device</footer>
    </div>
  );
}

const CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');
  * { box-sizing: border-box; margin: 0; }
  .rise { animation: rise .6s cubic-bezier(.2,.7,.2,1) both; }
  .d1{animation-delay:.1s}.d2{animation-delay:.2s}
  @keyframes rise { from{opacity:0;transform:translateY(18px)} to{opacity:1;transform:none} }
  .pop { animation: pop .6s cubic-bezier(.34,1.56,.64,1) both; }
  @keyframes pop { from{opacity:0;transform:scale(.96)} to{opacity:1;transform:scale(1)} }
  .bob { animation: bob 3s ease-in-out infinite; }
  @keyframes bob { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-6px)} }
  .spin { animation: spin .8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg) } }
  .dropHover:hover { transform: translateY(-3px); box-shadow: 0 20px 50px rgba(18,181,165,.18); border-color:#12b5a5; }
  .btnHover:hover { background:#12b5a5; color:#fff; }
`;

const HEAD = "'Plus Jakarta Sans', system-ui, sans-serif";
const BODY = "'Inter', system-ui, sans-serif";

const S = {
  page: { minHeight: "100vh", fontFamily: BODY, color: "#1c1f3a",
    background: "radial-gradient(1000px 600px at 85% -5%, #e3fbf6 0%, transparent 55%), radial-gradient(900px 600px at 0% 100%, #fde8f0 0%, transparent 55%), #f6f8fc" },
  nav: { display: "flex", justifyContent: "space-between", alignItems: "center", padding: "18px 30px",
    background: "rgba(255,255,255,.7)", backdropFilter: "blur(14px)", borderBottom: "1px solid #ecf0f5", position: "sticky", top: 0, zIndex: 10 },
  navBrand: { display: "flex", alignItems: "center", gap: 10 },
  navLogo: { width: 36, height: 36, borderRadius: 10, background: "linear-gradient(135deg,#12b5a5,#4fd6c6)", display: "grid", placeItems: "center", boxShadow: "0 4px 14px rgba(18,181,165,.4)" },
  navName: { fontFamily: HEAD, fontWeight: 800, fontSize: 18, letterSpacing: "-0.02em" },
  navStats: { display: "flex", alignItems: "center", gap: 18, fontSize: 13, color: "#6b7280" },
  navStat: { color: "#6b7280" },
  navBadge: { background: "#eafaf8", color: "#12b5a5", fontSize: 11.5, fontWeight: 600, padding: "5px 11px", borderRadius: 20 },
  main: { maxWidth: 880, margin: "0 auto", padding: "40px 24px" },
  hero: { textAlign: "center", marginBottom: 30 },
  pill: { display: "inline-block", background: "#fff", border: "1px solid #e8ecf2", color: "#12b5a5", fontSize: 12, fontWeight: 600, padding: "6px 14px", borderRadius: 30, boxShadow: "0 4px 14px rgba(28,31,58,.05)" },
  h1: { fontFamily: HEAD, fontSize: 38, fontWeight: 800, lineHeight: 1.15, letterSpacing: "-0.03em", margin: "18px 0 12px" },
  h1grad: { background: "linear-gradient(135deg,#12b5a5,#f0568c)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text" },
  heroSub: { color: "#6b7280", fontSize: 15, maxWidth: 520, margin: "0 auto", lineHeight: 1.6 },
  drop: { position: "relative", border: "2px dashed #d4dae3", borderRadius: 24, padding: "52px 24px", textAlign: "center",
    cursor: "pointer", background: "#fff", boxShadow: "0 12px 40px rgba(28,31,58,.06)", transition: "all .25s cubic-bezier(.2,.7,.2,1)" },
  dropOn: { borderColor: "#12b5a5", background: "#f3fdfb", transform: "scale(1.01)" },
  dropIcon: { width: 64, height: 64, borderRadius: 18, background: "#eafaf8", margin: "0 auto 14px", display: "grid", placeItems: "center" },
  dropText: { fontSize: 16, fontWeight: 700, fontFamily: HEAD },
  dropHint: { fontSize: 13, color: "#9aa0b0", marginTop: 4 },
  spinner: { width: 38, height: 38, borderRadius: "50%", border: "3px solid #e3e8ef", borderTopColor: "#12b5a5", margin: "0 auto 14px" },
  error: { background: "#fff2f6", color: "#c0506a", padding: 16, borderRadius: 14, marginTop: 20, fontSize: 14, textAlign: "center", border: "1px solid #fbd7e2" },
  dash: { display: "flex", flexDirection: "column", gap: 18 },
  banner: { display: "flex", justifyContent: "space-between", alignItems: "center", borderRadius: 24, padding: "30px 36px",
    color: "#fff", boxShadow: "0 18px 50px rgba(28,31,58,.18)" },
  bLabel: { fontSize: 11, letterSpacing: "0.12em", opacity: .85 },
  bClass: { fontFamily: HEAD, fontSize: 34, fontWeight: 800, margin: "6px 0" },
  bDesc: { fontSize: 14, opacity: .9 },
  bRing: { background: "rgba(255,255,255,.18)", borderRadius: "50%", padding: 6 },
  grid: { display: "grid", gridTemplateColumns: "1.3fr 1fr", gap: 18 },
  card: { background: "#fff", borderRadius: 22, padding: "22px 24px", border: "1px solid #eef1f6", boxShadow: "0 12px 36px rgba(28,31,58,.06)" },
  cardTitle: { fontFamily: HEAD, fontSize: 15, fontWeight: 700, marginBottom: 16, color: "#1c1f3a" },
  imgRow: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 },
  fig: {},
  cap: { fontSize: 10.5, letterSpacing: "0.06em", color: "#9aa0b0", marginBottom: 7, textTransform: "uppercase", fontWeight: 600 },
  img: { width: "100%", borderRadius: 12, display: "block" },
  pRow: { marginBottom: 16 },
  pTop: { display: "flex", justifyContent: "space-between", marginBottom: 6 },
  pName: { fontSize: 13.5, fontWeight: 500, color: "#3d4252" },
  pPct: { fontSize: 13.5, fontWeight: 700 },
  pBg: { height: 10, background: "#eef1f6", borderRadius: 6, overflow: "hidden" },
  pFill: { height: "100%", borderRadius: 6, transition: "width 1s cubic-bezier(.34,1.2,.4,1)" },
  btn: { marginTop: 10, width: "100%", padding: "12px", borderRadius: 12, border: "1.5px solid #12b5a5",
    background: "#fff", color: "#12b5a5", fontWeight: 700, fontFamily: HEAD, fontSize: 14, cursor: "pointer", transition: "all .2s ease" },
  footer: { textAlign: "center", color: "#9aa0b0", fontSize: 12, padding: "30px 0 40px" },
};