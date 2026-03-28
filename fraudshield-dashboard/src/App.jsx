import { useState, useEffect, useRef, useCallback } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import {
  Shield,
  Sun,
  Moon,
  Activity,
  AlertTriangle,
  CheckCircle,
  MessageSquare,
  Cpu,
} from "lucide-react";
import axios from "axios";
import "./index.css";

const API_BASE = "http://127.0.0.1:5000";
const API_KEY = import.meta.env.VITE_API_KEY || "";

const MPESA_TYPES = [
  "send_money",
  "lipa_na_mpesa_till",
  "lipa_na_mpesa_paybill",
  "pochi_la_biashara",
  "withdraw_agent",
  "deposit_agent",
  "mpesa_global",
  "fuliza",
];
const AIRTEL_TYPES = [
  "airtel_send",
  "airtel_pay",
  "airtel_withdraw",
  "airtel_deposit",
];
const TKASH_TYPES = ["tkash_send", "tkash_pay", "tkash_withdraw"];
const EQUITEL_TYPES = ["equitel_send", "eazzy_pay", "equitel_withdraw"];
const SIM_FRAUD_TYPES = [
  "send_money",
  "airtel_send",
  "tkash_send",
  "equitel_send",
];
const SIM_LEGIT_TYPES = [
  ...MPESA_TYPES.slice(1),
  ...AIRTEL_TYPES.slice(1),
  ...TKASH_TYPES.slice(1),
  ...EQUITEL_TYPES.slice(1),
];

function providerOf(type) {
  if (MPESA_TYPES.includes(type))
    return { name: "M-Pesa", cls: "provider-mpesa" };
  if (AIRTEL_TYPES.includes(type))
    return { name: "Airtel", cls: "provider-airtel" };
  if (TKASH_TYPES.includes(type))
    return { name: "T-Kash", cls: "provider-tkash" };
  if (EQUITEL_TYPES.includes(type))
    return { name: "Equitel", cls: "provider-equitel" };
  return { name: "Generic", cls: "" };
}

const headers = { "Content-Type": "application/json", "X-API-Key": API_KEY };

function randomTx(isFraud) {
  if (isFraud) {
    const type =
      SIM_FRAUD_TYPES[Math.floor(Math.random() * SIM_FRAUD_TYPES.length)];
    const amt = Math.round(10000 + Math.random() * 490000);
    return {
      step: Math.ceil(Math.random() * 720),
      type,
      amount: amt,
      oldbalanceOrg: amt,
      newbalanceOrig: 0,
      oldbalanceDest: 0,
      newbalanceDest: 0,
      isFlaggedFraud: 0,
    };
  }
  const type =
    SIM_LEGIT_TYPES[Math.floor(Math.random() * SIM_LEGIT_TYPES.length)];
  const amt = Math.round(50 + Math.random() * 49950);
  const bal = Math.round(amt + Math.random() * 200000);
  return {
    step: Math.ceil(Math.random() * 720),
    type,
    amount: amt,
    oldbalanceOrg: bal,
    newbalanceOrig: bal - amt,
    oldbalanceDest: Math.round(Math.random() * 50000),
    newbalanceDest: Math.round(Math.random() * 50000),
    isFlaggedFraud: 0,
  };
}

export default function App() {
  const [theme, setTheme] = useState("dark");
  const [tab, setTab] = useState("manual");
  const [stats, setStats] = useState({
    total: 0,
    fraud: 0,
    probs: [],
    peak: 0,
  });
  const [feed, setFeed] = useState([]);
  const [smsLog, setSmsLog] = useState([]);
  const [chartData, setChartData] = useState([]);
  const [simRunning, setSimRunning] = useState(false);
  const [simSpeed, setSimSpeed] = useState(1000);
  const [health, setHealth] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const simRef = useRef(null);

  // Form state
  const [form, setForm] = useState({
    type: "send_money",
    step: 1,
    amount: 4878,
    oldbalanceOrg: 170136,
    newbalanceOrig: 165258,
    oldbalanceDest: 0,
    newbalanceDest: 4878,
    isFlaggedFraud: 0,
  });

  // Fetch health on mount
  useEffect(() => {
    axios
      .get(`${API_BASE}/health`)
      .then((r) => setHealth(r.data))
      .catch(() => setHealth(null));
  }, []);

  const recordResult = useCallback((res, payload) => {
    setStats((prev) => {
      const probs = [...prev.probs, res.fraud_probability].slice(-200);
      return {
        total: prev.total + 1,
        fraud: prev.fraud + (res.prediction === 1 ? 1 : 0),
        probs,
        peak: Math.max(prev.peak, res.fraud_probability),
      };
    });
    setFeed((prev) =>
      [{ res, payload, ts: new Date().toLocaleTimeString() }, ...prev].slice(
        0,
        100,
      ),
    );
    setChartData((prev) =>
      [
        ...prev,
        {
          idx: prev.length + 1,
          prob: parseFloat((res.fraud_probability * 100).toFixed(2)),
        },
      ].slice(-60),
    );
    if (res.sms_alert_sent) {
      setSmsLog((prev) =>
        [
          {
            ts: new Date().toLocaleTimeString(),
            type: payload.type,
            amount: payload.amount,
            prob: res.fraud_probability,
            message: `[FraudShield Alert] SUSPICIOUS TRANSACTION DETECTED\nType: ${payload.type.toUpperCase()}\nAmount: KES ${Number(payload.amount).toLocaleString()}\nRisk: ${(res.fraud_probability * 100).toFixed(1)}%`,
          },
          ...prev,
        ].slice(0, 50),
      );
    }
  }, []);

  const callAPI = useCallback(async (payload) => {
    const res = await axios.post(`${API_BASE}/predict`, payload, { headers });
    return res.data;
  }, []);

  const handlePredict = async () => {
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const res = await callAPI({ ...form, type: form.type });
      recordResult(res, form);
      setResult(res);
    } catch (e) {
      setError("Could not reach API. Is the server running on port 5000?");
    } finally {
      setLoading(false);
    }
  };

  const loadFraudExample = () =>
    setForm({
      type: "send_money",
      step: 1,
      amount: 450000,
      oldbalanceOrg: 450000,
      newbalanceOrig: 0,
      oldbalanceDest: 0,
      newbalanceDest: 0,
      isFlaggedFraud: 0,
    });

  const runSimStep = useCallback(async () => {
    const isFraud = Math.random() < 0.15;
    const payload = randomTx(isFraud);
    try {
      const res = await callAPI(payload);
      if (!res.error) recordResult(res, payload);
    } catch {}
  }, [callAPI, recordResult]);

  const toggleSim = () => {
    if (simRunning) {
      clearInterval(simRef.current);
      setSimRunning(false);
    } else {
      runSimStep();
      simRef.current = setInterval(runSimStep, simSpeed);
      setSimRunning(true);
    }
  };

  useEffect(() => {
    if (simRunning) {
      clearInterval(simRef.current);
      simRef.current = setInterval(runSimStep, simSpeed);
    }
    return () => clearInterval(simRef.current);
  }, [simSpeed, simRunning, runSimStep]);

  const fraudRate = stats.total ? (stats.fraud / stats.total) * 100 : 0;
  const avgConf = stats.probs.length
    ? (stats.probs.reduce((a, b) => a + b, 0) / stats.probs.length) * 100
    : 0;
  const lastProb = stats.probs.length
    ? stats.probs[stats.probs.length - 1] * 100
    : 0;

  const chartColor = theme === "dark" ? "#00ff88" : "#059669";
  const gridColor =
    theme === "dark" ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.06)";

  return (
    <div
      className={theme}
      style={{
        minHeight: "100vh",
        background:
          theme === "dark"
            ? "radial-gradient(ellipse at top left, rgba(0,255,136,0.06) 0%, #0a0c10 40%, #060810 100%)"
            : "#f8fafc",
      }}
    >
      <div className="grid-bg" />
      <div className="app" style={{ position: "relative", zIndex: 1 }}>
        {/* Header */}
        <header className="header">
          <div className="logo">
            <div className="logo-icon">
              <Shield size={18} color="var(--accent)" />
            </div>
            <div>
              <div className="logo-text">FRAUDSHIELD</div>
              <div className="logo-sub">Kenya Mobile Money Monitor v3.0</div>
            </div>
          </div>
          <div className="header-right">
            <div className="status-dot">
              <div className={`dot ${simRunning ? "live" : ""}`} />
              <span>
                {simRunning
                  ? "Simulation live"
                  : health
                    ? "API connected"
                    : "API offline"}
              </span>
            </div>
            <div className="theme-toggle">
              <button
                className={`theme-btn ${theme === "light" ? "active" : ""}`}
                onClick={() => setTheme("light")}
              >
                <Sun size={12} style={{ marginRight: 4 }} />
                Light
              </button>
              <button
                className={`theme-btn ${theme === "dark" ? "active" : ""}`}
                onClick={() => setTheme("dark")}
              >
                <Moon size={12} style={{ marginRight: 4 }} />
                Dark
              </button>
            </div>
          </div>
        </header>

        {/* Metrics */}
        <div className="metric-grid">
          <div className="card metric-card">
            <div className="metric-label">Total Checked</div>
            <div className="metric-val">{stats.total}</div>
            <div className="metric-sub">all time</div>
          </div>
          <div className="card metric-card">
            <div className="metric-label">Fraud Detected</div>
            <div className="metric-val red">{stats.fraud}</div>
            <div className="metric-sub">
              {stats.total ? `${fraudRate.toFixed(2)}% rate` : "—"}
            </div>
          </div>
          <div className="card metric-card">
            <div className="metric-label">Avg Confidence</div>
            <div className="metric-val green">
              {stats.probs.length ? `${avgConf.toFixed(1)}%` : "—"}
            </div>
            <div className="metric-sub">
              threshold {(0.6835 * 100).toFixed(1)}%
            </div>
          </div>
          <div className="card metric-card">
            <div className="metric-label">Peak Probability</div>
            <div className={`metric-val ${stats.peak > 0.6835 ? "red" : ""}`}>
              {stats.peak ? `${(stats.peak * 100).toFixed(1)}%` : "—"}
            </div>
            <div className="metric-sub">SMS alerts: {smsLog.length}</div>
          </div>
        </div>

        {/* Main grid */}
        <div className="main-grid">
          {/* Left panel */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">Transaction Analysis</span>
            </div>
            <div className="card-body">
              <div className="tabs">
                {["manual", "sim", "feed", "sms"].map((t) => (
                  <button
                    key={t}
                    className={`tab ${tab === t ? "active" : ""}`}
                    onClick={() => setTab(t)}
                  >
                    {t}
                  </button>
                ))}
              </div>

              {/* Manual */}
              {tab === "manual" && (
                <div>
                  <div className="field">
                    <label>Transaction type</label>
                    <select
                      value={form.type}
                      onChange={(e) =>
                        setForm({ ...form, type: e.target.value })
                      }
                    >
                      <optgroup label="── M-Pesa">
                        {MPESA_TYPES.map((t) => (
                          <option key={t} value={t}>
                            {t
                              .replace(/_/g, " ")
                              .replace(/\b\w/g, (c) => c.toUpperCase())}
                          </option>
                        ))}
                      </optgroup>
                      <optgroup label="── Airtel Money">
                        {AIRTEL_TYPES.map((t) => (
                          <option key={t} value={t}>
                            {t
                              .replace(/_/g, " ")
                              .replace(/\b\w/g, (c) => c.toUpperCase())}
                          </option>
                        ))}
                      </optgroup>
                      <optgroup label="── T-Kash">
                        {TKASH_TYPES.map((t) => (
                          <option key={t} value={t}>
                            {t
                              .replace(/_/g, " ")
                              .replace(/\b\w/g, (c) => c.toUpperCase())}
                          </option>
                        ))}
                      </optgroup>
                      <optgroup label="── Equitel">
                        {EQUITEL_TYPES.map((t) => (
                          <option key={t} value={t}>
                            {t
                              .replace(/_/g, " ")
                              .replace(/\b\w/g, (c) => c.toUpperCase())}
                          </option>
                        ))}
                      </optgroup>
                    </select>
                  </div>
                  <div className="field-row">
                    <div className="field">
                      <label>Step (hour)</label>
                      <input
                        type="number"
                        value={form.step}
                        onChange={(e) =>
                          setForm({ ...form, step: +e.target.value })
                        }
                      />
                    </div>
                    <div className="field">
                      <label>Amount (KES)</label>
                      <input
                        type="number"
                        value={form.amount}
                        onChange={(e) =>
                          setForm({ ...form, amount: +e.target.value })
                        }
                      />
                    </div>
                  </div>
                  <div className="field-row">
                    <div className="field">
                      <label>Sender — before</label>
                      <input
                        type="number"
                        value={form.oldbalanceOrg}
                        onChange={(e) =>
                          setForm({ ...form, oldbalanceOrg: +e.target.value })
                        }
                      />
                    </div>
                    <div className="field">
                      <label>Sender — after</label>
                      <input
                        type="number"
                        value={form.newbalanceOrig}
                        onChange={(e) =>
                          setForm({ ...form, newbalanceOrig: +e.target.value })
                        }
                      />
                    </div>
                  </div>
                  <div className="field-row">
                    <div className="field">
                      <label>Receiver — before</label>
                      <input
                        type="number"
                        value={form.oldbalanceDest}
                        onChange={(e) =>
                          setForm({ ...form, oldbalanceDest: +e.target.value })
                        }
                      />
                    </div>
                    <div className="field">
                      <label>Receiver — after</label>
                      <input
                        type="number"
                        value={form.newbalanceDest}
                        onChange={(e) =>
                          setForm({ ...form, newbalanceDest: +e.target.value })
                        }
                      />
                    </div>
                  </div>
                  <div className="field">
                    <label>System fraud flag</label>
                    <select
                      value={form.isFlaggedFraud}
                      onChange={(e) =>
                        setForm({ ...form, isFlaggedFraud: +e.target.value })
                      }
                    >
                      <option value={0}>No (0)</option>
                      <option value={1}>Yes (1)</option>
                    </select>
                  </div>
                  <button
                    className="btn-primary"
                    onClick={handlePredict}
                    disabled={loading}
                  >
                    {loading ? "Analyzing..." : "Analyze Transaction →"}
                  </button>
                  <button className="btn-ghost" onClick={loadFraudExample}>
                    Load fraud scenario
                  </button>
                  {error && (
                    <div
                      style={{
                        marginTop: 10,
                        padding: "10px 14px",
                        borderRadius: 8,
                        background: "var(--danger-dim)",
                        color: "var(--danger)",
                        fontSize: 12,
                        fontFamily: "var(--mono)",
                      }}
                    >
                      {error}
                    </div>
                  )}
                  {result && (
                    <div
                      className={`result-box ${result.prediction === 1 ? "fraud" : "legit"}`}
                    >
                      <div
                        className={`result-verdict ${result.prediction === 1 ? "fraud" : "legit"}`}
                      >
                        {result.prediction === 1
                          ? "⚠ FRAUD DETECTED"
                          : "✓ LEGITIMATE"}
                      </div>
                      <div className="result-detail">
                        probability{" "}
                        {(result.fraud_probability * 100).toFixed(1)}% ·
                        threshold {(result.threshold_used * 100).toFixed(1)}%
                      </div>
                      <div className="result-tx">
                        {providerOf(result.transaction_type).name} ·{" "}
                        {result.transaction_type} · type {result.type_code}
                        {result.sms_alert_sent && (
                          <span style={{ marginLeft: 8, color: "var(--info)" }}>
                            📱 SMS sent
                          </span>
                        )}
                      </div>
                      <div className="prob-track">
                        <div
                          className={`prob-fill ${result.prediction === 1 ? "high" : ""}`}
                          style={{
                            width: `${Math.max(2, result.fraud_probability * 100)}%`,
                          }}
                        />
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Sim */}
              {tab === "sim" && (
                <div>
                  <div
                    style={{
                      padding: "10px 14px",
                      background: "var(--bg)",
                      borderRadius: 8,
                      border: "1px solid var(--border)",
                      fontSize: 11,
                      color: "var(--text3)",
                      fontFamily: "var(--mono)",
                      marginBottom: 16,
                    }}
                  >
                    Endpoint:{" "}
                    <span style={{ color: "var(--accent)" }}>
                      {API_BASE}/predict
                    </span>
                  </div>
                  <div
                    style={{
                      display: "flex",
                      gap: 10,
                      alignItems: "center",
                      marginBottom: 16,
                      flexWrap: "wrap",
                    }}
                  >
                    <button
                      className={`btn-sim ${simRunning ? "stop" : ""}`}
                      onClick={toggleSim}
                    >
                      {simRunning ? "Stop" : "Start"}
                    </button>
                    <button
                      className="btn-reset"
                      onClick={() => {
                        setStats({ total: 0, fraud: 0, probs: [], peak: 0 });
                        setChartData([]);
                      }}
                    >
                      Reset
                    </button>
                    <select
                      value={simSpeed}
                      onChange={(e) => setSimSpeed(+e.target.value)}
                      style={{
                        padding: "8px 12px",
                        borderRadius: 8,
                        border: "1px solid var(--border)",
                        background: "var(--bg)",
                        color: "var(--text3)",
                        fontFamily: "var(--mono)",
                        fontSize: 11,
                        marginLeft: "auto",
                      }}
                    >
                      <option value={2000}>Slow (2s)</option>
                      <option value={1000}>Normal (1s)</option>
                      <option value={500}>Fast (0.5s)</option>
                    </select>
                  </div>
                  <div
                    style={{
                      fontSize: 12,
                      color: "var(--text3)",
                      fontFamily: "var(--mono)",
                      lineHeight: 1.8,
                    }}
                  >
                    Generates random Kenyan mobile money transactions (~15%
                    fraud-like) across M-Pesa, Airtel, T-Kash and Equitel. Watch
                    the chart and feed update in real time.
                  </div>
                </div>
              )}

              {/* Feed */}
              {tab === "feed" && (
                <div>
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      marginBottom: 12,
                    }}
                  >
                    <span
                      style={{
                        fontSize: 11,
                        color: "var(--text3)",
                        fontFamily: "var(--mono)",
                      }}
                    >
                      {feed.length} transactions
                    </span>
                    <button
                      className="btn-reset"
                      style={{ padding: "5px 12px", fontSize: 10 }}
                      onClick={() => setFeed([])}
                    >
                      Clear
                    </button>
                  </div>
                  <div className="feed">
                    {feed.length === 0 ? (
                      <div className="empty-state">No transactions yet</div>
                    ) : (
                      feed.map((r, i) => {
                        const p = providerOf(r.payload.type);
                        return (
                          <div
                            key={i}
                            className={`feed-item ${r.res.prediction === 1 ? "fraud" : ""}`}
                          >
                            <span
                              className={`feed-badge ${r.res.prediction === 1 ? "fraud" : "ok"}`}
                            >
                              {r.res.prediction === 1 ? "FRAUD" : "OK"}
                            </span>
                            <span className="feed-amount">
                              KES {Number(r.payload.amount).toLocaleString()}
                            </span>
                            <span className="feed-type">
                              <span className={p.cls}>{p.name}</span> ·{" "}
                              {r.payload.type} · {r.ts}
                            </span>
                            <span
                              className={`feed-prob ${r.res.fraud_probability > 0.3 ? "high" : ""}`}
                            >
                              {(r.res.fraud_probability * 100).toFixed(1)}%
                            </span>
                            {r.res.sms_alert_sent && (
                              <span className="feed-badge sms">SMS</span>
                            )}
                          </div>
                        );
                      })
                    )}
                  </div>
                </div>
              )}

              {/* SMS Log */}
              {tab === "sms" && (
                <div>
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      marginBottom: 12,
                    }}
                  >
                    <span
                      style={{
                        fontSize: 11,
                        color: "var(--text3)",
                        fontFamily: "var(--mono)",
                      }}
                    >
                      {smsLog.length} alerts sent
                    </span>
                    <button
                      className="btn-reset"
                      style={{ padding: "5px 12px", fontSize: 10 }}
                      onClick={() => setSmsLog([])}
                    >
                      Clear
                    </button>
                  </div>
                  {smsLog.length === 0 ? (
                    <div className="empty-state">
                      No SMS alerts yet — trigger a fraud detection to see
                      alerts here
                    </div>
                  ) : (
                    smsLog.map((s, i) => (
                      <div key={i} className="sms-item">
                        <div className="sms-item-header">
                          <span>📱 Alert sent · {s.ts}</span>
                          <span>
                            KES {Number(s.amount).toLocaleString()} ·{" "}
                            {(s.prob * 100).toFixed(1)}% risk
                          </span>
                        </div>
                        <div className="sms-item-body">{s.message}</div>
                      </div>
                    ))
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Right column */}
          <div className="right-col">
            {/* Chart */}
            <div className="card">
              <div className="card-header">
                <span className="card-title">Fraud Probability — Live</span>
                <span
                  style={{
                    fontSize: 11,
                    color: "var(--text3)",
                    fontFamily: "var(--mono)",
                  }}
                >
                  {chartData.length} samples
                </span>
              </div>
              <div className="card-body" style={{ paddingBottom: 8 }}>
                <ResponsiveContainer width="100%" height={180}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
                    <XAxis dataKey="idx" hide />
                    <YAxis
                      domain={[0, 100]}
                      tickFormatter={(v) => `${v}%`}
                      tick={{ fontSize: 10, fill: "var(--text3)" }}
                      width={36}
                    />
                    <Tooltip
                      formatter={(v) => [
                        `${v.toFixed(1)}%`,
                        "Fraud probability",
                      ]}
                      contentStyle={{
                        background: "var(--bg2)",
                        border: "1px solid var(--border)",
                        borderRadius: 8,
                        fontSize: 12,
                        fontFamily: "var(--mono)",
                      }}
                      labelStyle={{ display: "none" }}
                    />
                    <Line
                      type="monotone"
                      dataKey="prob"
                      stroke={chartColor}
                      strokeWidth={1.5}
                      dot={false}
                      isAnimationActive={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
                <div className="ticker">
                  <div>
                    last: <span>{lastProb.toFixed(1)}%</span>
                  </div>
                  <div>
                    peak: <span>{(stats.peak * 100).toFixed(1)}%</span>
                  </div>
                  <div>
                    total: <span>{stats.total}</span>
                  </div>
                  <div>
                    fraud: <span>{stats.fraud}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Recent feed */}
            <div className="card">
              <div className="card-header">
                <span className="card-title">Recent Transactions</span>
                <span
                  style={{
                    fontSize: 11,
                    color: "var(--text3)",
                    fontFamily: "var(--mono)",
                  }}
                >
                  {stats.fraud} fraud
                </span>
              </div>
              <div className="card-body" style={{ padding: 12 }}>
                <div className="feed">
                  {feed.length === 0 ? (
                    <div className="empty-state">Waiting for transactions…</div>
                  ) : (
                    feed.slice(0, 8).map((r, i) => {
                      const p = providerOf(r.payload.type);
                      return (
                        <div
                          key={i}
                          className={`feed-item ${r.res.prediction === 1 ? "fraud" : ""}`}
                        >
                          <span
                            className={`feed-badge ${r.res.prediction === 1 ? "fraud" : "ok"}`}
                          >
                            {r.res.prediction === 1 ? "FRAUD" : "OK"}
                          </span>
                          <span className="feed-amount">
                            KES {Number(r.payload.amount).toLocaleString()}
                          </span>
                          <span className="feed-type">
                            <span className={p.cls}>{p.name}</span> · {r.ts}
                          </span>
                          <span
                            className={`feed-prob ${r.res.fraud_probability > 0.3 ? "high" : ""}`}
                          >
                            {(r.res.fraud_probability * 100).toFixed(1)}%
                          </span>
                          {r.res.sms_alert_sent && (
                            <span className="feed-badge sms">SMS</span>
                          )}
                        </div>
                      );
                    })
                  )}
                </div>
              </div>
            </div>

            {/* Model stats */}
            <div className="card">
              <div className="card-header">
                <span className="card-title">
                  <Cpu
                    size={12}
                    style={{ marginRight: 6, verticalAlign: "middle" }}
                  />
                  Model Stats
                </span>
                <span
                  style={{
                    fontSize: 11,
                    color: health ? "var(--accent)" : "var(--danger)",
                    fontFamily: "var(--mono)",
                  }}
                >
                  {health ? "● online" : "● offline"}
                </span>
              </div>
              <div className="card-body">
                {health ? (
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "1fr 1fr",
                      gap: 12,
                    }}
                  >
                    {[
                      ["Version", health.model_version],
                      ["Status", health.status],
                      ["Threshold", `${(health.threshold * 100).toFixed(2)}%`],
                      ["Uptime", health.uptime],
                      ["Features", health.features?.length],
                      ["TX Types", health.supported_types?.length],
                    ].map(([label, value]) => (
                      <div
                        key={label}
                        style={{
                          background: "var(--bg3)",
                          borderRadius: 8,
                          padding: "10px 12px",
                        }}
                      >
                        <div
                          style={{
                            fontSize: 10,
                            textTransform: "uppercase",
                            letterSpacing: "0.08em",
                            color: "var(--text3)",
                            fontFamily: "var(--mono)",
                            marginBottom: 4,
                          }}
                        >
                          {label}
                        </div>
                        <div
                          style={{
                            fontSize: 14,
                            fontWeight: 700,
                            fontFamily: "var(--mono)",
                            color: "var(--text)",
                          }}
                        >
                          {value}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="empty-state">
                    API offline — start the server on port 5000
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
