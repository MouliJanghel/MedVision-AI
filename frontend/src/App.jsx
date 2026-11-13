
import { useEffect, useState } from "react";
import axios from "axios";
import {
  Activity,
  CheckCircle2,
  Clock,
  FileImage,
  LayoutDashboard,
  RefreshCw,
  ShieldCheck,
  Upload,
  XCircle,
  Brain,
  Eye,
} from "lucide-react";
import "./App.css";

const API = "http://127.0.0.1:8002";

function App() {
  const [health, setHealth] = useState(null);
  const [scans, setScans] = useState([]);
  const [auditLogs, setAuditLogs] = useState([]);
  const [file, setFile] = useState(null);
  const [latestResult, setLatestResult] = useState(null);

  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [reviewing, setReviewing] = useState(null);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  // Release the previous image URL when the result changes.
  useEffect(() => {
    return () => {
      if (latestResult?.imageUrl) {
        URL.revokeObjectURL(latestResult.imageUrl);
      }
    };
  }, [latestResult]);

  // Load dashboard data and synchronize the latest result.
  const loadDashboard = async () => {
    try {
      setError("");

      const [healthResponse, scansResponse, logsResponse] =
        await Promise.all([
          axios.get(`${API}/health`),
          axios.get(`${API}/scans`),
          axios.get(`${API}/audit-logs`),
        ]);

      setHealth(healthResponse.data);

      const scanData = scansResponse.data;
      const updatedScans = Array.isArray(scanData)
        ? scanData
        : scanData.scans || [];

      setScans(updatedScans);

      // Update the Latest Result panel using the latest scan history.
      setLatestResult((previous) => {
        if (!previous) return previous;

        const latestId = previous.id || previous.scan_id;

        const matchingScan = updatedScans.find(
          (scan) => String(scan.id) === String(latestId)
        );

        if (!matchingScan) return previous;

        return {
          ...previous,
          ...matchingScan,
          imageUrl: previous.imageUrl,
          gradcam_heatmap: previous.gradcam_heatmap,
        };
      });

      const logData = logsResponse.data;
      setAuditLogs(
        Array.isArray(logData)
          ? logData
          : logData.logs || logData.audit_logs || []
      );
    } catch (err) {
      console.error("Dashboard loading error:", err);
      setError(
        "Unable to load dashboard data. Check whether the backend is running."
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDashboard();
  }, []);

  // Upload and analyze a scan.
  const handleUpload = async (e) => {
    e.preventDefault();

    if (!file) {
      setError("Please select a scan image first.");
      return;
    }

    try {
      setUploading(true);
      setMessage("");
      setError("");

      const uploadedFile = file;
      const formData = new FormData();
      formData.append("file", uploadedFile);

      const response = await axios.post(
        `${API}/predict`,
        formData
      );

      const result = response.data;
      const confidence = Number(result.confidence);

      setLatestResult({
        ...result,
        imageUrl: URL.createObjectURL(uploadedFile),
      });

      setMessage(
        `Prediction: ${result.prediction || "Completed"} | Confidence: ${
          Number.isFinite(confidence)
            ? `${(
                confidence <= 1 ? confidence * 100 : confidence
              ).toFixed(2)}%`
            : "N/A"
        }`
      );

      setFile(null);

      const input = document.getElementById("scan-file");
      if (input) input.value = "";

      await loadDashboard();
    } catch (err) {
      console.error("Upload error:", err);

      setError(
        err.response?.data?.detail ||
          "Scan upload failed. Please try again."
      );
    } finally {
      setUploading(false);
    }
  };

  // Approve or reject a scan.
  const handleReview = async (scanId, decision) => {
    try {
      setReviewing(scanId);
      setMessage("");
      setError("");

      await axios.post(`${API}/scans/${scanId}/review`, {
        decision,
        reviewed_by: "Dr. Demo",
        notes: "Reviewed for project demonstration",
      });

      // Refresh history, audit logs and Latest Result status.
      await loadDashboard();

      setMessage(`Scan successfully ${decision}.`);
    } catch (err) {
      console.error("Review error:", err);

      setError(
        err.response?.data?.detail ||
          "Unable to update scan review."
      );
    } finally {
      setReviewing(null);
    }
  };

  // Dashboard statistics.
  const approvedCount = scans.filter(
    (scan) => scan.review_status === "approved"
  ).length;

  const rejectedCount = scans.filter(
    (scan) => scan.review_status === "rejected"
  ).length;

  const pendingCount = scans.filter(
    (scan) =>
      !scan.review_status ||
      scan.review_status === "pending"
  ).length;

  // Formatting helpers.
  const formatConfidence = (value) => {
    if (value === null || value === undefined) return "N/A";

    const number = Number(value);
    if (!Number.isFinite(number)) return value;

    return `${(
      number <= 1 ? number * 100 : number
    ).toFixed(2)}%`;
  };

  const formatDate = (value) => {
    if (!value) return "N/A";

    const date = new Date(value);

    return Number.isNaN(date.getTime())
      ? value
      : date.toLocaleString();
  };

  const getStatus = (scan) =>
    scan.review_status || "pending";

  return (
    <div className="app-container">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <Activity size={25} />
          </div>

          <div>
            <h2>MedVision AI</h2>
            <p>Medical Imaging System</p>
          </div>
        </div>

        <nav className="navigation">
          <a href="#dashboard" className="nav-link active">
            <LayoutDashboard size={19} />
            Dashboard
          </a>

          <a href="#upload" className="nav-link">
            <Upload size={19} />
            Scan Analysis
          </a>

          <a href="#results" className="nav-link">
            <Brain size={19} />
            AI Results
          </a>

          <a href="#history" className="nav-link">
            <FileImage size={19} />
            Scan History
          </a>

          <a href="#audit" className="nav-link">
            <ShieldCheck size={19} />
            Audit Logs
          </a>
        </nav>

        <div className="sidebar-bottom">
          <div className="backend-indicator">
            <span
              className={`status-dot ${
                health?.status === "healthy"
                  ? "online"
                  : "offline"
              }`}
            />

            <span>
              {health?.status === "healthy"
                ? "Backend Connected"
                : "Backend Offline"}
            </span>
          </div>

          <p>MedVision-AI Prototype</p>
        </div>
      </aside>

      {/* Main Dashboard */}
      <main className="main-content" id="dashboard">
        <header className="topbar">
          <div>
            <p className="eyebrow">MEDICAL AI PLATFORM</p>
            <h1>Dashboard</h1>
            <p className="subtitle">
              Monitor scans, AI predictions and review activity.
            </p>
          </div>

          <button
            className="refresh-button"
            onClick={loadDashboard}
            disabled={loading}
          >
            <RefreshCw size={17} />
            Refresh
          </button>
        </header>

        {error && (
          <div className="alert error-alert">
            <XCircle size={19} />
            {error}
          </div>
        )}

        {message && (
          <div className="alert success-alert">
            <CheckCircle2 size={19} />
            {message}
          </div>
        )}

        {loading ? (
          <div className="loading">
            <RefreshCw className="spin" size={25} />
            <p>Loading dashboard...</p>
          </div>
        ) : (
          <>
            {/* Statistics */}
            <section className="stats-grid">
              <div className="stat-card">
                <div className="stat-icon blue">
                  <FileImage size={23} />
                </div>
                <div>
                  <p>Total Scans</p>
                  <h2>{scans.length}</h2>
                </div>
              </div>

              <div className="stat-card">
                <div className="stat-icon green">
                  <CheckCircle2 size={23} />
                </div>
                <div>
                  <p>Approved</p>
                  <h2>{approvedCount}</h2>
                </div>
              </div>

              <div className="stat-card">
                <div className="stat-icon orange">
                  <Clock size={23} />
                </div>
                <div>
                  <p>Pending Review</p>
                  <h2>{pendingCount}</h2>
                </div>
              </div>

              <div className="stat-card">
                <div className="stat-icon red">
                  <XCircle size={23} />
                </div>
                <div>
                  <p>Rejected</p>
                  <h2>{rejectedCount}</h2>
                </div>
              </div>
            </section>

            {/* Upload Section */}
            <section
              className="panel upload-section"
              id="upload"
            >
              <div className="section-heading">
                <div>
                  <h2>Scan Analysis</h2>
                  <p>
                    Upload a chest X-ray image for AI-assisted analysis.
                  </p>
                </div>

                <div className="heading-icon">
                  <Upload size={22} />
                </div>
              </div>

              <form onSubmit={handleUpload}>
                <label
                  htmlFor="scan-file"
                  className="upload-area"
                >
                  <div className="upload-icon">
                    <Upload size={28} />
                  </div>

                  <h3>
                    {file ? file.name : "Choose a scan image"}
                  </h3>

                  <p>Select an image from your computer.</p>

                  <span className="choose-file">
                    Browse Files
                  </span>

                  <input
                    id="scan-file"
                    type="file"
                    accept="image/png,image/jpeg,image/jpg"
                    onChange={(e) =>
                      setFile(e.target.files?.[0] || null)
                    }
                  />
                </label>

                <button
                  type="submit"
                  className="primary-button"
                  disabled={uploading || !file}
                >
                  {uploading ? (
                    <>
                      <RefreshCw
                        className="spin"
                        size={18}
                      />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Activity size={18} />
                      Upload and Analyze
                    </>
                  )}
                </button>
              </form>
            </section>

            {/* Latest AI Results */}
            {latestResult && (
              <section
                className="panel result-section"
                id="results"
              >
                <div className="section-heading">
                  <div>
                    <h2>
                      <Brain
                        size={21}
                        className="inline-icon"
                      />
                      AI Analysis Results
                    </h2>

                    <p>
                      Prediction and Grad-CAM visualization for the
                      latest scan.
                    </p>
                  </div>

                  <span className="count-label">
                    Latest Result
                  </span>
                </div>

                <div className="result-summary">
                  <div>
                    <span className="result-label">
                      AI Prediction
                    </span>
                    <h3>
                      {latestResult.prediction || "N/A"}
                    </h3>
                  </div>

                  <div>
                    <span className="result-label">
                      Model Confidence
                    </span>
                    <h3>
                      {formatConfidence(latestResult.confidence)}
                    </h3>
                  </div>

                  <div>
                    <span className="result-label">
                      Review Status
                    </span>

                    <span
                      className={`status-badge ${
                        latestResult.review_status || "pending"
                      }`}
                    >
                      {latestResult.review_status || "pending"}
                    </span>
                  </div>
                </div>

                <div className="result-grid">
                  <div className="result-image-card">
                    <h3>
                      <Eye size={18} />
                      Original X-ray
                    </h3>

                    <img
                      src={latestResult.imageUrl}
                      alt="Uploaded chest X-ray"
                      className="result-image"
                    />

                    <p>{latestResult.filename}</p>
                  </div>

                  <div className="result-image-card">
                    <h3>
                      <Brain size={18} />
                      Grad-CAM Heatmap
                    </h3>

                    {latestResult.gradcam_heatmap ? (
                      <img
                        src={`data:image/png;base64,${latestResult.gradcam_heatmap}`}
                        alt="Grad-CAM heatmap visualization"
                        className="result-image"
                      />
                    ) : (
                      <p className="empty-state">
                        Grad-CAM visualization was not returned.
                      </p>
                    )}

                    <p>
                      Heatmap highlighting image regions that
                      influenced the model output.
                    </p>
                  </div>
                </div>

                <p className="medical-disclaimer">
                  Educational visualization only. Grad-CAM
                  highlights model attention and does not establish
                  the presence or location of disease. Predictions
                  require qualified clinical interpretation.
                </p>
              </section>
            )}

            {/* Recent Scans */}
            <section
              className="panel history"
              id="history"
            >
              <div className="section-heading">
                <div>
                  <h2>Recent Scans</h2>
                  <p>
                    Review AI predictions and doctor decisions.
                  </p>
                </div>

                <span className="count-label">
                  {scans.length} Records
                </span>
              </div>

              {scans.length === 0 ? (
                <p className="empty-state">
                  No scans available. Upload an image to get started.
                </p>
              ) : (
                <div className="table-wrapper">
                  <table>
                    <thead>
                      <tr>
                        <th>Filename</th>
                        <th>Scan Type</th>
                        <th>Prediction</th>
                        <th>Confidence</th>
                        <th>Review Status</th>
                        <th>Doctor Review</th>
                      </tr>
                    </thead>

                    <tbody>
                      {scans.map((scan) => {
                        const status = getStatus(scan);

                        return (
                          <tr key={scan.id}>
                            <td className="filename">
                              {scan.filename || "Unknown"}
                            </td>

                            <td>
                              {scan.scan_type || "N/A"}
                            </td>

                            <td>
                              {scan.prediction || "N/A"}
                            </td>

                            <td>
                              {formatConfidence(scan.confidence)}
                            </td>

                            <td>
                              <span
                                className={`status-badge ${status}`}
                              >
                                {status}
                              </span>
                            </td>

                            <td>
                              {status === "pending" ? (
                                <div className="review-actions">
                                  <button
                                    className="approve-button"
                                    disabled={
                                      reviewing === scan.id
                                    }
                                    onClick={() =>
                                      handleReview(
                                        scan.id,
                                        "approved"
                                      )
                                    }
                                  >
                                    Approve
                                  </button>

                                  <button
                                    className="reject-button"
                                    disabled={
                                      reviewing === scan.id
                                    }
                                    onClick={() =>
                                      handleReview(
                                        scan.id,
                                        "rejected"
                                      )
                                    }
                                  >
                                    Reject
                                  </button>
                                </div>
                              ) : (
                                <span className="reviewed-by">
                                  {scan.reviewed_by || "Reviewed"}
                                </span>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </section>

            {/* Audit Logs */}
            <section
              className="panel history audit-section"
              id="audit"
            >
              <div className="section-heading">
                <div>
                  <h2>
                    <ShieldCheck
                      size={21}
                      className="inline-icon"
                    />
                    Audit Log History
                  </h2>

                  <p>
                    Track scan creation and review activities.
                  </p>
                </div>

                <span className="count-label">
                  {auditLogs.length} Events
                </span>
              </div>

              {auditLogs.length === 0 ? (
                <p className="empty-state">
                  No audit records available.
                </p>
              ) : (
                <div className="table-wrapper">
                  <table>
                    <thead>
                      <tr>
                        <th>ID</th>
                        <th>Action</th>
                        <th>Scan ID</th>
                        <th>Details</th>
                        <th>Timestamp</th>
                      </tr>
                    </thead>

                    <tbody>
                      {auditLogs.map((log, index) => (
                        <tr key={log.id ?? index}>
                          <td>{log.id ?? "N/A"}</td>

                          <td>
                            <span className="audit-action">
                              {log.action ||
                                log.event_type ||
                                "Activity"}
                            </span>
                          </td>

                          <td>{log.scan_id || "N/A"}</td>

                          <td>
                            {typeof log.details === "object"
                              ? JSON.stringify(log.details)
                              : log.details || "N/A"}
                          </td>

                          <td>
                            {formatDate(
                              log.timestamp || log.created_at
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </section>

            {/* Footer */}
            <footer className="footer">
              <p>
                MedVision-AI | AI-Assisted Medical Imaging and
                Diagnosis System
              </p>

              <p>
                Educational prototype — not intended for clinical
                diagnosis.
              </p>
            </footer>
          </>
        )}
      </main>
    </div>
  );
}

export default App;
