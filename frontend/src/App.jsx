
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
  Database,
  Download,
  Server,
  FileText,
  HardDrive,
} from "lucide-react";
import "./App.css";
import heroImage from "./assets/medvision-hero.png";


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

  // CT segmentation state.
  const [ctFile, setCtFile] = useState(null);
  const [segmenting, setSegmenting] = useState(false);
  const [segmentationResult, setSegmentationResult] = useState(null);
  const [ctError, setCtError] = useState("");
  const [maskDownloading, setMaskDownloading] = useState(false);

  // Orthanc DICOM archive state.
  const [orthancHealth, setOrthancHealth] = useState(null);
  const [dicomInstances, setDicomInstances] = useState([]);
  const [selectedDicomId, setSelectedDicomId] = useState("");
  const [selectedDicom, setSelectedDicom] = useState(null);
  const [dicomLoading, setDicomLoading] = useState(false);
  const [dicomError, setDicomError] = useState("");
  const [dicomDownloading, setDicomDownloading] = useState(false);

  // Release the previous image URL when the result changes.
  useEffect(() => {
    return () => {
      if (latestResult?.imageUrl) {
        URL.revokeObjectURL(latestResult.imageUrl);
      }
    };
  }, [latestResult]);

  // Load Orthanc connection and instance list.
  const loadOrthanc = async () => {
    setDicomLoading(true);
    setDicomError("");

    try {
      const [healthResponse, instancesResponse] = await Promise.all([
        axios.get(`${API}/orthanc/health`),
        axios.get(`${API}/orthanc/instances`),
      ]);

      setOrthancHealth(healthResponse.data);

      const instanceData = instancesResponse.data;
      const ids = Array.isArray(instanceData)
        ? instanceData
        : instanceData.instances || [];

      setDicomInstances(ids);

      setSelectedDicomId((current) => {
        if (current && ids.includes(current)) return current;
        return ids[0] || "";
      });
    } catch (err) {
      console.error("Orthanc loading error:", err);
      setOrthancHealth(null);
      setDicomError(
        err.response?.data?.detail ||
          "Unable to connect to Orthanc. Check whether the Orthanc container is running."
      );
    } finally {
      setDicomLoading(false);
    }
  };

  // Load metadata whenever a DICOM instance is selected.
  useEffect(() => {
    if (!selectedDicomId) {
      setSelectedDicom(null);
      return;
    }

    let cancelled = false;

    const loadDicomMetadata = async () => {
      setSelectedDicom(null);
      setDicomError("");

      try {
        const response = await axios.get(
          `${API}/orthanc/instances/${selectedDicomId}`
        );

        if (!cancelled) {
          setSelectedDicom(response.data);
        }
      } catch (err) {
        console.error("DICOM metadata error:", err);

        if (!cancelled) {
          setDicomError(
            err.response?.data?.detail ||
              "Unable to retrieve DICOM metadata."
          );
        }
      }
    };

    loadDicomMetadata();

    return () => {
      cancelled = true;
    };
  }, [selectedDicomId]);

  // Download a DICOM file through the FastAPI endpoint.
  const handleDicomDownload = async () => {
    if (!selectedDicomId) return;

    try {
      setDicomDownloading(true);
      setDicomError("");

      const response = await axios.get(
        `${API}/orthanc/instances/${selectedDicomId}/file`,
        { responseType: "blob" }
      );

      const blobUrl = URL.createObjectURL(
        new Blob([response.data], { type: "application/dicom" })
      );

      const link = document.createElement("a");
      link.href = blobUrl;
      link.download = `${selectedDicomId}.dcm`;

      document.body.appendChild(link);
      link.click();
      link.remove();

      setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);
      setMessage("DICOM file download started.");
    } catch (err) {
      console.error("DICOM download error:", err);
      setDicomError(
        err.response?.data?.detail ||
          "Unable to download the DICOM file."
      );
    } finally {
      setDicomDownloading(false);
    }
  };

  // Download the generated CT segmentation mask.
  const handleMaskDownload = async () => {
    if (!segmentationResult?.mask_id) {
      setCtError("No segmentation mask is available to download.");
      return;
    }

    try {
      setMaskDownloading(true);
      setCtError("");

      const response = await axios.get(
        `${API}/download-mask/${segmentationResult.mask_id}`,
        { responseType: "blob" }
      );

      const blobUrl = URL.createObjectURL(
        new Blob([response.data], { type: "application/gzip" })
      );

      const link = document.createElement("a");
      link.href = blobUrl;
      link.download = "predicted_spleen_mask.nii.gz";

      document.body.appendChild(link);
      link.click();
      link.remove();

      setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);
      setMessage("Segmentation mask download started.");
    } catch (err) {
      console.error("Mask download error:", err);

      setCtError(
        err.response?.data?.detail ||
          "Unable to download the segmentation mask. Please try again."
      );
    } finally {
      setMaskDownloading(false);
    }
  };

  // Load dashboard data.
  const loadDashboard = async () => {
    loadOrthanc();

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

  const handleRefresh = async () => {
    setLoading(true);
    await loadDashboard();
  };

  // Upload and analyze a chest X-ray.
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

  // Upload a CT scan and run 3D U-Net segmentation.
  const handleCTSegmentation = async (e) => {
    e.preventDefault();

    if (!ctFile) {
      setCtError("Please select a CT scan first.");
      return;
    }

    const filename = ctFile.name.toLowerCase();

    if (!filename.endsWith(".nii") && !filename.endsWith(".nii.gz")) {
      setCtError("Please upload a .nii or .nii.gz file.");
      return;
    }

    try {
      setSegmenting(true);
      setCtError("");
      setMessage("");
      setSegmentationResult(null);

      const formData = new FormData();
      formData.append("file", ctFile);

      const response = await axios.post(
        `${API}/segment`,
        formData
      );

      setSegmentationResult(response.data);
      setMessage("CT segmentation completed successfully.");

      setCtFile(null);

      const input = document.getElementById("ct-file");
      if (input) input.value = "";
    } catch (err) {
      console.error("CT segmentation error:", err);

      setCtError(
        err.response?.data?.detail ||
          "CT segmentation failed. Please try again."
      );
    } finally {
      setSegmenting(false);
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
      !scan.review_status || scan.review_status === "pending"
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

  const formatDicomDate = (value) => {
    if (!value) return "N/A";

    if (/^\d{8}$/.test(String(value))) {
      return `${value.slice(6, 8)}-${value.slice(4, 6)}-${value.slice(0, 4)}`;
    }

    return value;
  };

  const getStatus = (scan) => scan.review_status || "pending";

  const dicomMetadata = selectedDicom?.metadata || {};

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

          <a href="#dicom" className="nav-link">
            <Database size={19} />
            DICOM Archive
          </a>

          <a href="#ct-segmentation" className="nav-link">
            <Brain size={19} />
            CT Segmentation
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
                health?.status === "healthy" ? "online" : "offline"
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
            onClick={handleRefresh}
            disabled={loading}
          >
            <RefreshCw size={17} />
            Refresh
          </button>
        </header>
                {/* Medical AI Hero */}
        <section className="hero-section">
          <div className="hero-content">
            <span className="hero-label">
              <Activity size={15} />
              INTELLIGENT MEDICAL IMAGING
            </span>

            <h2>
              See Beyond the Scan.
              <span> Understand with AI.</span>
            </h2>

            <p>
              Explore AI-assisted chest X-ray analysis, 3D CT
              segmentation and medical imaging workflows in one
              integrated platform.
            </p>

            <div className="hero-features">
              <span><CheckCircle2 size={16} /> X-ray Analysis</span>
              <span><CheckCircle2 size={16} /> 3D Segmentation</span>
              <span><CheckCircle2 size={16} /> DICOM Archive</span>
            </div>

            <a href="#upload" className="hero-button">
              Start Scan Analysis <Upload size={17} />
            </a>
          </div>

          <div className="hero-visual">
            <img
              src={heroImage}
              alt="Medical AI visualization showing a holographic chest and lungs"
            />
          </div>
        </section>

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

            {/* X-ray Upload Section */}
            <section className="panel upload-section" id="upload">
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
                <label htmlFor="scan-file" className="upload-area">
                  <div className="upload-icon">
                    <Upload size={28} />
                  </div>

                  <h3>
                    {file ? file.name : "Choose a scan image"}
                  </h3>

                  <p>Select an image from your computer.</p>

                  <span className="choose-file">Browse Files</span>

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
                      <RefreshCw className="spin" size={18} />
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

            {/* CT Segmentation */}
            <section
              className="panel ct-section"
              id="ct-segmentation"
            >
              <div className="section-heading">
                <div>
                  <h2>
                    <Brain size={21} className="inline-icon" />
                    3D CT Segmentation
                  </h2>

                  <p>
                    Upload a NIfTI CT scan for experimental spleen
                    segmentation using the 3D U-Net model.
                  </p>
                </div>

                <div className="heading-icon">
                  <FileImage size={22} />
                </div>
              </div>

              {ctError && (
                <div className="alert error-alert">
                  <XCircle size={18} />
                  {ctError}
                </div>
              )}

              <form onSubmit={handleCTSegmentation}>
                <label htmlFor="ct-file" className="upload-area">
                  <div className="upload-icon">
                    <Upload size={28} />
                  </div>

                  <h3>
                    {ctFile ? ctFile.name : "Choose a CT scan"}
                  </h3>

                  <p>
                    Supported formats: NIfTI (.nii, .nii.gz)
                  </p>

                  <span className="choose-file">
                    Browse CT Files
                  </span>

                  <input
                    id="ct-file"
                    type="file"
                    accept=".nii,.nii.gz"
                    onChange={(e) => {
                      setCtFile(e.target.files?.[0] || null);
                      setCtError("");
                    }}
                  />
                </label>

                <button
                  type="submit"
                  className="primary-button"
                  disabled={segmenting || !ctFile}
                >
                  {segmenting ? (
                    <>
                      <RefreshCw className="spin" size={18} />
                      Running 3D Segmentation...
                    </>
                  ) : (
                    <>
                      <Brain size={18} />
                      Run CT Segmentation
                    </>
                  )}
                </button>
              </form>

              {segmentationResult && (
                <div className="ct-result">
                  <div className="ct-result-heading">
                    <div>
                      <h3>
                        <CheckCircle2 size={19} />
                        Segmentation Completed
                      </h3>

                      <p>{segmentationResult.filename}</p>
                    </div>

                    <span className="ct-completed-badge">
                      Completed
                    </span>
                  </div>

                  <div className="result-summary">
                    <div>
                      <span className="result-label">
                        Segmented Region
                      </span>
                      <h3>
                        {segmentationResult.segmented_region || "Spleen"}
                      </h3>
                    </div>

                    <div>
                      <span className="result-label">
                        Original CT Dimensions
                      </span>
                      <h3>
                        {segmentationResult.original_ct_shape?.join(" × ") ||
                          "N/A"}
                      </h3>
                    </div>

                    <div>
                      <span className="result-label">
                        Preview Slice
                      </span>
                      <h3>
                        {segmentationResult.preview_slice_index ?? "N/A"}
                      </h3>
                    </div>
                  </div>

                  <div className="ct-preview-card">
                    <h3>
                      <Eye size={19} />
                      CT Segmentation Overlay
                    </h3>

                    {segmentationResult.preview_slice_base64 ? (
                      <img
                        src={`data:image/png;base64,${segmentationResult.preview_slice_base64}`}
                        alt="CT slice with predicted spleen segmentation overlay"
                        className="ct-preview-image"
                      />
                    ) : (
                      <p className="empty-state">
                        No preview image was returned.
                      </p>
                    )}

                    <p className="ct-preview-caption">
                      Axial CT slice with the predicted spleen region
                      highlighted in red.
                    </p>
                  </div>

                  <div className="ct-mask-details">
                    <div>
                      <span>Foreground Voxels</span>
                      <strong>
                        {Number(
                          segmentationResult.foreground_voxels || 0
                        ).toLocaleString()}
                      </strong>
                    </div>

                    <div>
                      <span>Mask Format</span>
                      <strong>
                        {segmentationResult.mask_file_format || "NIfTI"}
                      </strong>
                    </div>

                    <div>
                      <span>Restored Mask Dimensions</span>
                      <strong>
                        {segmentationResult.restored_mask_shape?.join(" × ") ||
                          "N/A"}
                      </strong>
                    </div>
                  </div>

                  {segmentationResult.mask_id && (
                    <button
                      type="button"
                      className="dicom-download-button ct-download-button"
                      onClick={handleMaskDownload}
                      disabled={maskDownloading}
                    >
                      {maskDownloading ? (
                        <RefreshCw size={17} className="spin" />
                      ) : (
                        <Download size={17} />
                      )}
                      {maskDownloading
                        ? "Downloading Mask..."
                        : "Download Segmentation Mask"}
                    </button>
                  )}

                  <p className="medical-disclaimer">
                    Experimental AI output only. The 3D U-Net model has
                    limited validation performance. Segmentation may be
                    inaccurate and must not be used for clinical diagnosis
                    or treatment decisions.
                  </p>
                </div>
              )}
            </section>

            {/* Orthanc DICOM Archive */}
            <section className="panel dicom-section" id="dicom">
              <div className="section-heading">
                <div>
                  <h2>
                    <Database size={21} className="inline-icon" />
                    DICOM Archive
                  </h2>
                  <p>
                    Browse medical imaging instances stored in Orthanc.
                  </p>
                </div>

                <button
                  className="refresh-button"
                  onClick={loadOrthanc}
                  disabled={dicomLoading}
                >
                  <RefreshCw
                    size={15}
                    className={dicomLoading ? "spin" : ""}
                  />
                  Refresh DICOM
                </button>
              </div>

              <div className="orthanc-status">
                <div className="orthanc-status-icon">
                  <Server size={22} />
                </div>

                <div className="orthanc-status-text">
                  <strong>Orthanc PACS Server</strong>
                  <span>
                    {orthancHealth?.orthanc_name || "Orthanc"}
                    {orthancHealth?.orthanc_version
                      ? ` · Version ${orthancHealth.orthanc_version}`
                      : ""}
                  </span>
                </div>

                <span
                  className={`connection-badge ${
                    orthancHealth?.status === "connected"
                      ? "connected"
                      : "disconnected"
                  }`}
                >
                  <span className="status-dot" />
                  {orthancHealth?.status === "connected"
                    ? "Connected"
                    : "Disconnected"}
                </span>
              </div>

              {dicomError && (
                <div className="dicom-error">
                  <XCircle size={17} />
                  <span>{dicomError}</span>
                </div>
              )}

              <div className="dicom-count-row">
                <div>
                  <span className="result-label">
                    Stored DICOM Instances
                  </span>
                  <h3>{dicomInstances.length}</h3>
                </div>

                <div className="dicom-count-icon">
                  <HardDrive size={23} />
                </div>
              </div>

              {dicomLoading && dicomInstances.length === 0 ? (
                <div className="empty-state">
                  <RefreshCw className="spin" size={22} />
                  <p>Loading DICOM archive...</p>
                </div>
              ) : dicomInstances.length === 0 ? (
                <div className="dicom-empty">
                  <Database size={30} />
                  <h3>No DICOM instances found</h3>
                  <p>
                    Upload a DICOM file to Orthanc to see it listed here.
                  </p>
                </div>
              ) : (
                <div className="dicom-layout">
                  <div className="dicom-instance-list">
                    <h3>Available Instances</h3>

                    {dicomInstances.map((instanceId, index) => (
                      <button
                        key={instanceId}
                        className={`dicom-instance ${
                          selectedDicomId === instanceId
                            ? "selected"
                            : ""
                        }`}
                        onClick={() => setSelectedDicomId(instanceId)}
                      >
                        <div className="dicom-instance-icon">
                          <FileText size={19} />
                        </div>

                        <div className="dicom-instance-info">
                          <strong>Instance {index + 1}</strong>
                          <span>{instanceId}</span>
                        </div>

                        {selectedDicomId === instanceId && (
                          <CheckCircle2
                            size={17}
                            className="dicom-selected-icon"
                          />
                        )}
                      </button>
                    ))}
                  </div>

                  <div className="dicom-details">
                    <div className="dicom-details-heading">
                      <div>
                        <h3>Instance Details</h3>
                        <p>
                          Metadata retrieved from the Orthanc server.
                        </p>
                      </div>

                      {selectedDicom && (
                        <button
                          className="dicom-download-button"
                          onClick={handleDicomDownload}
                          disabled={dicomDownloading}
                        >
                          {dicomDownloading ? (
                            <RefreshCw size={16} className="spin" />
                          ) : (
                            <Download size={16} />
                          )}
                          {dicomDownloading
                            ? "Downloading..."
                            : "Download DICOM"}
                        </button>
                      )}
                    </div>

                    {!selectedDicom ? (
                      <div className="empty-state">
                        <RefreshCw className="spin" size={21} />
                        <p>Retrieving instance metadata...</p>
                      </div>
                    ) : (
                      <>
                        <div className="dicom-metadata-grid">
                          <div className="metadata-item">
                            <span>Patient ID</span>
                            <strong>
                              {dicomMetadata.PatientID || "N/A"}
                            </strong>
                          </div>

                          <div className="metadata-item">
                            <span>Patient Name</span>
                            <strong>
                              {dicomMetadata.PatientName || "N/A"}
                            </strong>
                          </div>

                          <div className="metadata-item">
                            <span>Modality</span>
                            <strong>
                              {dicomMetadata.Modality || "N/A"}
                            </strong>
                          </div>

                          <div className="metadata-item">
                            <span>Study Date</span>
                            <strong>
                              {formatDicomDate(dicomMetadata.StudyDate)}
                            </strong>
                          </div>

                          <div className="metadata-item">
                            <span>Image Dimensions</span>
                            <strong>
                              {dicomMetadata.Rows && dicomMetadata.Columns
                                ? `${dicomMetadata.Rows} × ${dicomMetadata.Columns}`
                                : "N/A"}
                            </strong>
                          </div>

                          <div className="metadata-item">
                            <span>Instance Number</span>
                            <strong>
                              {dicomMetadata.InstanceNumber || "N/A"}
                            </strong>
                          </div>
                        </div>

                        <div className="dicom-identifier">
                          <span>Orthanc Instance ID</span>
                          <code>{selectedDicom.instance_id}</code>
                        </div>

                        <div className="dicom-identifier">
                          <span>Study Instance UID</span>
                          <code>
                            {dicomMetadata.StudyInstanceUID || "N/A"}
                          </code>
                        </div>

                        <div className="dicom-identifier">
                          <span>Series Instance UID</span>
                          <code>
                            {dicomMetadata.SeriesInstanceUID || "N/A"}
                          </code>
                        </div>
                      </>
                    )}
                  </div>
                </div>
              )}

              <p className="dicom-note">
                DICOM archive integration: instances and metadata are
                retrieved from Orthanc through the FastAPI backend.
                This section currently supports metadata inspection and
                file download, not DICOM image viewing or AI inference.
              </p>
            </section>

            {/* Latest AI Results */}
            {latestResult && (
              <section className="panel result-section" id="results">
                <div className="section-heading">
                  <div>
                    <h2>
                      <Brain size={21} className="inline-icon" />
                      AI Analysis Results
                    </h2>

                    <p>
                      Prediction and Grad-CAM visualization for the latest scan.
                    </p>
                  </div>

                  <span className="count-label">Latest Result</span>
                </div>

                <div className="result-summary">
                  <div>
                    <span className="result-label">AI Prediction</span>
                    <h3>{latestResult.prediction || "N/A"}</h3>
                  </div>

                  <div>
                    <span className="result-label">Model Confidence</span>
                    <h3>
                      {formatConfidence(latestResult.confidence)}
                    </h3>
                  </div>

                  <div>
                    <span className="result-label">Review Status</span>
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
                      Heatmap highlighting image regions that influenced
                      the model output.
                    </p>
                  </div>
                </div>

                <p className="medical-disclaimer">
                  Educational visualization only. Grad-CAM highlights
                  model attention and does not establish the presence or
                  location of disease. Predictions require qualified
                  clinical interpretation.
                </p>
              </section>
            )}

            {/* Recent Scans */}
            <section className="panel history" id="history">
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

                            <td>{scan.scan_type || "N/A"}</td>
                            <td>{scan.prediction || "N/A"}</td>
                            <td>{formatConfidence(scan.confidence)}</td>

                            <td>
                              <span className={`status-badge ${status}`}>
                                {status}
                              </span>
                            </td>

                            <td>
                              {status === "pending" ? (
                                <div className="review-actions">
                                  <button
                                    className="approve-button"
                                    disabled={reviewing === scan.id}
                                    onClick={() =>
                                      handleReview(scan.id, "approved")
                                    }
                                  >
                                    Approve
                                  </button>

                                  <button
                                    className="reject-button"
                                    disabled={reviewing === scan.id}
                                    onClick={() =>
                                      handleReview(scan.id, "rejected")
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
                    <ShieldCheck size={21} className="inline-icon" />
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
                              {log.action || log.event_type || "Activity"}
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
                MedVision-AI | AI-Assisted Medical Imaging and Diagnosis System
              </p>

              <p>
                Educational prototype — not intended for clinical diagnosis.
              </p>
            </footer>
          </>
        )}
      </main>
    </div>
  );
}

export default App;
