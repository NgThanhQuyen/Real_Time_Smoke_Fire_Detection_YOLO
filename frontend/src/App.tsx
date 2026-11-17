import { useEffect, useRef, useState } from "react";
import { api, className, formatDate, formatVideoTime, json } from "./api";
import AnnotationEditor from "./AnnotationEditor";
import type { AnnotationBox } from "./AnnotationEditor";
import type { AdminEvent, AdminJob, DetectedFrame, Detection, Feedback, ModelVersion, Profile, RealtimeSession, User, VideoJob } from "./types";

type Area = "videos" | "realtime" | "profile" | "events" | "jobs" | "accounts" | "versions" | "reviews" | "candidates";

function homeArea(user: User): Area {
  return user.roles.includes("admin_ops") ? "events" : user.roles.includes("admin_model") ? "reviews" : "videos";
}

function Brand() {
  return (
    <div className="brand" aria-label="FLARE - Fire Location And Recognition Engine">
      <div className="brand-icon-wrap" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z" />
        </svg>
      </div>
      <div className="brand-text">
        <strong className="brand-name">FLARE</strong>
        <span className="brand-tagline">Fire Location &amp; Recognition Engine</span>
      </div>
    </div>
  );
}

const jobLabels: Record<VideoJob["status"], string> = {
  queued: "Đang chờ",
  processing: "Đang phân tích",
  finalizing: "Đang xuất video",
  completed: "Hoàn tất",
  failed: "Thất bại",
  interrupted: "Gián đoạn",
};

const feedbackLabels: Record<string, string> = {
  false_positive: "Phát hiện nhầm",
  false_negative: "Bỏ sót",
  wrong_class: "Sai loại",
  wrong_box: "Sai vị trí box",
};

const modelLabels: Record<string, string> = {
  active: "Đang sử dụng",
  candidate: "Ứng viên",
  retired: "Đã thay thế",
};

function useMessage() {
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  return {
    message,
    error,
    success(text: string) {
      setMessage(text);
      setError("");
    },
    fail(reason: unknown) {
      setError(reason instanceof Error ? reason.message : "Đã xảy ra lỗi");
      setMessage("");
    },
    clear() {
      setMessage("");
      setError("");
    },
  };
}

function Notice({ message, error }: { message: string; error: string }) {
  if (!message && !error) return null;
  return <div className={error ? "notice error" : "notice success"} role="status">{error || message}</div>;
}

function AuthView({ onLogin }: { onLogin: (user: User) => void }) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [passwordConfirm, setPasswordConfirm] = useState("");
  const [busy, setBusy] = useState(false);
  const notice = useMessage();

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    notice.clear();
    try {
      if (mode === "register" && password !== passwordConfirm) throw new Error("Mật khẩu xác nhận không trùng khớp.");
      const user = await api<User>(`/api/auth/${mode}`, json("POST", mode === "register" ? { username, password, password_confirm: passwordConfirm } : { username, password }));
      onLogin(user);
    } catch (reason) {
      notice.fail(reason);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="auth-shell">
      <section className="auth-intro">
        <Brand />
        <div className="auth-copy">
          <p className="auth-english">Real-Time Fire &amp; Smoke Detection System</p>
          <h1>Hệ thống cảnh báo hỏa hoạn thời gian thực</h1>
          <p className="auth-description">Phân tích video tải lên hoặc bật webcam laptop để xem luồng hình có box dự đoán trực tiếp trên trình duyệt.</p>
          <div className="auth-stats-bar">
            <div className="auth-stat-item">
              <span className="auth-stat-val">YOLOv11</span>
              <span className="auth-stat-lbl">Kiến trúc AI</span>
            </div>
            <div className="auth-stat-item">
              <span className="auth-stat-val">&lt; 50ms</span>
              <span className="auth-stat-lbl">Độ trễ xử lý</span>
            </div>
            <div className="auth-stat-item">
              <span className="auth-stat-val">24/7</span>
              <span className="auth-stat-lbl">Giám sát liên tục</span>
            </div>
          </div>
        </div>
        <div className="small muted">FLARE Safety Engine v1.0 · Bảo mật &amp; Kiểm soát hỏa hoạn</div>
      </section>
      <section className="auth-form-wrap">
        <div className="auth-form-card">
          <p className="eyebrow">Tài khoản</p>
          <h2>{mode === "login" ? "Đăng nhập" : "Tạo tài khoản"}</h2>
          <p className="muted">{mode === "login" ? "Tiếp tục công việc đang theo dõi." : "Tài khoản mới được cấp quyền người dùng."}</p>
          <form onSubmit={submit}>
            <label>Tên tài khoản<input autoComplete="username" value={username} onChange={(event) => setUsername(event.target.value)} required minLength={3} placeholder="Nhập tên tài khoản" /></label>
            <label>Mật khẩu<input type="password" autoComplete={mode === "login" ? "current-password" : "new-password"} value={password} onChange={(event) => setPassword(event.target.value)} required minLength={mode === "register" ? 10 : undefined} placeholder="••••••••••••" /></label>
            {mode === "register" && <label>Xác nhận mật khẩu<input type="password" autoComplete="new-password" value={passwordConfirm} onChange={(event) => setPasswordConfirm(event.target.value)} required minLength={10} placeholder="••••••••••••" /></label>}
            <Notice message={notice.message} error={notice.error} />
            <button className="button primary full" disabled={busy}>{busy ? "Đang xử lý..." : mode === "login" ? "Đăng nhập" : "Đăng ký"}</button>
          </form>
          <button className="text-button" onClick={() => { setMode(mode === "login" ? "register" : "login"); setPasswordConfirm(""); notice.clear(); }}>
            {mode === "login" ? "Chưa có tài khoản? Đăng ký ngay" : "Đã có tài khoản? Đăng nhập"}
          </button>
        </div>
      </section>
    </main>
  );
}

function Progress({ job }: { job: VideoJob }) {
  const value = job.total_frames ? Math.min(100, Math.round(job.processed_frames / job.total_frames * 100)) : 0;
  const labels: Record<string, string> = {
    queued: "Đang chờ xử lý",
    processing: "Đang phân tích video",
    finalizing: "Đang xuất video kết quả",
    completed: "Hoàn tất",
    failed: "Xử lý thất bại",
    interrupted: "Tác vụ bị gián đoạn",
  };
  return (
    <div className="progress-wrap">
      <div className="progress-label"><strong>{labels[job.status]}</strong><span>{job.status === "processing" ? `${value}%` : ""}</span></div>
      <div className="progress-track"><div style={{ width: `${job.status === "completed" ? 100 : value}%` }} /></div>
    </div>
  );
}

function VideoDetail({ jobId, onBack }: { jobId: string; onBack: () => void }) {
  const [job, setJob] = useState<VideoJob | null>(null);
  const [currentMs, setCurrentMs] = useState(0);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [frames, setFrames] = useState<DetectedFrame[]>([]);
  const [feedbackOpen, setFeedbackOpen] = useState(false);
  const [feedbackType, setFeedbackType] = useState("false_positive");
  const [selectedDetection, setSelectedDetection] = useState("");
  const [expectedClass, setExpectedClass] = useState(0);
  const [comment, setComment] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const notice = useMessage();

  useEffect(() => {
    let active = true;
    async function refresh() {
      try {
        const value = await api<VideoJob>(`/api/videos/${jobId}`);
        if (active) setJob(value);
      } catch (reason) {
        if (active) notice.fail(reason);
      }
    }
    void refresh();
    const timer = window.setInterval(() => {
      if (active) void refresh();
    }, 2500);
    return () => { active = false; window.clearInterval(timer); };
  }, [jobId]);

  useEffect(() => {
    if (job?.status !== "completed") return;
    api<DetectedFrame[]>(`/api/videos/${jobId}/detected-frames`).then(setFrames).catch(notice.fail);
  }, [jobId, job?.status]);

  async function openFeedback() {
    if (!job) return;
    videoRef.current?.pause();
    const at = Math.round((videoRef.current?.currentTime || 0) * 1000);
    setCurrentMs(at);
    setFeedbackOpen(true);
    setSelectedDetection("");
    const start = Math.max(0, at - 100);
    try {
      const values = await api<Detection[]>(`/api/videos/${job.id}/detections?start_ms=${start}&end_ms=${at + 100}`);
      const nearest = values.length ? Math.min(...values.map((item) => Math.abs(item.video_time_ms - at))) : 999999;
      setDetections(values.filter((item) => Math.abs(item.video_time_ms - at) === nearest));
    } catch (reason) {
      notice.fail(reason);
    }
  }

  async function sendFeedback(event: React.FormEvent) {
    event.preventDefault();
    if (!job) return;
    setSubmitting(true);
    try {
      await api(`/api/videos/${job.id}/feedback`, json("POST", {
        video_time_ms: currentMs,
        type: feedbackType,
        detection_id: feedbackType === "false_negative" ? null : selectedDetection,
        expected_class: feedbackType === "false_negative" || feedbackType === "wrong_class" ? expectedClass : null,
        comment,
      }));
      notice.success("Đã gửi phản hồi cho quản trị mô hình.");
      setFeedbackOpen(false);
      setComment("");
    } catch (reason) {
      notice.fail(reason);
    } finally {
      setSubmitting(false);
    }
  }

  async function retry() {
    try {
      const updated = await api<VideoJob>(`/api/videos/${jobId}/retry`, { method: "POST" });
      setJob(updated);
      notice.success("Đã đưa tác vụ vào hàng chờ xử lý lại.");
    } catch (reason) {
      notice.fail(reason);
    }
  }

  if (!job) return <div className="page"><button className="text-button" onClick={onBack}>Quay lại lịch sử</button><Notice message={notice.message} error={notice.error} /><p>Đang tải tác vụ...</p></div>;
  return (
    <div className="page">
      <button className="text-button back" onClick={onBack}>Quay lại lịch sử</button>
      <div className="page-header"><div><p className="eyebrow">Chi tiết tác vụ</p><h1>{job.original_name}</h1><p className="muted">Tải lên {formatDate(job.created_at)} · Video đã tải lên, không phải camera trực tiếp</p></div><span className={`status ${job.status}`}>{jobLabels[job.status]}</span></div>
      <Notice message={notice.message} error={notice.error} />
      {job.status !== "completed" ? (
        <section className="work-surface">
          <Progress job={job} />
          {job.error_code && <p className="muted">Mã lỗi: {job.error_code}</p>}
          {(job.status === "failed" || job.status === "interrupted") && <button className="button secondary" onClick={retry}>Chạy lại tác vụ</button>}
          <p className="muted">Bạn có thể rời trang. Tác vụ tiếp tục xử lý ở backend.</p>
        </section>
      ) : (
        <>
          <div className="detail-layout">
            <section className="player-section">
              <video ref={videoRef} controls preload="metadata" src={job.video_url || undefined} onTimeUpdate={(event) => setCurrentMs(Math.round(event.currentTarget.currentTime * 1000))} />
              <div className="player-action"><span>Khung hiện tại: {formatVideoTime(currentMs)}</span><button className="button secondary" onClick={openFeedback}>Báo dự đoán sai tại đây</button></div>
            </section>
            <aside className="detail-side">
              <h2>Tóm tắt xử lý</h2>
              <dl className="metric-list">
                <div><dt>Thời lượng video</dt><dd>{formatVideoTime(job.duration_ms)}</dd></div>
                <div><dt>Khung đã xử lý</dt><dd>{job.processed_frames.toLocaleString("vi-VN")}</dd></div>
                <div><dt>Khung có phát hiện</dt><dd>{job.detected_frames.toLocaleString("vi-VN")}</dd></div>
                <div><dt>Sự kiện</dt><dd>{job.events_count}</dd></div>
                <div><dt>Thời gian xử lý</dt><dd>{job.processing_ms ? (job.processing_ms / 1000).toFixed(1) + " giây" : "Chưa có"}</dd></div>
                <div><dt>Tốc độ xử lý</dt><dd>{job.processing_ms ? (job.processed_frames / (job.processing_ms / 1000)).toFixed(1) + " khung/giây" : "Chưa có"}</dd></div>
              </dl>
              <p className="small muted">Các số liệu này mô tả quá trình xử lý, không phải Precision, Recall hay mAP.</p>
            </aside>
          </div>
          <section className="section-block">
            <div className="section-heading"><div><p className="eyebrow">Dự đoán từng khung</p><h2>Khung hình có box</h2></div><p className="muted">Chọn ảnh để tua video và phản hồi dự đoán sai. Tối đa 48 ảnh, lấy mẫu theo mỗi nửa giây.</p></div>
            {frames.length ? <div className="evidence-grid">
              {frames.map((frame) => <button key={frame.frame_index} className="evidence-item" onClick={() => { if (videoRef.current) { videoRef.current.currentTime = frame.video_time_ms / 1000; videoRef.current.pause(); } setCurrentMs(frame.video_time_ms); }}>
                <img src={frame.image_url} alt={`Khung ${frame.frame_index} có dự đoán`} loading="lazy" />
                <span><strong>Khung {frame.frame_index}</strong><small>{formatVideoTime(frame.video_time_ms)} · {Math.round(frame.confidence * 100)}%</small></span>
              </button>)}
            </div> : <p className="empty-state">Không có box nào trong video. Bạn vẫn có thể tạm dừng video và báo trường hợp bị bỏ sót.</p>}
          </section>
          <section className="section-block">
            <div className="section-heading"><div><p className="eyebrow">Bằng chứng</p><h2>Mốc phát hiện</h2></div><p className="muted">Chọn ảnh để tua video tới đúng thời điểm.</p></div>
            {job.events?.length ? <div className="evidence-grid">
              {job.events.map((item) => <button key={item.id} className="evidence-item" onClick={() => { if (videoRef.current) videoRef.current.currentTime = item.start_ms / 1000; }}>
                {item.evidence_id && <img src={`/api/videos/${job.id}/evidence/${item.evidence_id}`} alt={`Bằng chứng ${className(item.class_id)}`} />}
                <span><strong>{className(item.class_id)}</strong><small>{formatVideoTime(item.start_ms)} – {formatVideoTime(item.end_ms)} · {Math.round(item.peak_confidence * 100)}%</small></span>
              </button>)}
            </div> : <p className="empty-state">Không có sự kiện lửa hoặc khói được chốt trong video này.</p>}
          </section>
          {feedbackOpen && <div className="modal-backdrop" onClick={() => setFeedbackOpen(false)}><section className="modal" onClick={(event) => event.stopPropagation()}>
            <div className="modal-head"><div><p className="eyebrow">Phản hồi dự đoán</p><h2>Khung {formatVideoTime(currentMs)}</h2></div><button className="text-button" onClick={() => setFeedbackOpen(false)}>Đóng</button></div>
            <form onSubmit={sendFeedback}>
              <label>Vấn đề
                <select value={feedbackType} onChange={(event) => setFeedbackType(event.target.value)}>
                  <option value="false_positive">Phát hiện nhầm</option><option value="false_negative">Bỏ sót lửa hoặc khói</option>
                  <option value="wrong_class">Sai loại</option><option value="wrong_box">Sai vị trí box</option>
                </select>
              </label>
              {feedbackType !== "false_negative" && <label>Box cần phản hồi
                <select value={selectedDetection} onChange={(event) => setSelectedDetection(event.target.value)} required>
                  <option value="">Chọn box ở khung này</option>
                  {detections.map((item) => <option key={item.id} value={item.id}>{className(item.class_id)} · {Math.round(item.confidence * 100)}% · khung {item.frame_index}</option>)}
                </select>
                {!detections.length && <span className="field-note">Không có box gần thời điểm này. Hãy chọn “Bỏ sót lửa hoặc khói” hoặc tua tới khung có box.</span>}
              </label>}
              {(feedbackType === "false_negative" || feedbackType === "wrong_class") && <label>Loại đúng
                <select value={expectedClass} onChange={(event) => setExpectedClass(Number(event.target.value))}><option value={0}>Khói</option><option value={1}>Lửa</option></select>
              </label>}
              <label>Mô tả thêm<textarea value={comment} onChange={(event) => setComment(event.target.value)} rows={3} placeholder="Ví dụ: đây là mây, không phải khói." /></label>
              <p className="small muted">Quản trị mô hình sẽ xem ảnh gốc và kiểm tra lại trước khi dùng phản hồi cho huấn luyện.</p>
              <button className="button primary" disabled={submitting || (feedbackType !== "false_negative" && !selectedDetection)}>{submitting ? "Đang gửi..." : "Gửi phản hồi"}</button>
            </form>
          </section></div>}
        </>
      )}
    </div>
  );
}

function VideosView() {
  const [jobs, setJobs] = useState<VideoJob[]>([]);
  const [feedback, setFeedback] = useState<Feedback[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const notice = useMessage();

  useEffect(() => {
    let active = true;
    async function refresh() {
      try {
        const [values, reports] = await Promise.all([api<VideoJob[]>("/api/videos"), api<Feedback[]>("/api/me/feedback")]);
        if (active) { setJobs(values); setFeedback(reports); }
      } catch (reason) {
        if (active) notice.fail(reason);
      }
    }
    void refresh();
    const timer = window.setInterval(refresh, 3000);
    return () => { active = false; window.clearInterval(timer); };
  }, []);

  async function upload(event: React.FormEvent) {
    event.preventDefault();
    if (!file) return;
    setBusy(true);
    notice.clear();
    const form = new FormData();
    form.append("file", file);
    try {
      const job = await api<VideoJob>("/api/videos", { method: "POST", body: form });
      setJobs((previous) => [job, ...previous]);
      setSelected(job.id);
      setFile(null);
    } catch (reason) {
      notice.fail(reason);
    } finally {
      setBusy(false);
    }
  }

  async function deleteJob(job: VideoJob) {
    if (!window.confirm(`Xóa video ${job.original_name} cùng sự kiện và phản hồi liên quan?`)) return;
    try {
      await api(`/api/videos/${job.id}`, { method: "DELETE" });
      setJobs((items) => items.filter((item) => item.id !== job.id));
      setFeedback((items) => items.filter((item) => item.job_id !== job.id));
      notice.success("Đã xóa video và dữ liệu liên quan.");
    } catch (reason) { notice.fail(reason); }
  }

  async function deleteReport(item: Feedback) {
    if (!window.confirm("Xóa phản hồi này?")) return;
    try {
      await api(`/api/feedback/${item.id}`, { method: "DELETE" });
      setFeedback((items) => items.filter((report) => report.id !== item.id));
      notice.success("Đã xóa phản hồi.");
    } catch (reason) { notice.fail(reason); }
  }

  if (selected) return <VideoDetail jobId={selected} onBack={() => setSelected(null)} />;
  return <div className="page">
    <div className="page-header"><div><p className="eyebrow">Không gian người dùng</p><h1>Phân tích video</h1><p className="muted">Mỗi lần tải một video. Bạn có thể tạo nhiều tác vụ và xem lại lịch sử.</p></div></div>
    <Notice message={notice.message} error={notice.error} />
    <section className="upload-zone">
      <div className="upload-zone-info">
        <p className="eyebrow">Tác vụ mới</p>
        <h2>Chọn video cần phân tích</h2>
        <p>Hệ thống xử lý video trên máy, sau đó xuất bản có box để bạn xem lại.</p>
        <div className="spec-badges">
          <span className="spec-badge">MP4</span>
          <span className="spec-badge">Tối đa 30s</span>
          <span className="spec-badge">Tối đa 50 MB</span>
          <span className="spec-badge">Đầu vào tới 4K, 60 FPS</span>
          <span className="spec-badge">Xuất 720p</span>
        </div>
      </div>
      <form onSubmit={upload}>
        <label className="file-input">
          <svg className="file-input-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242" />
            <path d="M12 12v9" /><path d="m16 16-4-4-4 4" />
          </svg>
          <span className="file-input-label">{file ? file.name : "Chọn tệp MP4 từ máy tính"}</span>
          <input type="file" accept=".mp4,video/mp4" onChange={(event) => setFile(event.target.files?.[0] || null)} required />
        </label>
        {file && <span className="file-name">{file.name}</span>}
        <button className="button primary" disabled={!file || busy}>{busy ? "Đang xử lý..." : "Bắt đầu phân tích"}</button>
      </form>
    </section>
    <section className="section-block">
      <div className="section-heading"><div><p className="eyebrow">Lịch sử của bạn</p><h2>Các video đã tải</h2></div><span className="count-label">{jobs.length} tác vụ</span></div>
      {jobs.length ? <div className="table-scroll"><table><thead><tr><th>Video</th><th>Thời gian</th><th>Trạng thái</th><th>Sự kiện</th><th></th></tr></thead><tbody>
        {jobs.map((job) => <tr key={job.id}><td><strong>{job.original_name}</strong><small>{formatVideoTime(job.duration_ms)}</small></td><td>{formatDate(job.created_at)}</td><td><span className={`status ${job.status}`}>{jobLabels[job.status]}</span></td><td>{job.events_count}</td><td><button className="text-button" onClick={() => setSelected(job.id)}>Xem chi tiết</button>{!(["queued", "processing", "finalizing"] as string[]).includes(job.status) && <button className="text-button danger-text" onClick={() => void deleteJob(job)}>Xóa</button>}</td></tr>)}
      </tbody></table></div> : <p className="empty-state">Chưa có video nào. Tải một video để bắt đầu.</p>}
    </section>
    <section className="section-block">
      <div className="section-heading"><div><p className="eyebrow">Phản hồi của bạn</p><h2>Khung hình đã báo sai</h2></div><span className="count-label">{feedback.length} phản hồi</span></div>
      {feedback.length ? <div className="evidence-grid">{feedback.map((item) => <div key={item.id}><button className="evidence-item" onClick={() => setSelected(item.job_id)}>
        <img src={item.image_url} alt={`Ảnh phản hồi ${feedbackLabels[item.type] || item.type}`} loading="lazy" />
        <span><strong>{feedbackLabels[item.type] || item.type}</strong><small>{formatVideoTime(item.video_time_ms)} · {item.status === "pending" ? "Chờ duyệt" : item.status === "accepted" ? "Đã chấp nhận" : "Đã từ chối"}</small></span>
      </button><button className="text-button danger-text" onClick={() => void deleteReport(item)}>Xóa phản hồi</button></div>)}</div> : <p className="empty-state">Bạn chưa gửi phản hồi dự đoán nào.</p>}
    </section>
  </div>;
}

type RealtimeDetection = { class_id: number; confidence: number; xyxy: [number, number, number, number] };

function RealtimeView() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const resultCanvasRef = useRef<HTMLCanvasElement>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const detectionsRef = useRef<RealtimeDetection[]>([]);
  const busyUntilRef = useRef(0);
  const pendingRef = useRef(false);
  const inferenceCountRef = useRef(0);
  const fpsWindowStartedRef = useRef(0);
  const [starting, setStarting] = useState(false);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState("Webcam chưa được bật.");
  const [latency, setLatency] = useState<number | null>(null);
  const [fps, setFps] = useState<number | null>(null);
  const [modelName, setModelName] = useState("");
  const [zaloNotice, setZaloNotice] = useState("Zalo chưa được kiểm tra.");
  const [error, setError] = useState("");
  const [sessionId, setSessionId] = useState("");
  const [sessionStartedAt, setSessionStartedAt] = useState("");
  const [sessions, setSessions] = useState<RealtimeSession[]>([]);
  const [feedbackOpen, setFeedbackOpen] = useState(false);
  const [feedbackType, setFeedbackType] = useState("false_positive");
  const [expectedClass, setExpectedClass] = useState<0 | 1>(1);
  const [feedbackComment, setFeedbackComment] = useState("");
  const [feedbackBusy, setFeedbackBusy] = useState(false);
  const feedbackNotice = useMessage();

  async function refreshSessions() {
    try { setSessions(await api<RealtimeSession[]>("/api/realtime-sessions")); }
    catch (reason) { feedbackNotice.fail(reason); }
  }

  async function deleteRealtimeSession(id: string) {
    if (!window.confirm("Xóa phiên webcam và các ảnh minh chứng/phản hồi liên quan?")) return;
    try {
      await api(`/api/realtime-sessions/${id}`, { method: "DELETE" });
      await refreshSessions();
    } catch (reason) { feedbackNotice.fail(reason); }
  }

  async function reportRealtimeEvent(eventId: string) {
    if (!window.confirm("Gửi cảnh báo này cho admin mô hình kiểm tra là phát hiện nhầm?")) return;
    try {
      await api(`/api/realtime-events/${eventId}/feedback`, json("POST", {
        type: "false_positive",
        expected_class: null,
        comment: "User báo cảnh báo webcam là phát hiện nhầm.",
      }));
      feedbackNotice.success("Đã gửi cảnh báo webcam cùng ảnh minh chứng cho admin mô hình.");
    } catch (reason) { feedbackNotice.fail(reason); }
  }

  useEffect(() => { void refreshSessions(); }, []);

  function stop() {
    socketRef.current?.close();
    socketRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    detectionsRef.current = [];
    pendingRef.current = false;
    setStarting(false);
    setRunning(false);
    setStatus("Webcam đã dừng.");
    window.setTimeout(() => { void refreshSessions(); }, 250);
  }

  async function submitRealtimeFeedback(event: React.FormEvent) {
    event.preventDefault();
    const video = videoRef.current;
    const canvas = captureCanvasRef.current;
    if (!sessionId || !video || !canvas || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      feedbackNotice.fail(new Error("Chưa có khung hình webcam để gửi phản hồi."));
      return;
    }
    const context = canvas.getContext("2d");
    if (!context) return;
    const scale = Math.min(1, 960 / video.videoWidth, 960 / video.videoHeight);
    canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
    canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    setFeedbackBusy(true);
    try {
      const blob = await new Promise<Blob>((resolve, reject) => canvas.toBlob((value) => value ? resolve(value) : reject(new Error("Không chụp được khung hình.")), "image/jpeg", 0.88));
      const form = new FormData();
      form.append("elapsed_ms", String(Math.max(0, Date.now() - Date.parse(sessionStartedAt))));
      form.append("type", feedbackType);
      if (feedbackType === "false_negative" || feedbackType === "wrong_class") form.append("expected_class", String(expectedClass));
      if (feedbackComment.trim()) form.append("comment", feedbackComment.trim());
      form.append("frame", blob, "webcam-feedback.jpg");
      await api(`/api/realtime-sessions/${sessionId}/feedback`, { method: "POST", body: form });
      feedbackNotice.success("Đã gửi ảnh và phản hồi webcam cho admin mô hình xem xét.");
      setFeedbackOpen(false);
      setFeedbackComment("");
      await refreshSessions();
    } catch (reason) { feedbackNotice.fail(reason); }
    finally { setFeedbackBusy(false); }
  }

  useEffect(() => {
    let frameRequest = 0;
    let lastSentAt = 0;
    fpsWindowStartedRef.current = performance.now();
    const video = videoRef.current;
    const canvas = resultCanvasRef.current;
    const captureCanvas = document.createElement("canvas");
    captureCanvasRef.current = captureCanvas;
    const context = canvas?.getContext("2d");
    const captureContext = captureCanvas.getContext("2d");

    const render = (now: number) => {
      if (video && canvas && context && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
        const width = video.videoWidth;
        const height = video.videoHeight;
        if (width && height) {
          if (canvas.width !== width || canvas.height !== height) {
            canvas.width = width;
            canvas.height = height;
          }
          context.drawImage(video, 0, 0, width, height);
          context.lineWidth = Math.max(2, width / 480);
          context.font = `700 ${Math.max(14, width / 38)}px Inter, Arial, sans-serif`;
          for (const detection of detectionsRef.current) {
            const [x1, y1, x2, y2] = detection.xyxy;
            const color = detection.class_id === 0 ? "#3f8de7" : "#d95143";
            const label = `${detection.class_id === 0 ? "Khói" : "Lửa"} ${Math.round(detection.confidence * 100)}%`;
            context.strokeStyle = color;
            context.fillStyle = color;
            context.strokeRect(x1, y1, x2 - x1, y2 - y1);
            const textWidth = context.measureText(label).width;
            const textTop = Math.max(0, y1 - 28);
            context.fillRect(x1, textTop, textWidth + 12, 27);
            context.fillStyle = "#fff";
            context.fillText(label, x1 + 6, textTop + 19);
          }

          if (socketRef.current?.readyState === WebSocket.OPEN && !pendingRef.current && now >= busyUntilRef.current && now - lastSentAt >= 500 && captureContext) {
            const scale = Math.min(1, 960 / width, 960 / height);
            const captureWidth = Math.max(1, Math.round(width * scale));
            const captureHeight = Math.max(1, Math.round(height * scale));
            captureCanvas.width = captureWidth;
            captureCanvas.height = captureHeight;
            captureContext.drawImage(video, 0, 0, captureWidth, captureHeight);
            pendingRef.current = true;
            lastSentAt = now;
            captureCanvas.toBlob((blob) => {
              if (!blob || socketRef.current?.readyState !== WebSocket.OPEN) {
                pendingRef.current = false;
                return;
              }
              void blob.arrayBuffer().then((buffer) => {
                if (socketRef.current?.readyState === WebSocket.OPEN) socketRef.current.send(buffer);
                else pendingRef.current = false;
              }).catch(() => { pendingRef.current = false; });
            }, "image/jpeg", 0.72);
          }
        }
      }
      frameRequest = window.requestAnimationFrame(render);
    };
    frameRequest = window.requestAnimationFrame(render);
    return () => {
      window.cancelAnimationFrame(frameRequest);
      socketRef.current?.close();
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  async function start() {
    if (starting || running) return;
    setStarting(true);
    setError("");
    setLatency(null);
    setFps(null);
    setModelName("");
    setZaloNotice("Đang kiểm tra kết nối Zalo...");
    busyUntilRef.current = 0;
    pendingRef.current = false;
    detectionsRef.current = [];
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error("Trình duyệt không hỗ trợ webcam hoặc trang chưa chạy trên localhost/HTTPS.");
      const stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 15, max: 30 } } });
      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) throw new Error("Không tìm thấy vùng xem camera.");
      video.srcObject = stream;
      await video.play();
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const socket = new WebSocket(`${protocol}//${window.location.host}/api/realtime`);
      socket.binaryType = "arraybuffer";
      socketRef.current = socket;
      socket.onopen = () => { setStarting(false); setRunning(true); setStatus("Đang kết nối bộ dự đoán..."); };
      socket.onmessage = (event) => {
        if (typeof event.data !== "string") return;
        let message: { type: string; model?: string; zalo_status?: string; status?: string; latency_ms?: number; width?: number; height?: number; detections?: RealtimeDetection[]; message?: string; session_id?: string; session_started_at?: string };
        try { message = JSON.parse(event.data); } catch { return; }
        if (message.type === "ready") {
          setStatus("Đang dự đoán trực tiếp từ webcam.");
          setModelName(message.model || "");
          setSessionId(message.session_id || "");
          setSessionStartedAt(message.session_started_at || new Date().toISOString());
          setZaloNotice(message.zalo_status === "verified"
            ? "Bot Zalo đã xác minh; cảnh báo và tin nhắc sẽ gửi khi phát hiện còn duy trì."
            : message.zalo_status === "unverified"
              ? "Bot Zalo đã lưu nhưng chưa xác minh nên chưa thể nhận cảnh báo."
              : message.zalo_status === "error"
                ? "Không giải mã được cấu hình Zalo; hãy kết nối lại bot trong hồ sơ."
                : "Chưa kết nối bot Zalo; dự đoán webcam vẫn hoạt động nhưng không gửi thông báo.");
        } else if (message.type === "prediction") {
          pendingRef.current = false;
          const scaleX = videoRef.current && message.width ? videoRef.current.videoWidth / message.width : 1;
          const scaleY = videoRef.current && message.height ? videoRef.current.videoHeight / message.height : 1;
          detectionsRef.current = (message.detections || []).map((detection) => ({
            ...detection,
            xyxy: [detection.xyxy[0] * scaleX, detection.xyxy[1] * scaleY, detection.xyxy[2] * scaleX, detection.xyxy[3] * scaleY] as [number, number, number, number],
          }));
          setLatency(message.latency_ms ?? null);
          inferenceCountRef.current += 1;
          const now = performance.now();
          if (now - fpsWindowStartedRef.current >= 2000) {
            setFps(Math.round((inferenceCountRef.current * 1000 / (now - fpsWindowStartedRef.current)) * 10) / 10);
            inferenceCountRef.current = 0;
            fpsWindowStartedRef.current = now;
          }
          setStatus(detectionsRef.current.length ? `Có ${detectionsRef.current.length} box trên khung hình hiện tại.` : "Không có box trên khung hình hiện tại.");
        } else if (message.type === "busy") {
          pendingRef.current = false;
          busyUntilRef.current = performance.now() + 3000;
          detectionsRef.current = [];
          setStatus("Worker đang xử lý video tải lên; webcam sẽ thử lại sau.");
        } else if (message.type === "frame_error") {
          pendingRef.current = false;
          setStatus(message.message || "Không thể xử lý khung hình.");
        } else if (message.type === "zalo_status") {
          setZaloNotice(message.message || (message.status === "sent" ? "Đã gửi thông báo webcam qua Zalo." : "Không gửi được thông báo webcam qua Zalo."));
        }
      };
      socket.onerror = () => setError("Không kết nối được máy chủ dự đoán. Hãy kiểm tra backend đang chạy.");
      socket.onclose = (event) => {
        pendingRef.current = false;
        if (socketRef.current === socket) socketRef.current = null;
        streamRef.current?.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
        if (videoRef.current) videoRef.current.srcObject = null;
        setStarting(false);
        setRunning(false);
        if (event.code === 4401) setError("Phiên đăng nhập hết hạn. Hãy đăng nhập lại.");
        else if (event.code === 4403) setError("Nguồn kết nối webcam không được chấp nhận.");
        else if (event.code === 4409) setError("Tài khoản này đang có một phiên webcam khác. Hãy dừng phiên đó rồi thử lại.");
        else if (event.code !== 1000 && event.code !== 1001) setError(event.reason || "Kết nối webcam đã đóng.");
        setStatus("Webcam đã dừng.");
        void refreshSessions();
      };
    } catch (reason) {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      setStarting(false);
      setRunning(false);
      setError(reason instanceof Error ? reason.message : "Không thể khởi động webcam.");
    }
  }

  return <div className="page realtime-page">
    <div className="page-header"><div><p className="eyebrow">Dự đoán trực tiếp</p><h1>Webcam laptop</h1><p className="muted">Hai khung hình chạy liên tục: camera gốc và cùng luồng hình có box dự đoán. Máy chủ xử lý tối đa khoảng 2 khung hình mỗi giây; video xem trước vẫn chạy theo camera.</p></div></div>
    <div className="realtime-toolbar">
      <div className="realtime-status-pill">
        <span className={`realtime-indicator ${running ? "on" : ""}`} />
        <span>{status}</span>
      </div>
      <div className="realtime-metrics">
        {modelName && <span className="metric-pill">Mô hình: {modelName}</span>}
        {latency !== null && <span className="metric-pill">Độ trễ: {latency} ms</span>}
        {fps !== null && <span className="metric-pill">Tốc độ: {fps} khung/giây</span>}
      </div>
    </div>
    {error && <div className="notice error" role="alert">{error}</div>}
    <Notice message={feedbackNotice.message} error={feedbackNotice.error} />
    <div className="realtime-grid">
      <section className="realtime-panel">
        <div className="realtime-panel-title"><h2>Camera gốc</h2><span>Trực tiếp</span></div>
        <div className="realtime-screen"><video ref={videoRef} muted playsInline aria-label="Hình ảnh webcam gốc" /></div>
      </section>
      <section className="realtime-panel">
        <div className="realtime-panel-title"><h2>Kết quả dự đoán</h2><span>YOLOv11 Detection Stream</span></div>
        <div className="realtime-screen"><canvas ref={resultCanvasRef} aria-label="Luồng webcam có box dự đoán" /></div>
      </section>
    </div>
    <div className="realtime-actions">
      <button className="button primary" onClick={() => void start()} disabled={running || starting}>
        <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>
        {starting ? "Đang khởi động..." : running ? "Đang chạy webcam" : "Bật webcam và dự đoán"}
      </button>
      <button className="button secondary" onClick={stop} disabled={!running}>
        <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>
        Dừng
      </button>
    </div>
    <div className="inline-actions">
      <button className="button secondary" disabled={!running || !sessionId} onClick={() => { feedbackNotice.clear(); setFeedbackOpen(true); }}>
        <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"/><line x1="4" y1="22" x2="4" y2="15"/></svg>
        Báo dự đoán sai hoặc bị bỏ sót
      </button>
    </div>
    <p className="small realtime-notification-status">{zaloNotice}</p>
    <p className="small muted realtime-privacy">Trình duyệt cần bạn cấp quyền camera. Khung hình chỉ truyền đến máy chủ để suy luận; hệ thống không lưu video webcam. Ảnh tại thời điểm cảnh báo được lưu làm minh chứng; ảnh gửi kèm phản hồi được lưu để admin rà soát.</p>
    {feedbackOpen && <div className="modal-backdrop" onClick={() => setFeedbackOpen(false)}><section className="modal" onClick={(event) => event.stopPropagation()}><div className="modal-head"><div><p className="eyebrow">Phản hồi webcam</p><h2>Gửi khung hình đang xem</h2></div><button className="text-button" onClick={() => setFeedbackOpen(false)}>Đóng</button></div><Notice message={feedbackNotice.message} error={feedbackNotice.error} /><form className="stack-form" onSubmit={submitRealtimeFeedback}>
      <label>Loại phản hồi<select value={feedbackType} onChange={(event) => setFeedbackType(event.target.value)}><option value="false_positive">Đây không phải lửa/khói</option><option value="false_negative">Mô hình bỏ sót lửa/khói</option><option value="wrong_class">Nhận sai loại</option><option value="wrong_box">Box sai vị trí</option></select></label>
      {(feedbackType === "false_negative" || feedbackType === "wrong_class") && <label>Lớp đúng<select value={expectedClass} onChange={(event) => setExpectedClass(Number(event.target.value) as 0 | 1)}><option value={1}>Lửa</option><option value={0}>Khói</option></select></label>}
      <label>Ghi chú<textarea rows={3} value={feedbackComment} onChange={(event) => setFeedbackComment(event.target.value)} maxLength={1000} /></label>
      <p className="small muted">Ảnh webcam hiện tại sẽ được gửi cùng phản hồi. Admin mô hình sẽ kiểm tra ảnh trước khi dùng làm dữ liệu.</p><button className="button primary" disabled={feedbackBusy}>{feedbackBusy ? "Đang gửi..." : "Gửi phản hồi"}</button>
    </form></section></div>}
    <section className="page-section"><div className="page-header"><div><p className="eyebrow">Lịch sử realtime</p><h2>Các phiên webcam gần đây</h2></div><button className="text-button" onClick={() => void refreshSessions()}>Làm mới</button></div><div className="table-scroll"><table><thead><tr><th>Bắt đầu</th><th>Trạng thái</th><th>Thời lượng</th><th>Khung đã xử lý</th><th>Sự kiện</th><th></th></tr></thead><tbody>{sessions.map((item) => <tr key={item.id}><td>{formatDate(item.started_at)}</td><td>{item.status === "running" ? "Đang chạy" : item.status === "failed" ? "Lỗi" : item.status === "interrupted" ? "Bị gián đoạn" : "Đã dừng"}</td><td>{formatVideoTime(item.duration_ms)}</td><td>{item.frame_count}</td><td>{item.events_count}{item.events?.length > 0 && <details><summary>Xem cảnh báo</summary>{item.events.map((event) => <div key={event.id} className="history-event">{event.evidence_url && <img className="history-evidence" src={event.evidence_url} alt="Ảnh minh chứng webcam" />}<p>{className(event.class_id)} · {formatVideoTime(event.start_ms)}–{formatVideoTime(event.end_ms)} · {Math.round(event.peak_confidence * 100)}%</p><button className="text-button" onClick={() => void reportRealtimeEvent(event.id)}>Báo cảnh báo nhầm</button></div>)}</details>}</td><td>{item.status !== "running" && <button className="text-button danger-text" onClick={() => void deleteRealtimeSession(item.id)}>Xóa</button>}</td></tr>)}{!sessions.length && <tr><td colSpan={6}>Chưa có phiên webcam được lưu.</td></tr>}</tbody></table></div></section>
  </div>;
}

function ProfileView() {
  const [profile, setProfile] = useState<Profile | null>(null);
  const [fullName, setFullName] = useState("");
  const [phone, setPhone] = useState("");
  const [address, setAddress] = useState("");
  const [zaloToken, setZaloToken] = useState("");
  const [guideOpen, setGuideOpen] = useState(false);
  const notice = useMessage();

  async function refresh() {
    try {
      const value = await api<Profile>("/api/me/profile");
      setProfile(value);
      setFullName(value.full_name || "");
      setPhone(value.phone || "");
      setAddress(value.address || "");
    } catch (reason) {
      notice.fail(reason);
    }
  }
  useEffect(() => { void refresh(); }, []);

  async function saveProfile(event: React.FormEvent) {
    event.preventDefault();
    try {
      await api("/api/me/profile", json("PATCH", { full_name: fullName, phone, address }));
      notice.success("Đã cập nhật thông tin cá nhân.");
      await refresh();
    } catch (reason) { notice.fail(reason); }
  }

  async function connectZalo(event: React.FormEvent) {
    event.preventDefault();
    try {
      await api("/api/me/zalo/connect", json("POST", { bot_token: zaloToken }));
      setZaloToken("");
      notice.success("Đã kết nối bot Zalo và gửi tin thử thành công.");
      await refresh();
    } catch (reason) { notice.fail(reason); }
  }

  async function testZalo() {
    try {
      await api("/api/me/zalo/test", { method: "POST" });
      notice.success("Đã gửi tin thử qua Zalo.");
      await refresh();
    } catch (reason) { notice.fail(reason); }
  }

  async function deleteZalo() {
    try {
      await api("/api/me/zalo", { method: "DELETE" });
      notice.success("Đã ngắt kết nối bot Zalo.");
      await refresh();
    } catch (reason) { notice.fail(reason); }
  }

  return <div className="page">
    <div className="page-header"><div><p className="eyebrow">Tài khoản</p><h1>Hồ sơ và thông báo</h1><p className="muted">Thông tin liên hệ giúp quản trị viên nhận biết người gửi video khi có sự kiện.</p></div></div>
    <Notice message={notice.message} error={notice.error} />
    <div className="profile-layout">
      <section><h2>Thông tin cá nhân</h2><form className="stack-form" onSubmit={saveProfile}>
        <label>Họ tên<input value={fullName} onChange={(event) => setFullName(event.target.value)} maxLength={120} /></label>
        <label>Số điện thoại<input value={phone} onChange={(event) => setPhone(event.target.value)} maxLength={30} /></label>
        <label>Địa chỉ<textarea value={address} onChange={(event) => setAddress(event.target.value)} maxLength={300} rows={3} /></label>
        <button className="button primary">Lưu thông tin</button>
      </form></section>
      <section><div className="split-title"><h2>Thông báo Zalo</h2><span className={`status ${profile?.zalo_status || "unconfigured"}`}>{profile?.zalo_status === "verified" ? "Đã xác minh" : profile?.zalo_status === "saved" ? "Chưa xác minh" : "Chưa cấu hình"}</span></div>
        <p className="muted">Kết nối bot để nhận cảnh báo từ video tải lên và webcam trực tiếp.</p>
        <form className="stack-form" onSubmit={connectZalo}>
          <label>Bot Token<input type="password" value={zaloToken} onChange={(event) => setZaloToken(event.target.value)} autoComplete="off" required placeholder="Dán Bot Token tại đây" /></label>
          <p className="small muted">Bạn đã nhắn cho bot rồi. Backend sẽ tự lấy mã cuộc trò chuyện, gửi tin thử và lưu token đã mã hóa.</p>
          <button className="button secondary">Kết nối bot Zalo</button>
        </form>
        {profile?.zalo_status !== "unconfigured" && <div className="inline-actions"><button className="text-button" onClick={testZalo}>Gửi tin thử</button><button className="text-button danger-text" onClick={deleteZalo}>Ngắt kết nối</button></div>}
        <button className="text-button guide-toggle" onClick={() => setGuideOpen(!guideOpen)}>{guideOpen ? "Ẩn hướng dẫn Zalo" : "Xem hướng dẫn kết nối bot"}</button>
        {guideOpen && <div className="guide"><h3>Kết nối bot Zalo</h3><ol>
          <li>Mở cuộc trò chuyện với bot trong Zalo và gửi <code>/start</code> hoặc một tin nhắn.</li>
          <li>Dán Bot Token vào ô trên. Không gửi token qua chat hoặc chia sẻ cho người khác.</li>
          <li>Nhấn “Kết nối bot Zalo”. Hệ thống tự lấy mã cuộc trò chuyện và gửi tin thử.</li>
          <li>Khi trạng thái là “Đã xác minh”, cảnh báo video và webcam sẽ gửi đến cuộc trò chuyện này.</li>
        </ol></div>}
      </section>
    </div>
  </div>;
}

function EventsView() {
  const [events, setEvents] = useState<AdminEvent[]>([]);
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [selectedJob, setSelectedJob] = useState<AdminJob | null>(null);
  const notice = useMessage();
  useEffect(() => {
    let active = true;
    async function refresh() {
      const params = new URLSearchParams();
      if (dateFrom) params.set("date_from", dateFrom);
      if (dateTo) params.set("date_to", dateTo);
      try { const value = await api<AdminEvent[]>(`/api/admin/events?${params}`); if (active) setEvents(value); }
      catch (reason) { if (active) notice.fail(reason); }
    }
    void refresh();
    const timer = window.setInterval(refresh, 3000);
    return () => { active = false; window.clearInterval(timer); };
  }, [dateFrom, dateTo]);
  useEffect(() => {
    if (!selectedJobId) { setSelectedJob(null); return; }
    api<AdminJob>(`/api/admin/jobs/${selectedJobId}`).then(setSelectedJob).catch(notice.fail);
  }, [selectedJobId]);
  return <div className="page">
    <div className="page-header"><div><p className="eyebrow">Quản trị vận hành</p><h1>Luồng cảnh báo</h1><p className="muted">Sự kiện từ webcam realtime và video tải lên, mới nhất ở trên.</p></div><span className="count-label">{events.length} sự kiện gần đây</span></div>
    <Notice message={notice.message} error={notice.error} />
    <div className="date-filter"><label>Từ ngày<input type="date" value={dateFrom} onChange={(event) => setDateFrom(event.target.value)} /></label><label>Đến ngày<input type="date" value={dateTo} onChange={(event) => setDateTo(event.target.value)} /></label><button className="text-button" onClick={() => { setDateFrom(""); setDateTo(""); }}>Xóa lọc</button></div>
    <div className="event-feed">{events.length ? events.map((item) => <article className={`event-row ${item.class_id === 1 ? "fire-event" : "smoke-event"}`} key={item.id}>
      <div className={`event-accent ${item.class_id === 1 ? "fire" : "smoke"}`} />
      {item.evidence_url && <img src={item.evidence_url} alt={`Khung hình phát hiện ${className(item.class_id)}`} />}
      <div className="event-body"><div className="event-topline"><span className="eyebrow">{className(item.class_id)} · {item.source_type === "webcam" ? "Webcam realtime" : "Video tải lên"}</span><time>{formatDate(item.created_at)}</time></div>
        <h2>{item.contact.full_name || item.username}</h2>
        <p><strong>{item.video_name}</strong> · {formatVideoTime(item.start_ms)} – {formatVideoTime(item.end_ms)} · độ tin cậy cao nhất {Math.round(item.peak_confidence * 100)}%</p>
        <p className="muted">{item.contact.phone || "Chưa có số điện thoại"} · {item.contact.address || "Chưa có địa chỉ"}</p>
        <small>Zalo: {item.notification_status === "sent" ? "Đã gửi" : item.notification_status === "pending" ? "Đang chờ gửi" : item.notification_status === "failed" ? "Gửi thất bại" : item.notification_status === "skipped" ? "Chưa cấu hình bot" : "Chưa cấu hình"}</small>
        {item.job_id && <div><button className="text-button" onClick={() => setSelectedJobId(item.job_id)}>Xem thông tin video</button></div>}
      </div>
    </article>) : <p className="empty-state">Chưa có sự kiện nào từ các video đã xử lý.</p>}</div>
    {selectedJobId && <div className="modal-backdrop" onClick={() => setSelectedJobId(null)}><section className="modal admin-video-modal" onClick={(event) => event.stopPropagation()}>
      <div className="modal-head"><div><p className="eyebrow">Video tải lên</p><h2>{selectedJob?.original_name || "Đang tải..."}</h2></div><button className="text-button" onClick={() => setSelectedJobId(null)}>Đóng</button></div>
      {selectedJob && <><p>{selectedJob.contact.full_name || selectedJob.username} · {formatDate(selectedJob.created_at)} · {jobLabels[selectedJob.status]}</p>
        {selectedJob.video_url ? <video controls preload="metadata" src={selectedJob.video_url} /> : <p>Video kết quả chưa sẵn sàng.</p>}
        <p>Thời lượng {formatVideoTime(selectedJob.duration_ms)} · {selectedJob.processed_frames} khung đã xử lý · {selectedJob.events_count} sự kiện</p>
        {selectedJob.events?.map((event) => <p key={event.id}>{className(event.class_id)} · {formatVideoTime(event.start_ms)} – {formatVideoTime(event.end_ms)} · {Math.round(event.peak_confidence * 100)}%</p>)}
      </>}
    </section></div>}
  </div>;
}

function AdminJobsView() {
  const [jobs, setJobs] = useState<AdminJob[]>([]);
  const notice = useMessage();
  useEffect(() => {
    let active = true;
    async function refresh() {
      try { const values = await api<AdminJob[]>("/api/admin/jobs"); if (active) setJobs(values); }
      catch (reason) { if (active) notice.fail(reason); }
    }
    void refresh();
    const timer = window.setInterval(refresh, 5000);
    return () => { active = false; window.clearInterval(timer); };
  }, []);
  return <div className="page">
    <div className="page-header"><div><p className="eyebrow">Quản trị vận hành</p><h1>Lịch sử video</h1><p className="muted">Theo dõi các tác vụ của mọi người dùng, kể cả video không tạo cảnh báo.</p></div><span className="count-label">{jobs.length} tác vụ gần đây</span></div>
    <Notice message={notice.message} error={notice.error} />
    {jobs.length ? <div className="table-scroll"><table><thead><tr><th>Người dùng</th><th>Video</th><th>Tải lên</th><th>Trạng thái</th><th>Sự kiện</th></tr></thead><tbody>{jobs.map((job) => <tr key={job.id}>
      <td><strong>{job.contact.full_name || job.username}</strong><small>{job.username}</small></td><td>{job.original_name}</td><td>{formatDate(job.created_at)}</td><td><span className={`status ${job.status}`}>{jobLabels[job.status]}</span></td><td>{job.events_count}</td>
    </tr>)}</tbody></table></div> : <p className="empty-state">Chưa có video nào được tải lên.</p>}
  </div>;
}

function AccountsView() {
  const [users, setUsers] = useState<User[]>([]);
  const [selected, setSelected] = useState<User | null>(null);
  const [editActive, setEditActive] = useState(true);
  const [editName, setEditName] = useState("");
  const [editPhone, setEditPhone] = useState("");
  const [editAddress, setEditAddress] = useState("");
  const [editPassword, setEditPassword] = useState("");
  const notice = useMessage();

  async function refresh() {
    try { setUsers(await api<User[]>("/api/admin/users")); }
    catch (reason) { notice.fail(reason); }
  }
  useEffect(() => { void refresh(); }, []);
  function choose(user: User) {
    setSelected(user); setEditActive(user.active);
    setEditName(user.profile.full_name || ""); setEditPhone(user.profile.phone || "");
    setEditAddress(user.profile.address || ""); setEditPassword("");
  }
  async function update() {
    if (!selected) return;
    try {
      await api(`/api/admin/users/${selected.id}`, json("PATCH", {
        active: editActive, full_name: editName, phone: editPhone,
        address: editAddress, ...(editPassword ? { new_password: editPassword } : {}),
      }));
      notice.success("Đã cập nhật tài khoản.");
      setSelected(null);
      await refresh();
    } catch (reason) { notice.fail(reason); }
  }
  async function disable() {
    if (!selected) return;
    try {
      await api(`/api/admin/users/${selected.id}`, { method: "DELETE" });
      notice.success("Đã khóa tài khoản. Lịch sử video và phản hồi được giữ để đối chiếu.");
      setSelected(null);
      await refresh();
    } catch (reason) { notice.fail(reason); }
  }
  return <div className="page">
    <div className="page-header"><div><p className="eyebrow">Quản trị vận hành</p><h1>Tài khoản và quyền</h1><p className="muted">Tài khoản tự đăng ký luôn bắt đầu với quyền người dùng.</p></div></div>
    <Notice message={notice.message} error={notice.error} />
    <div className="admin-split"><section><h2>Danh sách tài khoản</h2><div className="table-scroll"><table><thead><tr><th>Tài khoản</th><th>Vai trò</th><th>Trạng thái</th><th></th></tr></thead><tbody>
      {users.map((user) => <tr key={user.id}><td><strong>{user.profile.full_name || user.username}</strong><small>{user.username}</small></td><td>{user.roles.join(", ")}</td><td>{user.active ? "Hoạt động" : "Đã khóa"}</td><td>{user.roles.length === 1 && user.roles[0] === "user" ? <button className="text-button" onClick={() => choose(user)}>Chỉnh sửa</button> : <span className="muted">Quản lý bằng lệnh</span>}</td></tr>)}
    </tbody></table></div></section>
      <aside>{selected ? <><h2>Chỉnh sửa {selected.username}</h2><div className="stack-form">
          <label>Họ tên<input value={editName} onChange={(event) => setEditName(event.target.value)} maxLength={120} /></label>
          <label>Số điện thoại<input value={editPhone} onChange={(event) => setEditPhone(event.target.value)} maxLength={30} /></label>
          <label>Địa chỉ<textarea value={editAddress} onChange={(event) => setEditAddress(event.target.value)} maxLength={300} rows={2} /></label>
          <label>Mật khẩu mới (để trống nếu giữ nguyên)<input type="password" value={editPassword} onChange={(event) => setEditPassword(event.target.value)} minLength={10} /></label>
          <label className="check-row"><input type="checkbox" checked={editActive} onChange={(event) => setEditActive(event.target.checked)} />Tài khoản hoạt động</label>
          <div className="inline-actions"><button className="button primary" onClick={update}>Lưu thay đổi</button><button className="text-button" onClick={() => setSelected(null)}>Hủy</button>{editActive && <button className="text-button danger-text" onClick={disable}>Khóa tài khoản</button>}</div>
        </div></> : <p className="muted">Chọn một tài khoản người dùng để chỉnh sửa. Tài khoản admin được tạo riêng qua lệnh trên máy.</p>}
      </aside>
    </div>
  </div>;
}

function VersionsView() {
  const [models, setModels] = useState<ModelVersion[]>([]);
  const [reason, setReason] = useState("");
  const notice = useMessage();
  async function refresh() {
    try { setModels(await api<ModelVersion[]>("/api/admin/models")); }
    catch (error) { notice.fail(error); }
  }
  useEffect(() => { void refresh(); }, []);
  async function activate(model: ModelVersion) {
    try {
      await api(`/api/admin/models/${model.id}/activate`, json("POST", { reason }));
      notice.success("Đã kích hoạt phiên bản mô hình. Tác vụ mới sẽ dùng phiên bản này.");
      setReason("");
      await refresh();
    } catch (error) { notice.fail(error); }
  }
  return <div className="page"><div className="page-header"><div><p className="eyebrow">Quản trị vận hành</p><h1>Phiên bản mô hình</h1><p className="muted">Chỉ kích hoạt sau khi xem báo cáo đánh giá; có thể trở lại phiên bản trước.</p></div></div>
    <Notice message={notice.message} error={notice.error} />
    <label className="reason-field">Lý do kích hoạt hoặc rollback<input value={reason} onChange={(event) => setReason(event.target.value)} placeholder="Ví dụ: giảm báo động giả trên bộ video âm tính" minLength={5} /></label>
    <div className="version-list">{models.map((model) => <article key={model.id} className="version-row"><div><div className="split-title"><h2>{model.name}</h2><span className={`status ${model.status}`}>{modelLabels[model.status] || model.status}</span></div><p className="muted">Tạo {formatDate(model.created_at)} · SHA256 {model.sha256.slice(0, 14)}...</p><details><summary>Xem báo cáo đánh giá</summary><pre>{JSON.stringify(model.metrics, null, 2)}</pre></details></div>{model.status !== "active" && <button className="button secondary" disabled={reason.trim().length < 5} onClick={() => activate(model)}>{model.status === "retired" ? "Rollback về bản này" : "Kích hoạt"}</button>}</article>)}</div>
  </div>;
}

function ReviewsView() {
  const [items, setItems] = useState<Feedback[]>([]);
  const [state, setState] = useState("pending");
  const [selected, setSelected] = useState<Feedback | null>(null);
  const [decision, setDecision] = useState<"accepted" | "rejected">("accepted");
  const [annotationStatus, setAnnotationStatus] = useState<"objects" | "background" | null>(null);
  const [boxes, setBoxes] = useState<AnnotationBox[]>([]);
  const [suggestedBoxes, setSuggestedBoxes] = useState<AnnotationBox[]>([]);
  const [drawClass, setDrawClass] = useState<0 | 1>(0);
  const [reviewNote, setReviewNote] = useState("");
  const [saving, setSaving] = useState(false);
  const notice = useMessage();
  useEffect(() => {
    let active = true;
    api<Feedback[]>(`/api/model-admin/feedback?state=${state}`).then((value) => { if (active) setItems(value); }).catch((reason) => { if (active) notice.fail(reason); });
    return () => { active = false; };
  }, [state]);
  useEffect(() => {
    if (!selected) { setSuggestedBoxes([]); return; }
    let active = true;
    setSuggestedBoxes([]);
    api<AnnotationBox[]>(`/api/model-admin/feedback/${selected.id}/suggested-boxes`)
      .then((value) => { if (active) setSuggestedBoxes(value); })
      .catch((reason) => { if (active) notice.fail(reason); });
    return () => { active = false; };
  }, [selected?.id]);
  async function review(event: React.FormEvent) {
    event.preventDefault();
    if (!selected) return;
    setSaving(true);
    try {
      await api(`/api/model-admin/feedback/${selected.id}`, json("PATCH", {
        decision,
        annotation_status: decision === "accepted" ? annotationStatus : null,
        annotations: decision === "accepted" ? boxes : [],
        review_note: reviewNote,
      }));
      setItems(await api<Feedback[]>(`/api/model-admin/feedback?state=${state}`));
      setSelected(null);
      notice.success("Đã lưu nhãn và kết quả kiểm tra toàn ảnh.");
    } catch (reason) { notice.fail(reason); }
    finally { setSaving(false); }
  }
  async function downloadExport() {
    try {
      const response = await api<Response>("/api/model-admin/exports", { method: "POST" });
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url; link.download = "mau-da-gan-nhan.zip"; link.click();
      URL.revokeObjectURL(url);
    } catch (reason) { notice.fail(reason); }
  }
  function choose(item: Feedback) {
    setSelected(item);
    setDecision(item.status === "rejected" ? "rejected" : "accepted");
    setAnnotationStatus(item.annotation_status || null);
    setBoxes(item.annotations || []);
    setDrawClass(item.expected_class === 1 ? 1 : 0);
    setReviewNote(item.review_note || "");
    notice.clear();
  }
  function importSuggestions() {
    setBoxes((current) => [
      ...current,
      ...suggestedBoxes.filter((suggestion) => !current.some((box) =>
        box.class_id === suggestion.class_id &&
        Math.abs(box.x1 - suggestion.x1) < 0.01 &&
        Math.abs(box.y1 - suggestion.y1) < 0.01 &&
        Math.abs(box.x2 - suggestion.x2) < 0.01 &&
        Math.abs(box.y2 - suggestion.y2) < 0.01
      )),
    ]);
    setAnnotationStatus("objects");
  }
  const annotationValid = annotationStatus === "background" ? boxes.length === 0 : annotationStatus === "objects" && boxes.length > 0;
  return <div className="page"><div className="page-header"><div><p className="eyebrow">Quản trị mô hình</p><h1>Phản hồi dự đoán</h1><p className="muted">Gắn nhãn mọi vùng lửa/khói trên ảnh gốc rồi xác nhận toàn ảnh trước khi xuất.</p></div><button className="button secondary" onClick={downloadExport}>Xuất ảnh và nhãn YOLO đã duyệt</button></div>
    <Notice message={notice.message} error={notice.error} />
    <div className="tabs">{["pending", "accepted", "rejected", "all"].map((item) => <button key={item} className={state === item ? "active" : ""} onClick={() => setState(item)}>{item === "pending" ? "Chờ duyệt" : item === "accepted" ? "Đã chấp nhận" : item === "rejected" ? "Đã từ chối" : "Tất cả"}</button>)}</div>
    <div className="review-layout"><div className="review-list">{items.length ? items.map((item) => <button key={item.id} className={selected?.id === item.id ? "review-item selected" : "review-item"} onClick={() => choose(item)}><img src={item.image_url} alt="Khung hình phản hồi" /><span><strong>{item.video_name}</strong><small>{item.username} · {formatVideoTime(item.video_time_ms)}</small><small>{feedbackLabels[item.type] || item.type} · {formatDate(item.created_at)}</small><small>{item.annotation_status ? "Đã kiểm tra toàn ảnh" : item.status === "accepted" ? "Cần gắn nhãn lại" : "Chưa gắn nhãn"}</small></span></button>) : <p className="empty-state">Không có phản hồi trong nhóm này.</p>}</div>
      <aside className="review-panel">{selected ? <><h2>Kiểm tra mẫu</h2><p><strong>{feedbackLabels[selected.type] || selected.type}</strong> · {selected.comment || "Không có ghi chú"}</p><form className="stack-form" onSubmit={review}>
        <label>Quyết định<select value={decision} onChange={(event) => setDecision(event.target.value as typeof decision)}><option value="accepted">Duyệt ảnh đã gắn nhãn</option><option value="rejected">Từ chối phản hồi</option></select></label>
        {decision === "accepted" && <><div className="annotation-status"><label className="check-row"><input type="radio" name="annotation-status" checked={annotationStatus === "objects"} onChange={() => setAnnotationStatus("objects")} />Ảnh có lửa hoặc khói</label><label className="check-row"><input type="radio" name="annotation-status" checked={annotationStatus === "background"} disabled={boxes.length > 0} onChange={() => setAnnotationStatus("background")} />Ảnh âm tính: không có lửa hoặc khói ở bất kỳ vị trí nào</label></div>
          {annotationStatus !== "background" && <label>Lớp cho box mới<select value={drawClass} onChange={(event) => setDrawClass(Number(event.target.value) as 0 | 1)}><option value={0}>Khói</option><option value={1}>Lửa</option></select></label>}
        <p className="small muted">Gắn nhãn trên ảnh gốc bên dưới. Box dự đoán tham khảo chưa được lưu làm nhãn.</p>
          {suggestedBoxes.length > 0 && <button type="button" className="button secondary" onClick={importSuggestions}>Sao chép {suggestedBoxes.length} box dự đoán vào ảnh để kiểm tra</button>}
          <AnnotationEditor key={selected.id} imageUrl={selected.image_url} boxes={boxes} setBoxes={setBoxes} drawClass={drawClass} onDraw={() => setAnnotationStatus("objects")} />
        <p className="small muted">Kiểm tra, sửa hoặc xóa từng box dự đoán đã sao chép; thêm box cho mọi vùng lửa/khói còn thiếu trước khi duyệt. Chỉ chọn ảnh âm tính khi toàn ảnh không có lửa hoặc khói.</p>
        </>}
        <p className="small muted">{selected.realtime_session_id ? "Ảnh được chụp trực tiếp từ phiên webcam tại thời điểm user gửi phản hồi." : "Video kết quả chỉ để tham khảo. Không thể vẽ box trực tiếp trên video này."}</p>
        {!selected.realtime_session_id && <video key={selected.id} controls preload="metadata" src={`/api/model-admin/feedback/${selected.id}/video`} onLoadedMetadata={(event) => { event.currentTarget.currentTime = selected.video_time_ms / 1000; }} />}
        <label>Ghi chú duyệt<textarea value={reviewNote} onChange={(event) => setReviewNote(event.target.value)} rows={3} /></label><button className="button primary" disabled={saving || (decision === "accepted" && !annotationValid)}>{saving ? "Đang lưu..." : decision === "accepted" ? "Xác nhận đã kiểm tra toàn ảnh" : "Từ chối phản hồi"}</button>
      </form></> : <p className="empty-state">Chọn một phản hồi để xem chi tiết.</p>}</aside>
    </div>
  </div>;
}

function CandidatesView() {
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [metrics, setMetrics] = useState("");
  const [busy, setBusy] = useState(false);
  const notice = useMessage();
  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (!file) return;
    const form = new FormData();
    form.append("file", file); form.append("name", name); form.append("metrics_json", metrics);
    setBusy(true);
    try {
      await api("/api/model-admin/models", { method: "POST", body: form });
      notice.success("Đã đăng ký mô hình ứng viên. Admin vận hành sẽ xem báo cáo trước khi kích hoạt.");
      setFile(null); setName(""); setMetrics("");
    } catch (reason) { notice.fail(reason); }
    finally { setBusy(false); }
  }
  return <div className="page"><div className="page-header"><div><p className="eyebrow">Quản trị mô hình</p><h1>Đăng ký mô hình ứng viên</h1><p className="muted">Nộp checkpoint và kết quả đánh giá trên cùng bộ test. Web không tự huấn luyện mô hình.</p></div></div><Notice message={notice.message} error={notice.error} />
    <form className="candidate-form stack-form" onSubmit={submit}>
      <label>Tên phiên bản<input value={name} onChange={(event) => setName(event.target.value)} required placeholder="Ví dụ: yolo11n-v7-candidate" /></label>
      <label>Checkpoint .pt<input type="file" accept=".pt" onChange={(event) => setFile(event.target.files?.[0] || null)} required /></label>
      <label>Báo cáo metrics dạng JSON<textarea value={metrics} onChange={(event) => setMetrics(event.target.value)} rows={9} required placeholder='{"test_mAP50": 0.75, "negative_video_false_events": 0}' /></label>
      <p className="small muted">Bao gồm kết quả trên video âm tính và độ trễ cảnh báo. Chỉ admin vận hành được kích hoạt phiên bản.</p>
      <button className="button primary" disabled={busy}>{busy ? "Đang tải..." : "Đăng ký ứng viên"}</button>
    </form>
  </div>;
}

export default function App() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [area, setArea] = useState<Area>("videos");

  useEffect(() => {
    api<User>("/api/auth/me").then((value) => {
      setUser(value);
      setArea(homeArea(value));
    }).catch(() => undefined).finally(() => setLoading(false));
    const resetRestoredPage = (event: PageTransitionEvent) => {
      if (event.persisted) window.location.reload();
    };
    window.addEventListener("pageshow", resetRestoredPage);
    return () => window.removeEventListener("pageshow", resetRestoredPage);
  }, []);

  async function logout() {
    await api("/api/auth/logout", { method: "POST" });
    setUser(null);
  }

  if (loading) return <div className="loading-page"><Brand /><p>Đang mở hệ thống...</p></div>;
  if (!user) return <AuthView onLogin={(value) => { setUser(value); setArea(homeArea(value)); }} />;

  function renderNavIcon(itemArea: Area) {
    switch (itemArea) {
      case "videos":
        return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>;
      case "realtime":
        return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M23 12a11 11 0 1 1-22 0 11 11 0 0 1 22 0Z"/><circle cx="12" cy="12" r="3"/><path d="M12 2a10 10 0 0 1 10 10"/></svg>;
      case "profile":
        return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>;
      case "events":
        return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"/></svg>;
      case "jobs":
        return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>;
      case "accounts":
        return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>;
      case "versions":
        return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m7.5 4.27 9 5.15"/><path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/><path d="m3.3 7 8.7 5 8.7-5"/><path d="M12 22V12"/></svg>;
      case "reviews":
        return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="m9 12 2 2 4-4"/></svg>;
      case "candidates":
        return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242"/><path d="M12 12v9"/><path d="m16 16-4-4-4 4"/></svg>;
    }
  }

  const navigation: { area: Area; label: string; role: string }[] = [
    { area: "videos", label: "Video của tôi", role: "user" },
    { area: "realtime", label: "Webcam trực tiếp", role: "user" },
    { area: "profile", label: "Hồ sơ và Zalo", role: "user" },
    { area: "events", label: "Luồng cảnh báo", role: "admin_ops" },
    { area: "jobs", label: "Lịch sử video", role: "admin_ops" },
    { area: "accounts", label: "Tài khoản", role: "admin_ops" },
    { area: "versions", label: "Phiên bản mô hình", role: "admin_ops" },
    { area: "reviews", label: "Phản hồi sai", role: "admin_model" },
    { area: "candidates", label: "Mô hình ứng viên", role: "admin_model" },
  ];

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="sidebar-top"><Brand /></div>
      <nav aria-label="Điều hướng">
        {navigation.filter((item) => user.roles.includes(item.role as never) && !(item.role === "user" && user.roles.some((role) => role.startsWith("admin_")))).map((item) => (
          <button key={item.area} className={area === item.area ? "active" : ""} onClick={() => setArea(item.area)}>
            {renderNavIcon(item.area)}
            <span>{item.label}</span>
          </button>
        ))}
      </nav>
      <div className="sidebar-bottom">
        <div className="sidebar-user-info">
          <div className="sidebar-avatar" aria-hidden="true">
            {(user.profile.full_name || user.username).charAt(0).toUpperCase()}
          </div>
          <div className="sidebar-user-details">
            <strong>{user.profile.full_name || user.username}</strong>
            <small>{user.roles.join(" · ")}</small>
          </div>
        </div>
        <button className="logout-button" onClick={logout}>
          <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/></svg>
          Đăng xuất
        </button>
      </div>
    </aside>
    <main className="content">
      {area === "videos" && !user.roles.some((role) => role.startsWith("admin_")) && <VideosView />}
      {area === "realtime" && !user.roles.some((role) => role.startsWith("admin_")) && <RealtimeView />}
      {area === "profile" && !user.roles.some((role) => role.startsWith("admin_")) && <ProfileView />}
      {area === "events" && <EventsView />}
      {area === "jobs" && <AdminJobsView />}
      {area === "accounts" && <AccountsView />}
      {area === "versions" && <VersionsView />}
      {area === "reviews" && <ReviewsView />}
      {area === "candidates" && <CandidatesView />}
    </main>
  </div>;
}
