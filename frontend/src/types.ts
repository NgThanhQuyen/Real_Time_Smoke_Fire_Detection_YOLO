export type Role = "user" | "admin_ops" | "admin_model";

export type User = {
  id: string;
  username: string;
  active: boolean;
  roles: Role[];
  profile: { full_name: string | null; phone: string | null; address: string | null };
  created_at: string;
};

export type JobStatus = "queued" | "processing" | "finalizing" | "completed" | "failed" | "interrupted";

export type AlertEvent = {
  id: string;
  job_id: string | null;
  class_id: number;
  start_ms: number;
  end_ms: number;
  peak_confidence: number;
  created_at: string;
  evidence_id: string | null;
};

export type VideoJob = {
  id: string;
  original_name: string;
  status: JobStatus;
  duration_ms: number;
  source_fps: number;
  total_frames: number;
  processed_frames: number;
  detected_frames: number;
  inference_ms: number | null;
  processing_ms: number | null;
  model_version_id: string;
  error_code: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  events_count: number;
  events?: AlertEvent[];
  video_url?: string | null;
};

export type Detection = {
  id: string;
  frame_index: number;
  video_time_ms: number;
  class_id: number;
  confidence: number;
  box: [number, number, number, number];
};

export type DetectedFrame = {
  frame_index: number;
  video_time_ms: number;
  confidence: number;
  image_url: string;
};

export type Feedback = {
  id: string;
  job_id: string | null;
  realtime_session_id?: string | null;
  realtime_event_id?: string | null;
  user_id: string;
  detection_id: string | null;
  video_time_ms: number;
  image_url: string;
  type: string;
  expected_class: number | null;
  proposed_box: Record<string, number> | null;
  comment: string | null;
  status: string;
  created_at: string;
  video_name?: string;
  username?: string;
  annotation_status?: "objects" | "background" | null;
  annotations?: { class_id: 0 | 1; x1: number; y1: number; x2: number; y2: number }[];
  review_note?: string | null;
};

export type AdminEvent = AlertEvent & {
  job_id: string | null;
  source_type: "video" | "webcam";
  session_id?: string | null;
  video_name: string;
  video_status: string;
  user_id: string;
  username: string;
  contact: { full_name: string | null; phone: string | null; address: string | null };
  notification_status: string;
  evidence_url: string | null;
};

export type RealtimeEvent = {
  id: string;
  session_id: string;
  class_id: number;
  start_ms: number;
  end_ms: number;
  peak_confidence: number;
  status: string;
  created_at: string;
  evidence_id: string | null;
  evidence_url: string | null;
};

export type RealtimeSession = {
  id: string;
  status: string;
  model_version_id: string;
  started_at: string;
  ended_at: string | null;
  duration_ms: number;
  frame_count: number;
  detected_frame_count: number;
  events_count: number;
  events: RealtimeEvent[];
};

export type AdminJob = VideoJob & {
  username: string;
  contact: { full_name: string | null; phone: string | null; address: string | null };
};

export type ModelVersion = {
  id: string;
  name: string;
  sha256: string;
  class_map: Record<string, string>;
  metrics: Record<string, unknown>;
  config: Record<string, unknown>;
  status: string;
  created_at: string;
  activated_at: string | null;
};

export type Profile = {
  full_name: string | null;
  phone: string | null;
  address: string | null;
  zalo_status: "unconfigured" | "saved" | "verified";
};

export type MlflowStatus = {
  configured: boolean;
  connected: boolean;
  message: string;
  tracking_uri: string;
};

export type MlflowRun = {
  run_id: string;
  run_name: string;
  status: string;
  start_time: string | null;
  params: Record<string, string>;
  metrics: Record<string, number>;
  has_checkpoint: boolean;
  checkpoint_path: string;
  is_imported: boolean;
};

export type DatasetStatus = {
  is_ready: boolean;
  images_count: number;
  labels_count: number;
  total_mb: number;
  folder_path: string;
  images_dir: string | null;
  labels_dir: string | null;
  yaml_path: string | null;
  recommended_count: number;
};

export type ClassMetric = {
  precision: number;
  recall: number;
  map50: number;
  map50_95: number;
};

export type BenchmarkMetricsSummary = {
  overall: {
    precision: number;
    recall: number;
    map50: number;
    map50_95: number;
    f1: number;
  };
  classes: {
    khoi: ClassMetric;
    lua: ClassMetric;
  };
  speed: {
    preprocess_ms: number;
    inference_ms: number;
    loss_ms: number;
    postprocess_ms: number;
    total_duration_s: number;
    fps: number;
  };
  dataset_info: {
    image_count: number;
    labels_count: number;
  };
};

export type BenchmarkRun = {
  id: string;
  model_version_id: string;
  model_name: string | null;
  dataset_name: string;
  status: "queued" | "running" | "completed" | "failed";
  image_count: number;
  metrics_summary: BenchmarkMetricsSummary;
  error_message: string | null;
  created_by: string | null;
  created_at: string;
  completed_at: string | null;
};

export type BenchmarkComparison = {
  candidate: { id: string; name: string; benchmark_id: string; completed_at: string };
  base: { id: string; name: string; benchmark_id: string; completed_at: string };
  comparison: {
    verdict: "improved" | "regressed" | "comparable";
    verdict_text: string;
    overall: {
      map50: { candidate: number; base: number; delta: number; pct: number };
      map50_95: { candidate: number; base: number; delta: number };
      precision: { candidate: number; base: number; delta: number };
      recall: { candidate: number; base: number; delta: number };
      speed_ms: { candidate: number; base: number; delta: number };
    };
    classes: {
      khoi: { candidate: ClassMetric; base: ClassMetric; delta_map50: number };
      lua: { candidate: ClassMetric; base: ClassMetric; delta_map50: number };
    };
  };
};

export type RetrainRun = {
  id: string;
  kernel_slug: string;
  status: "queued" | "running" | "completed" | "failed";
  feedback_count: number;
  base_model_name: string;
  kaggle_status: string | null;
  mlflow_run_id: string | null;
  error_message: string | null;
  created_by: string | null;
  created_at: string;
  completed_at: string | null;
  kaggle_url: string;
};

export type RetrainStats = {
  accepted_total: number;
  hard_negatives: number;
  hard_positives: number;
  total_boxes: number;
  pending_reviews: number;
  active_model_name: string;
  active_model_id: string | null;
  kaggle_configured: boolean;
  kaggle_username: string;
  latest_run: RetrainRun | null;
};


