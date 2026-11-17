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
