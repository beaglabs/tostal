export interface SessionUser {
  id: string;
  email: string;
  name?: string;
  image?: string;
}

export interface ApiError {
  detail: string;
  code?: string;
}

export interface Notebook {
  id: string;
  display_id: string;
  name: string;
  description?: string;
  cell_count: number;
  status: string;
  created_at: string;
  updated_at: string;
}

export interface JobStatus {
  job_id: string;
  status: "pending" | "processing" | "completed" | "failed" | "canceled";
  result_path?: string;
  result_shape?: number[];
  classes?: string[];
  confidence?: number;
  rendering_url?: string;
}