export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

function detailMessage(value: unknown): string {
  if (typeof value === "string") return value;
  if (Array.isArray(value)) return value.map((item) => item.msg || "Dữ liệu không hợp lệ").join("; ");
  return "Yêu cầu không thành công";
}

export async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(path, {
    credentials: "same-origin",
    ...options,
    headers: {
      ...(options.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...options.headers,
    },
  });
  if (!response.ok) {
    let message = response.statusText;
    try {
      const body = await response.json();
      message = detailMessage(body.detail);
    } catch {
      message = "Không thể đọc phản hồi từ máy chủ";
    }
    throw new ApiError(message, response.status);
  }
  if (response.headers.get("content-type")?.includes("application/json")) {
    return (await response.json()) as T;
  }
  return response as T;
}

export function json(method: string, body: unknown): RequestInit {
  return { method, body: JSON.stringify(body) };
}

export function formatDate(value: string | null | undefined): string {
  return value
    ? new Intl.DateTimeFormat("vi-VN", {
        dateStyle: "short",
        timeStyle: "medium",
        timeZone: "Asia/Ho_Chi_Minh",
      }).format(new Date(value))
    : "Chưa có";
}

export function formatVideoTime(milliseconds: number): string {
  const seconds = Math.max(0, Math.floor(milliseconds / 1000));
  return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}

export function className(classId: number): string {
  return classId === 0 ? "Khói" : "Lửa";
}
