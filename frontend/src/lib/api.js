// Response bodies cannot be assumed to be JSON, and `empty` records whether a
// body existed at all: the Vite dev proxy answers 5xx with a zero-byte body when
// it cannot reach the API, and response.json() throws on that.
export async function readJson(response) {
  const text = await response.text();
  if (!text) return { payload: {}, empty: true };
  try {
    return { payload: JSON.parse(text), empty: false };
  } catch {
    return { payload: {}, empty: false };
  }
}

// FastAPI puts string errors in `detail`, but validation errors arrive as a list
// of objects, so normalise both into something the UI can show directly.
export function readError(result, response) {
  const detail = result.payload?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((item) => item.msg).join("; ");

  // An empty 5xx is the dev proxy failing to reach the API, not an error from it:
  // there is nothing in the body to explain, so name the likely cause instead.
  if (result.empty && response.status >= 500) {
    return `The API is not responding (status ${response.status}). Is uvicorn running?`;
  }
  return `Request failed with status ${response.status}`;
}