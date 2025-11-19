// api.js

// Define the production URL for your deployed FastAPI backend
const BACKEND_BASE_URL = "https://accubattery.onrender.com";

export async function uploadCSVToBackend(file) {
  const formData = new FormData();
  // Ensure the key "file" matches the FastAPI endpoint definition:
  // async def predict_csv(file: UploadFile = File(...)):
  formData.append("file", file);

  const res = await fetch(`${BACKEND_BASE_URL}/predict_csv`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    // Attempt to parse error message from backend if available
    const errorBody = await res.json().catch(() => ({}));
    const errorMessage = errorBody.error || `Backend returned error with status ${res.status}`;
    throw new Error(errorMessage);
  }

  return await res.json();
}
