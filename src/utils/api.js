export async function uploadCSVToBackend(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch("http://127.0.0.1:8000/predict_csv", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error("Backend returned error");
  }

  return await res.json();
}
