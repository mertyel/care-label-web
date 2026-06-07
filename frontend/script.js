const API_BASE_URL = "https://care-label-api.onrender.com";

const imageInput = document.getElementById("imageInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const loading = document.getElementById("loading");
const resultSection = document.getElementById("resultSection");
const resultImage = document.getElementById("resultImage");
const detectionsList = document.getElementById("detectionsList");

analyzeBtn.addEventListener("click", async () => {
  const file = imageInput.files[0];

  if (!file) {
    alert("Please select or capture an image first.");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  loading.classList.remove("hidden");
  resultSection.classList.add("hidden");
  detectionsList.innerHTML = "";

  analyzeBtn.disabled = true;
  analyzeBtn.innerText = "Analyzing...";

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 90000);

  try {
    console.log("Sending image to backend:", `${API_BASE_URL}/predict`);

    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      body: formData,
      signal: controller.signal
    });

    clearTimeout(timeoutId);

    console.log("Backend response status:", response.status);

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const data = await response.json();

    loading.classList.add("hidden");
    resultSection.classList.remove("hidden");

    resultImage.src = data.result_image_url + "?t=" + new Date().getTime();

    if (!data.detections || data.detections.length === 0) {
      detectionsList.innerHTML = `
        <div class="detection-card">
          No care label symbols were detected.
        </div>
      `;
      return;
    }

    detectionsList.innerHTML = "";

    data.detections.forEach((item, index) => {
      const card = document.createElement("div");
      card.className = "detection-card";

      card.innerHTML = `
        <strong>${index + 1}. ${item.label}</strong><br>
        Confidence: ${(item.confidence * 100).toFixed(1)}%<br>
        Meaning: ${item.description}
      `;

      detectionsList.appendChild(card);
    });

  } catch (error) {
    clearTimeout(timeoutId);

    loading.classList.add("hidden");
    resultSection.classList.add("hidden");

    if (error.name === "AbortError") {
      alert("The request took too long. Render backend is slow or the uploaded image is too large.");
    } else {
      alert("An error occurred while analyzing the image. Please check the browser console.");
    }

    console.error("Frontend error:", error);

  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.innerText = "Analyze Image";
  }
});