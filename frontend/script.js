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

  try {
    const response = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      body: formData
    });

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
    loading.classList.add("hidden");
    alert("An error occurred while analyzing the image.");
    console.error(error);
  }
});