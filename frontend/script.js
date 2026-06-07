const API_BASE_URL = "https://care-label-api.onrender.com";

const imageInput = document.getElementById("imageInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const loading = document.getElementById("loading");
const resultSection = document.getElementById("resultSection");
const resultImage = document.getElementById("resultImage");
const detectionsList = document.getElementById("detectionsList");

async function resizeImageBeforeUpload(file, maxSize = 1000, quality = 0.75) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const reader = new FileReader();

    reader.onload = (event) => {
      img.onload = () => {
        let width = img.width;
        let height = img.height;

        if (width > height && width > maxSize) {
          height = Math.round((height * maxSize) / width);
          width = maxSize;
        } else if (height > width && height > maxSize) {
          width = Math.round((width * maxSize) / height);
          height = maxSize;
        }

        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;

        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0, width, height);

        canvas.toBlob(
          (blob) => {
            if (!blob) {
              reject(new Error("Image compression failed."));
              return;
            }

            const resizedFile = new File([blob], "resized_upload.jpg", {
              type: "image/jpeg"
            });

            resolve(resizedFile);
          },
          "image/jpeg",
          quality
        );
      };

      img.onerror = reject;
      img.src = event.target.result;
    };

    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

analyzeBtn.addEventListener("click", async () => {
  const file = imageInput.files[0];

  if (!file) {
    alert("Please select or capture an image first.");
    return;
  }

  loading.classList.remove("hidden");
  resultSection.classList.add("hidden");
  detectionsList.innerHTML = "";

  analyzeBtn.disabled = true;
  analyzeBtn.innerText = "Analyzing...";

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 60000);

  try {
    const resizedFile = await resizeImageBeforeUpload(file, 1000, 0.75);

    const formData = new FormData();
    formData.append("file", resizedFile);

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

    console.log("Processing time:", data.processing_time_seconds, "seconds");

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
      alert("The request took too long. Please try again with a smaller image.");
    } else {
      alert("An error occurred while analyzing the image.");
    }

    console.error("Frontend error:", error);

  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.innerText = "Analyze Image";
  }
});