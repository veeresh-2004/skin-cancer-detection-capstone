// Always target the Node backend on port 3000
const BACKEND_URL = "http://127.0.0.1:3000/predict";
console.log("🔧 BACKEND_URL:", BACKEND_URL);

document.addEventListener("DOMContentLoaded", () => {
  console.log("🔧 script.js DOMContentLoaded");

  const uploadForm = document.getElementById("uploadForm");
  const imageInput = document.getElementById("imageInput");
  const previewImg = document.getElementById("previewImg");
  const previewPlaceholder = document.getElementById("previewPlaceholder");
  const predictBtn = document.getElementById("predictBtn");
  const result = document.getElementById("result");
  const error = document.getElementById("error");
  const clipStatus = document.getElementById("clipStatus");
  const gradcamImg = document.getElementById("gradcamImg");
  const gradcamPlaceholder = document.getElementById("gradcamPlaceholder");

  if (!imageInput || !predictBtn) {
    console.error("❌ imageInput or predictBtn not found");
    return;
  }

  // Monitor for page navigation
  window.addEventListener("beforeunload", (e) => {
    console.warn("⚠️ Page unload/refresh detected!");
  });

  // Prevent form submission (in case Enter key is pressed)
  if (uploadForm && uploadForm.tagName === "FORM") {
    uploadForm.addEventListener("submit", (e) => {
      e.preventDefault();
      e.stopPropagation();
      console.log("🛑 Form submission prevented");
      return false;
    });
  }

  // Preview uploaded image
  imageInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      previewImg.src = event.target.result;
      previewPlaceholder.style.display = "none";
      gradcamImg.src = "";
      gradcamPlaceholder.style.display = "block";
      gradcamPlaceholder.textContent = "⏳ Waiting for analysis...";
      result.innerText = "";
      error.innerText = "";
      clipStatus.innerText = "";
      console.log("✅ Image selected");
    };
    reader.readAsDataURL(file);
  });

  // Define the analyze click handler function
  async function handleAnalyzeClick(event) {
    if (event) {
      event.preventDefault();
      event.stopPropagation();
    }
    
    const file = imageInput.files[0];
    if (!file) {
      error.innerText = "Please select an image first";
      result.innerText = "";
      console.log("⚠️ No file selected");
      return false;
    }

    // Disable button during processing
    predictBtn.disabled = true;
    predictBtn.style.opacity = "0.5";
    
    result.innerText = "Analyzing...";
    error.innerText = "";
    clipStatus.innerText = "";

    const formData = new FormData();
    formData.append("image", file);

    try {
      console.log("📤 Sending request to:", BACKEND_URL);
      const response = await fetch(BACKEND_URL, { 
        method: "POST", 
        body: formData,
        credentials: "same-origin"
      });

      console.log("✅ Response received, status:", response.status);
      console.log("✅ Response type:", response.type);
      console.log("✅ Response URL:", response.url);
      console.log("✅ Content-Type:", response.headers.get("content-type"));
      
      previewPlaceholder.style.display = "none";

      if (!response.ok) {
        const rawError = await response.text();
        console.error("❌ Backend error response:", rawError);

        let userMessage = `Server error: ${response.status} ${response.statusText}`;
        if (rawError) {
          try {
            const parsed = JSON.parse(rawError);
            if (parsed.error) {
              userMessage = parsed.error;
            }
          } catch (parseErr) {
            userMessage = rawError;
          }
        }

        result.innerText = "";
        gradcamImg.style.display = "none";
        gradcamPlaceholder.style.display = "block";
        gradcamPlaceholder.textContent = "Please upload a valid skin lesion image to generate Grad-CAM.";
        clipStatus.innerText = "";

        if (response.status === 400 && userMessage.toLowerCase().includes("skin lesion")) {
          clipStatus.innerText = "Invalid image detected. Please upload a clear skin lesion photo.";
          clipStatus.style.color = "#d32f2f";
          error.innerText = "";
        } else {
          error.innerText = userMessage;
        }

        predictBtn.disabled = false;
        predictBtn.style.opacity = "1";
        return false;
      }

      const contentType = response.headers.get("content-type");
      if (!contentType || !contentType.includes("application/json")) {
        const responseText = await response.text();
        console.error("❌ Response is not JSON, got:", contentType);
        console.error("❌ Response body:", responseText);
        error.innerText = `Invalid response type: ${contentType}`;
        result.innerText = "";
        clipStatus.innerText = "";
        predictBtn.disabled = false;
        predictBtn.style.opacity = "1";
        return false;
      }

      const data = await response.json();
      console.log("📥 Backend response:", data);

      if (data.error) {
        error.innerText = data.error;
        result.innerText = "";
        clipStatus.innerText = "";
        predictBtn.disabled = false;
        predictBtn.style.opacity = "1";
        return false;
      }

      // Show both fraction (0-1) and percentage for clarity
      const frac = (typeof data.confidence === 'number') ? data.confidence : (data.confidence_percent ? data.confidence_percent / 100 : null);
      const pct = (typeof data.confidence_percent === 'number') ? data.confidence_percent : (frac !== null ? frac * 100 : null);
      if (frac !== null && pct !== null) {
        result.innerText = `Prediction: ${data.label}\nConfidence: ${frac.toFixed(4)} (${pct.toFixed(2)}%)`;
      } else if (typeof data.confidence === 'number') {
        result.innerText = `Prediction: ${data.label}\nConfidence: ${data.confidence}`;
      } else {
        result.innerText = `Prediction: ${data.label}`;
      }

      if (data.clip_validation) {
        clipStatus.innerText = data.clip_validation;
        clipStatus.style.color = "#2e7d32";
      } else {
        clipStatus.innerText = "";
      }

      // Display melanoma stage if available
      const stageSection = document.getElementById("stageSection");
      const stageLabel = document.getElementById("stageLabel");
      if (data.stage) {
        stageLabel.innerText = data.stage;
        stageSection.style.display = "block";
        console.log("🎭 Melanoma stage displayed:", data.stage);
      } else {
        stageSection.style.display = "none";
        console.log("ℹ️ No stage data (benign prediction)");
      }

      if (previewImg.src) {
        previewImg.style.display = "block";
        previewImg.style.width = "100%";
        previewImg.style.maxWidth = "400px";
        console.log("✅ Original preview image displayed");
      }

      if (data.gradcam_image && data.gradcam_image.length > 0) {
        console.log("✅ Grad-CAM image received, length:", data.gradcam_image.length);
        gradcamPlaceholder.style.display = "none";
        gradcamImg.src = "data:image/png;base64," + data.gradcam_image;
        gradcamImg.style.display = "block";
        gradcamImg.onload = () => console.log("✅ Grad-CAM image loaded");
        gradcamImg.onerror = (err) => {
          console.error("❌ Failed to load Grad-CAM image:", err);
          error.innerText = "Failed to load Grad-CAM visualization";
        };
      } else {
        console.warn("⚠️ No gradcam_image in response or empty");
        gradcamImg.style.display = "none";
      }
      
      predictBtn.disabled = false;
      predictBtn.style.opacity = "1";
      
    } catch (err) {
      console.error("❌ Error:", err);
      console.error("❌ Error stack:", err.stack);
      error.innerText = `Error: ${err.message}`;
      result.innerText = "";
      clipStatus.innerText = "";
      predictBtn.disabled = false;
      predictBtn.style.opacity = "1";
    }
    
    return false;
  }

  // Attach single click listener to button
  predictBtn.addEventListener("click", handleAnalyzeClick);
});