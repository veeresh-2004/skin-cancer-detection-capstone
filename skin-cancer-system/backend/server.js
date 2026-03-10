const express = require("express");
const multer = require("multer");
const axios = require("axios");
const FormData = require("form-data");
const fs = require("fs");
const cors = require("cors");
const path = require("path");

const app = express();
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || "http://127.0.0.1:10000";

app.use(cors({
  origin: ["http://127.0.0.1:5500", "http://127.0.0.1:3000", "http://localhost:5500", "http://localhost:3000", "http://localhost:5174"],
  credentials: true
}));

// Serve the frontend statically so visiting http://127.0.0.1:3000 shows the UI
const frontendDir = path.join(__dirname, "..", "frontend");
app.use(express.static(frontendDir));

// Ensure uploads directory exists
const uploadsDir = path.join(__dirname, "uploads");
if (!fs.existsSync(uploadsDir)) {
    fs.mkdirSync(uploadsDir);
}

const upload = multer({ dest: "uploads/" });

app.post("/predict", upload.any(), async (req, res) => {
    const uploadedFiles = req.files || [];
    const imageFiles = uploadedFiles.filter(
        (f) => f.fieldname === "image" || f.fieldname === "images"
    );

    try {
        if (!imageFiles.length) {
            return res.status(400).json({ error: "No image uploaded" });
        }

        console.log(`📤 Forwarding ${imageFiles.length} image(s) to ML Service...`);

        const form = new FormData();
        let endpointPath = "/predict";

        if (imageFiles.length > 1) {
            endpointPath = "/predict-multiview";
            imageFiles.forEach((file) => {
                form.append("images", fs.createReadStream(file.path), {
                    filename: file.originalname,
                });
            });

            if (req.body && req.body.view_labels) {
                form.append("view_labels", req.body.view_labels);
            }
        } else {
            const [file] = imageFiles;
            form.append("image", fs.createReadStream(file.path), {
                filename: file.originalname,
            });
        }

        const response = await axios.post(
            `${ML_SERVICE_URL}${endpointPath}`,
            form,
            { 
                headers: form.getHeaders(),
                timeout: 30000 // 30 second timeout
            }
        );

        console.log("✅ Prediction received:", response.data.label);
        res.json(response.data);

    } catch (err) {
        console.error("❌ Backend Error:", err.message);

        if (err.code === 'ECONNREFUSED') {
            res.status(503).json({ error: "ML Service is not running. Please start the Python service." });
        } else if (err.response && err.response.data) {
            res.status(err.response.status || 500).json(err.response.data);
        } else {
            res.status(500).json({ error: " ❌ Invalid Image ❌ || ✅ Please Select valid skin lesion ✅"  });
        }
    } finally {
        imageFiles.forEach((file) => {
            if (file.path && fs.existsSync(file.path)) {
                fs.unlinkSync(file.path);
            }
        });
    }
});

app.get("/health", (req, res) => {
    res.json({ status: "Backend is running" });
});

// Root route serves the frontend entrypoint
app.get("/", (req, res) => {
    res.sendFile(path.join(frontendDir, "index.html"));
});

app.listen(3000, () => {
    console.log("🚀 Node backend running on http://127.0.0.1:3000");
    console.log(`🤖 ML Service target: ${ML_SERVICE_URL}`);
    console.log("💡 Waiting for requests from frontend...");
});
