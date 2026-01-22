const express = require("express");
const multer = require("multer");
const axios = require("axios");
const FormData = require("form-data");
const fs = require("fs");
const cors = require("cors");
const path = require("path");

const app = express();
app.use(cors({
  origin: ["http://127.0.0.1:5500", "http://127.0.0.1:3000", "http://localhost:5500", "http://localhost:3000"],
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

app.post("/predict", upload.single("image"), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: "No image uploaded" });
        }

        console.log("📤 Forwarding image to ML Service...");

        const form = new FormData();
        form.append(
            "image",
            fs.createReadStream(req.file.path),
            { filename: req.file.originalname }
        );

        const response = await axios.post(
            "http://127.0.0.1:5000/predict",
            form,
            { 
                headers: form.getHeaders(),
                timeout: 30000 // 30 second timeout
            }
        );

        // Cleanup uploaded file
        fs.unlinkSync(req.file.path);

        console.log("✅ Prediction received:", response.data.label);
        res.json(response.data);

    } catch (err) {
        console.error("❌ Backend Error:", err.message);
        
        // Cleanup file on error
        if (req.file && fs.existsSync(req.file.path)) {
            fs.unlinkSync(req.file.path);
        }

        if (err.code === 'ECONNREFUSED') {
            res.status(503).json({ error: "ML Service is not running. Please start the Python service." });
        } else {
            res.status(500).json({ error: "Prediction failed: " + err.message });
        }
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
    console.log("💡 Waiting for requests from frontend...");
});
