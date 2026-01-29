# 🏥 OrOeye: Deep Learning Oral Lesion Analysis

**This isn't just another classification script.**

OrOeye is a production-grade, full-stack medical AI platform designed to detect potential oral anomalies with explainable precision. We moved beyond simple "black box" predictions to build a system that medical professionals can actually verify.

It combines **Computer Vision (CNNs)** for detection, **Heuristic Forensics** for data validation, and **Generative AI** for system analytics into a single, cohesive architecture.

---

## ⚡ The "Heavy Lifting" (Technical Deep Dive)

We didn't just `pip install tensorflow` and call it a day. Here is what's actually happening under the hood:

### 1. Explainable AI (XAI) with Grad-CAM
Most AI models just say "Cancer: 90%". That's useless for a doctor.
*   **What we built:** We implemented **Gradient-weighted Class Activation Mapping (Grad-CAM)**.
*   **How it works:** We hook into the final convolutional layer of our **InceptionResNetV2** model to extract the gradients. We then overlay a heatmap on the original image, showing you *exactly* which pixels triggered the prediction.
*   **The Result:** You don't just get a diagnosis; you get visual proof of the lesion the AI detected.

### 2. Heuristic Image Forensics (The "Gatekeeper")
Garbage In, Garbage Out. We built a custom validation engine using **OpenCV** that runs *before* the AI even touches the image. It uses physics-based heuristics to reject invalid data:
*   **Spectral Analysis:** Rejects images with dominant blue/sky tones (outdoors/non-medical).
*   **Flesh-Tone Ratios:** Calculates the saturation of specific HSV ranges to ensure the image contains mucosal tissue.
*   **Face-Rejection Logic:** Uses Haar Cascades to detect if a user uploaded a "selfie" instead of a zoomed-in oral cavity shot. If it sees a face, it blocks the upload.

### 3. Hybrid AI Architecture
We aren't relying on one model.
*   **The Eye:** A fine-tuned **InceptionResNetV2** (CNN) handles the visual diagnostics.
*   **The Brain:** We integrated **LLMs (GPT-4)** into the Admin Dashboard (`/admin`). It analyzes system logs, user patterns, and scan statistics to provide natural language insights to administrators.

---

## 🛠️ System Architecture

*   **Backend:** Python (Flask) serving as the REST API and inference engine.
*   **Database:** MongoDB Atlas (NoSQL) for flexible user data and scan history storage.
*   **Auth:** Secure session management with `bcrypt` encryption.
*   **Reporting:** Automated PDF generation engine (`ReportLab`) that compiles the original image, the AI heatmap, and clinical notes into a downloadable medical report.

---

## 🚀 How to Run It

### 1. Clone & Prep
```bash
git clone https://github.com/thisistanishq/oroeye.git
cd oroeye
```

### 2. Virtual Env (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows
```

### 3. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 4. Run the Server
```bash
# We handle the frontend serving directly from Flask for speed
python wsgi.py
# OR
cd backend && python app.py
```

Visit `http://127.0.0.1:5001`.

---

## 🔐 Admin Dashboard

The system includes a fully equipped **Admin Dashboard** for clinic managers.

*   **Real-Time Analytics:** Live tracking of daily/weekly scans, cancer detection rates, and user growth.
*   **System Logs:** Real-time event logging engine that aggregates user actions, scans, and system warnings.
*   **Message Center:** Full management interface for contact form submissions.
*   **User Management:** Search, filter, and manage user accounts and access roles.



---

## ⚠️ Simulation Mode

**Note:** The trained model files (`.h5` / `.keras`) are massive (~200MB+).
If you clone this repo without pulling the LFS files, the system detects the missing model and automatically switches to **Simulation Mode**.
*   It will function normally but return simulated "Non-Cancerous" results to prevent crashing.
*   To get real predictions, ensure you have the model files in `backend/model/`.

---

**Built by Tanishq.** Code that actually works.
