# CastLens: Casting Defect Detection 🔍

CastLens is a machine learning web application that automatically inspects top-down photos of industrial castings and classifies them as either **OK** or **Defective**. 

The app features a FastAPI + ONNX Runtime backend and a modern React + Framer Motion frontend.

🔗 **The codebase is live on GitHub:** [https://github.com/thenun123/Castlens](https://github.com/thenun123/Castlens)

---

## 📊 Exploratory Data Analysis (EDA)

Before building the model, an extensive EDA was performed to understand the dataset:
- **Class Balance:** The dataset contains casting photos categorized into `ok_front` (acceptable) and `def_front` (defective). The dataset is generally well-balanced but requires careful handling to prevent model bias.
- **Image Characteristics:** All images are top-down views of circular casting parts. Some defects are very subtle (small chips, cracks, or blowholes), while others are obvious surface deformities.
- **Lighting & Shadows:** Lighting conditions vary slightly across the dataset. The EDA revealed that a model could easily overfit to shadows rather than actual defects, necessitating strong data augmentation.

## 🧠 Training Setup

The model was trained using PyTorch with the following pipeline:
- **Data Augmentation:** To combat overfitting and make the model robust to real-world production environments, we applied augmentations such as random rotations, flips, and slight color jittering.
- **Optimizer & Scheduler:** We used the AdamW optimizer with a cosine annealing learning rate scheduler to smoothly converge to the optimal weights.
- **Loss Function:** We used Cross-Entropy Loss, heavily tuned based on validation metrics to ensure high recall for defects.

## 🤖 Why EfficientNet-B0?

For this task, we selected **EfficientNet-B0** as our core architecture. Here is why:
1. **Lightweight & Fast:** It has very few parameters compared to older models like ResNet50, making it extremely fast for real-time inference (typically under 100ms per image).
2. **High Accuracy:** Despite its small size, EfficientNet uses compound scaling (balancing depth, width, and resolution) to achieve state-of-the-art accuracy, perfectly capturing subtle cracks and blowholes on the castings.
3. **Deployment Friendly:** Because it is lightweight, it easily fits into a Docker container and can run effortlessly on cloud free-tiers (like Render's 512MB RAM limit) without requiring an expensive GPU.

## ⚖️ The Inspection Threshold

In industrial quality control, missing a defect (False Negative) is usually much more expensive than accidentally flagging a good part for manual review (False Positive). 

To account for this, the model does **not** simply output whichever class has a probability > 50%. Instead, we use a custom **Cost-Tuned Inspection Threshold of 0.21** (21%).
- If the probability of a defect is **> 21%**, the part is flagged as **Defective**.
- **Borderline Cases:** If a part has a 30% defect probability, the model actually thinks it is *probably* OK (70%). However, because 30% crosses our strict 21% safety threshold, the API correctly flags it as a defect and triggers a `borderline: true` warning for manual inspection.

---

## 💻 Local Setup (Development)

To run the application locally on your machine:

1. **Install Backend Dependencies:**
   ```bash
   cd backend
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements-dev.txt
   ```

2. **Start the FastAPI Backend:**
   ```bash
   make dev-api
   ```
   *The API will run on http://127.0.0.1:8000*

3. **Start the React Frontend:**
   Open a new terminal and run:
   ```bash
   make dev-web
   ```
   *The web UI will run on http://localhost:5173*

> **Note on Render Free Tier:** If deployed on Render's free tier, the backend goes to sleep after inactivity. The frontend contains a built-in banner that detects this and politely asks the user to wait (~50 seconds) while the instance wakes up!

---

## 🐳 Docker Setup (Production)

The entire application (frontend and backend) is packaged into a single, highly optimized Docker container. The frontend is built into static files and served directly by FastAPI.

1. **Build the Docker Image:**
   ```bash
   make build
   ```
   *(This command runs `docker build -t castlens .`)*

2. **Run the Docker Container:**
   ```bash
   make run
   ```
   *(This runs the container and exposes it on port 10000)*

3. **Use the App:**
   Open your browser and navigate to **http://localhost:10000**. 

---

### Folder Structure
```text
castlens/
├── backend/            FastAPI app (app/), tests/, requirements*.txt
├── frontend/           Vite + React + TypeScript + Framer Motion
├── model/              trained model (.pt source, .onnx runtime, inference_config.json)
├── notebooks/          01_eda.ipynb, 02_training.ipynb
├── scripts/            export_onnx.py  (.pt -> .onnx + parity check)
├── Dockerfile          Production multi-stage build
├── render.yaml         Blueprint for Render deployment
└── Makefile            Shortcuts for dev, build, and testing
```
