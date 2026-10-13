# EcoSort

EcoSort is an intelligent, real-time waste classification system utilizing computer vision and deep learning methodologies to automatically identify household waste materials and deliver precise disposal, recycling, or composting recommendations.

---

## Project Description

The system enables users to capture or upload images of various waste items through a streamlined interface. An integrated deep learning model analyzes the visual input, evaluates the item's features, and instantly displays the predicted material category, confidence score, and appropriate disposal pathway to reduce sorting errors and enhance recycling efficiency.

---

## Key Features

* **Real-Time Image Classification:** Instantly processes uploaded or captured waste item images using trained computer vision models.
* **Automated Categorization:** Classifies everyday household refuse into designated material categories.
* **Disposal Recommendations:** Provides targeted guidelines and recommendations for proper waste disposal and recycling.
* **Confidence Scoring:** Outputs model confidence metrics alongside classification results to ensure transparency and reliability.

---

## Members

* John Ernest M. Beloy
* Jacinth Ulpiah N. Evangelista

## Table of Contents

- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [1. Environment Setup](#1-environment-setup)
  - [Windows](#windows)
  - [Linux](#linux)
- [2. Running the System](#2-running-the-system)
  - [Step 1: Start the Backend API](#step-1-start-the-backend-api)
  - [Step 2: Start the Frontend GUI](#step-2-start-the-frontend-gui)
- [Stopping the Application](#stopping-the-application)
- [Troubleshooting](#troubleshooting)

## How It Works

1. The user selects a `.jpg` or `.png` image in the desktop GUI.
2. The GUI sends the image bytes to the backend with an HTTP `POST` request to `/predict`.
3. The FastAPI backend preprocesses the image (Resize 224×224 → ToTensor → Normalize).
4. The EfficientNet-B0 model runs inference and returns raw logits.
5. The backend applies Softmax to get the predicted class and confidence score.
6. The backend returns a JSON response, and the GUI displays the prediction and confidence to the user.

## Tech Stack

| Component | Technology |
|---|---|
| Model | PyTorch EfficientNet-B0 (fine-tuned) |
| Backend | FastAPI, Uvicorn |
| Containerization | Docker, Docker Compose |
| Frontend | CustomTkinter, Requests, Pillow |

## Project Structure

```
EcoSort_Deployment/
├── docker-compose.yml
├── models/
│   └── ecosort_efficientnet_b0.pth   # Pre-trained PyTorch weights
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt              # API dependencies (FastAPI, torch, numpy<2.0.0)
│   └── main.py                       # FastAPI server
└── frontend/
    └── gui.py                        # CustomTkinter interface
```

## 1. Environment Setup

### Windows

1. Install [Python](https://www.python.org/downloads/) (3.10 or newer).
2. Install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/).
3. **Important:** Docker on Windows requires the Windows Subsystem for Linux (WSL). If Docker fails to start, open **PowerShell as Administrator** and enable the Virtual Machine Platform:

   ```powershell
   Enable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform
   ```

4. Restart your computer, then launch Docker Desktop and wait until it shows **"Engine running"**.

### Linux

1. Install Python 3 for your distribution (for example, on Debian/Ubuntu):

   ```bash
   sudo apt install python3 python3-pip
   ```

2. Install [Docker Engine](https://docs.docker.com/engine/install/) and the [Docker Compose plugin](https://docs.docker.com/compose/install/) for your distribution.
3. Make sure the Docker daemon is running:

   ```bash
   sudo systemctl start docker
   ```

## 2. Running the System

> **Note:** You must start the backend server **before** launching the user interface.

### Step 1: Start the Backend API

Open a terminal in the root `EcoSort_Deployment` directory and build and start the Docker container:

```bash
docker-compose up --build
```

> If your Docker installation uses the newer Compose plugin syntax, run `docker compose up --build` instead (no hyphen).

Wait until the terminal shows that Uvicorn is running on <http://0.0.0.0:8000>.

After the image has been built once, you can leave out the `--build` flag on future launches:

```bash
docker-compose up
```

### Step 2: Start the Frontend GUI

Open a **second terminal**, go to the `frontend` directory, and install the local UI dependencies:

```bash
cd frontend
pip install customtkinter requests pillow
```

Then launch the application:

```bash
python gui.py
```

In the GUI, click **"Select Waste Image"** and choose a `.jpg` or `.png` file to receive a prediction (category and confidence score).

## Stopping the Application

- Close the GUI window.
- In the terminal running Docker, press `Ctrl + C`, then remove the containers:

  ```bash
  docker-compose down
  ```

## Troubleshooting

### `ModuleNotFoundError: No module named 'customtkinter'`

The frontend needs local packages that do not ship with standard Python. Run the following in your active Python environment before running `gui.py`:

```bash
pip install customtkinter requests pillow
```

### `failed to connect to the docker API at npipe:////./pipe/docker_engine` (Windows)

The Docker engine is not running. Open **Docker Desktop** and wait for it to fully initialize. If it shows a Virtual Machine or WSL error, follow the [Windows setup steps](#windows) above to enable the required OS features.

### `RuntimeError: Numpy is not available` (Backend API error)

The `torchvision` library inside the Docker container needs NumPy to process images, and newer PyTorch versions can conflict with NumPy 2.0+. Make sure `backend/requirements.txt` pins:

```
numpy<2.0.0
```

If you change the requirements file, stop the container completely and rebuild it:

```bash
docker-compose down
docker-compose up --build
```
