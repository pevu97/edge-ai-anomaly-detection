# 🚀 Edge AI for Onboard Data Prioritization (Anomaly Detection)

This project presents a proof-of-concept Edge AI system designed for **onboard data prioritization and selective transmission** in bandwidth-constrained environments, such as planetary missions.

The system simulates how spacecraft (or other edge devices) can automatically select and transmit only the most valuable data using anomaly detection.

---

## 🧠 Problem

Modern space missions generate massive amounts of visual data, while communication bandwidth remains extremely limited.

As a result:

* most data cannot be transmitted,
* valuable observations may be delayed or lost,
* manual prioritization is not scalable.

---

## 💡 Solution

This project implements an **onboard AI pipeline** that:

1. Processes incoming images
2. Computes anomaly scores using an autoencoder
3. Ranks images by importance
4. Simulates selective transmission under bandwidth constraints

---

## ⚙️ Pipeline Overview

```
run_inference.py
↓
simulate_transmission.py
↓
generate_report.py
```

---

## 📊 Features

* Autoencoder-based anomaly detection
* Image prioritization using reconstruction error
* Simulation of transmission scenarios:

  * 10%
  * 5%
  * 1%
* Performance metrics (inference time, throughput)
* Visual reports and plots

---

## 📁 Project Structure

```
.
├── data/
├── inference results/
    └── inference_records.json
    └── inference_summary.json
├── report/
├── run_inference.py
├── simulate_transmission.py
├── generate_report.py
├── requirements.txt
│
│   
│
├── ae_results/
│   └── best_autoencoder.pth
│
└── simulation/
    ├── results/
    └── analysis/
        └── report/
```

---

## ▶️ Quick Start

### 1. Clone repository

```
git clone https://github.com/pevu97/edge-ai-anomaly-detection.git
cd edge-ai-anomaly-detection
```

---

### 2. Install dependencies

```
pip install -r requirements.txt
```

---

### 3. ⚠️ IMPORTANT – Clean previous results

Before running the pipeline, **you must clear the contents of the folder**:

```
simulation/results/
```

Otherwise:

* results will be duplicated
* reports may become inconsistent

---

### 4. Run full pipeline

```
python run_inference.py
python simulate_transmission.py
python generate_report.py
```

---

## 📈 Output

After execution:

### `simulation/results/`

* inference_records.json
* inference_summary.json
* scenario_10pct.csv
* scenario_5pct.csv
* scenario_1pct.csv
* selected image folders

### `simulation/analysis/report/`

* histograms
* comparison tables
* selected image visualizations
* rejected image examples

---

## 🧪 Demo Dataset

The repository includes a **small sample dataset** for demonstration.

Full experiments were conducted on a significantly larger dataset (~20k images), not included due to size.

---

## ⚡ Performance

Example results:

* ~11 images/sec inference throughput
* ~0.088 s per image
* up to **99% data reduction**

---

## 🛰️ Use Cases

* Planetary missions (Mars rovers, orbiters)
* Autonomous satellites
* Remote sensing systems
* Edge AI systems
* Drone-based exploration

---

## 🚧 Future Work

* Edge deployment (embedded hardware)
* Multi-modal data
* Real-time processing
* Improved anomaly detection

---

## 📬 Contact

Author: **[Your Name]**
GitHub: https://github.com/pevu97

---

## 📄 License

Research / demonstration purposes.
