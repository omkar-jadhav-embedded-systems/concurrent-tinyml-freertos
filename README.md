# Dual-Model TinyML Architecture via FreeRTOS

![C++](https://img.shields.io/badge/C++-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow_Lite-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![FreeRTOS](https://img.shields.io/badge/FreeRTOS-22314E?style=for-the-badge&logo=rtos&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

> **A reference architecture for executing multiple independent Edge AI workloads on a resource-constrained, single-core microcontroller.**

## 🧠 The Engineering Challenge

As Edge AI continues moving from prototypes to production, embedded systems are increasingly required to run multiple AI workloads concurrently. For example, a modern Battery Management System (BMS) might need to monitor State of Health (SoH) while simultaneously performing real-time anomaly detection.

While a traditional **superloop** is simple, it risks missing real-time deadlines as inference pipelines grow in complexity. 

This repository serves as a proof-of-concept for migrating from a sequential superloop to a **thread-safe, RTOS-based architecture**—allowing two independent TensorFlow Lite Micro models to share a single MCU core predictably and safely.

---

## 🏗️ System Architecture

![Two TinyML Models. One MCU Core.](https://github.com/omkar-jadhav-embedded-systems/concurrent-tinyml-freertos/blob/main/2_ML_1_Core.png)

### Key Architectural Features:
* **Concurrent Execution:** Replaced blocking superloop logic with independent, prioritized FreeRTOS tasks.
* **Thread Safety:** Implemented a FreeRTOS Mutex (`xSemaphoreTake` / `xSemaphoreGive`) to synchronize access to the TensorFlow Lite Micro interpreter and shared DSP filter states, eliminating race conditions.
* **Quantization-Aware Training (QAT):** Both neural networks were trained using QAT in Python, exporting as `INT8` quantized payloads to fit strict embedded memory constraints.
* **Deterministic Inference:** DSP noise filtering and inference steps run with zero missed real-time control deadlines.
* **Ultra-Low Memory Footprint:** The combined static memory footprint (models + FreeRTOS overhead) is kept under **16KB**.

---

## 📂 Repository Structure

The project in environments: Firmware (C++).

```text

├── anomaly_model.h  
├── sketch.ino 
├── soc_soh.h  
├── LICENSE
└── README.md
