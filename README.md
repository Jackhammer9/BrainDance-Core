<img src="static/Braindance Logo.png">

# Project BrainDance

A hardware-in-the-loop bio-robotic interface designed to ingest real-time neural signals, apply edge-optimized digital signal processing, and isolate motor imagery intents for multi-axis robotic manipulator control.

This repository serves as the public open-source storefront for the standalone core architecture, data ingestion frameworks, and machine learning pipelines developed independently by Arnav Bajaj.

## Funding and Academic Vetting
This independent research framework was selected and is actively backed by a 7500 EUR prototyping grant via the THWS KickStart Initiative, funded by the German Federal Ministry of Education and Research (BMBF) and the DAAD.

## System Architecture and Technical Capabilities
The pipeline is modularly engineered from raw signal acquisition to model inference:
* **Real-Time Data Ingestion:** Formulated a multi-threaded streaming data capture architecture utilizing the BrainFlow library to interface directly with an OpenBCI Ultracortex Mark IV 8-channel hardware stack.
* **Digital Signal Processing:** Implemented standalone preprocessing modules for real-time artifact removal and signal enhancement, including 50Hz/60Hz notch filtering, Butterworth bandpass configurations, Fast Fourier Transforms, Power Spectral Density, and spatial Common Spatial Patterns or Laplacian filtering.
* **Deep Learning Layer:** Architected a hybrid CNN-LSTM-MLP network configured to extract dynamic spatial-temporal features from continuous multi-channel EEG data arrays.

## Active Research Bottlenecks and Current Milestones
The engineering infrastructure functions reliably, but the neural classification layer is currently experiencing standard real-world deployment challenges:
* **Signal-to-Noise Ratio Boundaries:** Actively optimizing hyper-parameters to isolate structural ocular or muscular artifacts and increase the validation accuracy of non-invasive motor imagery intents.
* **Trajectory Mapping:** Refining the translation of low-latency model inference loops into smooth Cartesian-space trajectory arrays for physical manipulators.
