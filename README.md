
# Intrusion Detection System (IDS) 

## Objective

This project aims to build a real-time Intrusion Detection System (IDS) that monitors network traffic, detects anomalies using machine learning, and presents the results through an interactive and visually intuitive web dashboard. It is designed to help security analysts and developers understand traffic behavior and identify potential threats in a modular, scalable, and user-friendly environment.

---

## Why This Project Exists

Traditional IDS tools often lack visual clarity and are difficult to customize or extend. This project bridges that gap by combining machine learning with modern web technologies to create a lightweight, interactive dashboard that is both educational and practical. It serves as a foundation for experimenting with anomaly detection models, simulating traffic patterns, and building security analytics tools with real-time feedback.

---

## Technologies Used

- Python: Core backend logic and data processing
- Flask: Lightweight web framework for serving the dashboard
- Pandas & NumPy: Data manipulation and simulation
- Scikit-learn: Machine learning model (One-Class SVM)
- HTML/CSS: Frontend structure and styling
- Chart.js: Interactive bar and line charts for visualizing predictions
- JavaScript: Dynamic content rendering and API integration
- Git: Version control and collaboration

---

## Dataset Used

The dataset simulates outbound TCP traffic from a local machine to Google.com (`142.250.190.78`). It includes both normal and anomalous packets, with anomalies injected by spoofing source IPs and increasing packet sizes.

### Features

| Column      | Description                                 |
|-------------|---------------------------------------------|
| timestamp   | Time when the packet was simulated          |
| src_ip      | Source IP address (normal or spoofed)       |
| dst_ip      | Destination IP (Google)                     |
| protocol    | Protocol used (TCP)                         |
| length      | Packet size in bytes                        |
| src_port    | Source port number                          |
| dst_port    | Destination port (443)                      |
| anomaly     | Label: 0 = normal, 1 = anomaly              |

The dataset is stored in `data/simulated_google_traffic.csv` and is used for both model training and dashboard visualization.

---

## Dashboard Overview

The IDS dashboard provides the following capabilities:

- Prediction Table: Displays the latest traffic records with color-coded anomaly labels.
- Bar Chart: Shows the count of normal vs. anomalous packets.
- Line Chart: Visualizes packet length over time, highlighting anomalies.
- Dataset Preview: Displays a sample of the raw dataset used for inference.
- Responsive UI: Styled with glassmorphism and designed for clarity and accessibility.

All components are dynamically updated via JavaScript and Flask API routes.

---

## Setup Instructions

1. Clone the Repository
   ```bash
   git clone https://github.com/your-username/Intrusion-Detection-System.git
   cd Intrusion-Detection-System
   ```

2. Create and Activate a Virtual Environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install Dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Application
   ```bash
   python app.py
   ```

5. Open in Browser
   ```
   http://localhost:5000
   ```

---

## Screenshots

> Add screenshots of the dashboard interface, prediction charts, and dataset preview here.

---

## License

This project is licensed under the MIT License.  
See the [MIT](LICENSE) file for details.

---

## Conclusion

This IDS dashboard demonstrates how machine learning and modern web technologies can be combined to create a practical, extensible, and visually engaging security tool. It is ideal for learning, experimentation, and real-time traffic analysis. Future enhancements may include support for multiple models, live packet capture, and advanced visual analytics.
```
