

# FlightVector

**FlightVector** is a Python project for generating, analyzing, and visualizing synthetic flight data. It includes 3D trajectory plotting, flight statistics computation, and anomaly detection to simulate and study aircraft behavior.

## Features

* Generates synthetic flight profiles with altitude, speed, and heading data.
* Computes key flight statistics such as maximum altitude, speed, climb rate, and fuel consumption.
* Visualizes flight paths in 2D and 3D for intuitive understanding.
* Detects anomalies in flight data using machine learning techniques (Isolation Forest).
* Interactive plots compatible with VS Code and Windows PowerShell.

## Installation

```bash
git clone https://github.com/YourUsername/FlightVector.git
cd FlightVector
pip install -r requirements.txt
```

*(Requirements: `numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-learn`)*

## Usage

Run the main Python script:

```bash
python flight_analyzer.py
```

This will generate synthetic flight data, display 2D/3D visualizations, and print flight statistics and detected anomalies.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

