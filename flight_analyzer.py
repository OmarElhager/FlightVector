import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from mpl_toolkits.mplot3d import Axes3D

# --------------------------
# 1. Generate synthetic flight data
# --------------------------
np.random.seed(42)
time = np.arange(0, 61, 1)  # minutes of flight

altitude = np.piecewise(time,
                        [time < 10, (time >= 10) & (time < 40), time >= 40],
                        [lambda t: 300*t,              # climb
                         lambda t: 3000 + 50*(t-10),  # cruise slight climb
                         lambda t: 4500 - 150*(t-40)])# descent
altitude = altitude.astype(float)  # ensure float for adding noise
altitude += np.random.normal(0, 50, size=len(time))  # small noise

speed = np.piecewise(time,
                     [time < 10, (time >= 10) & (time < 40), time >= 40],
                     [lambda t: 200 + 25*t,        # accelerating climb
                      lambda t: 450 + np.sin(t/3)*10,  # cruise
                      lambda t: 400 - 5*(t-40)])  # descent
speed = speed.astype(float)  # ensure float
speed += np.random.normal(0, 5, size=len(time))

climb_rate = np.gradient(altitude)
fuel_used = np.cumsum(0.5 + 0.01*speed)  # fuel burn
heading = (np.cumsum(np.random.normal(0.5, 0.2, len(time))) % 360)  # degrees

# Inject anomalies (simulate turbulence or system faults)
altitude[25] += 1500  # sudden altitude spike
speed[45] -= 200      # sudden speed drop

# Create DataFrame
df = pd.DataFrame({
    "time_min": time,
    "altitude_m": altitude,
    "speed_kmh": speed,
    "climb_rate_mpm": climb_rate,
    "fuel_used_kg": fuel_used,
    "heading_deg": heading
})

# --------------------------
# 2. Compute flight statistics
# --------------------------
print("\n--- FLIGHT SUMMARY ---")
print("Duration:", df['time_min'].iloc[-1], "minutes")
print("Max altitude:", round(df['altitude_m'].max(), 1), "m")
print("Max speed:", round(df['speed_kmh'].max(), 1), "km/h")
print("Average climb rate:", round(df['climb_rate_mpm'].mean(), 2), "m/min")
print("Total fuel used:", round(df['fuel_used_kg'].iloc[-1], 2), "kg")

# --------------------------
# 3. 2D Visualization
# --------------------------
plt.figure(figsize=(9,5))
plt.plot(df["time_min"], df["altitude_m"], label="Altitude (m)", color="tab:blue")
plt.plot(df["time_min"], df["speed_kmh"], label="Speed (km/h)", color="tab:orange")
plt.xlabel("Time (min)")
plt.ylabel("Value")
plt.title("Flight Profile: Altitude & Speed vs Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------
# 4. 3D Trajectory Visualization
# --------------------------
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection="3d")
ax.plot(df["heading_deg"], df["speed_kmh"], df["altitude_m"], color="green")
ax.set_xlabel("Heading (°)")
ax.set_ylabel("Speed (km/h)")
ax.set_zlabel("Altitude (m)")
ax.set_title("3D Flight Path Visualization")
plt.tight_layout()
plt.show()

# --------------------------
# 5. Anomaly Detection using Isolation Forest
# --------------------------
model = IsolationForest(contamination=0.08, random_state=42)
features = df[["altitude_m", "speed_kmh", "climb_rate_mpm"]]
df["anomaly"] = model.fit_predict(features)

# Identify anomalies
anomalies = df[df["anomaly"] == -1]
print("\n--- ANOMALIES DETECTED ---")
if len(anomalies) == 0:
    print("No anomalies found.")
else:
    print(anomalies[["time_min", "altitude_m", "speed_kmh", "climb_rate_mpm"]].to_string(index=False))

# --------------------------
# 6. Visualize anomalies on 2D chart
# --------------------------
plt.figure(figsize=(9,5))
plt.plot(df["time_min"], df["altitude_m"], label="Normal", color="blue")
plt.scatter(anomalies["time_min"], anomalies["altitude_m"], color="red", label="Anomaly", zorder=5)
plt.xlabel("Time (min)")
plt.ylabel("Altitude (m)")
plt.title("Altitude Anomalies During Flight")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------
# 7. Save results
# --------------------------
df.to_csv("flight_results.csv", index=False)
print("\nProcessed data saved to flight_results.csv")
