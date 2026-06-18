# come back and include the last 4 data points and either pick the best or average

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ekf_DONE import EKF
from Kalman_Versions.sim_DONE import Rover, Soil


UWB_CSV = "uwb_data.csv"
GRAIN_CSV = "grain_data.csv"
VO_CSV = "vo_data.csv"

dt = 0.1
wheel_omega = 8.0


def load_csv(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    rows.sort(key=lambda r: r["timestamp"])
    return rows


def soil_from_camera(features, prev_soil):
    if features is None:
        return prev_soil

    D10, D50, D90, Cu = features

    kc = 1000 + 2 * D50
    kphi = 800 + 1.5 * Cu
    n = 1.0 + 0.05 * (D90 / (D10 + 1e-6))
    c = 200 + 0.2 * D10
    phi = 30 + 0.05 * Cu
    k = 0.02 + 0.001 * D50

    alpha = 0.2

    prev_soil.kc = (1 - alpha) * prev_soil.kc + alpha * kc
    prev_soil.kphi = (1 - alpha) * prev_soil.kphi + alpha * kphi
    prev_soil.n = (1 - alpha) * prev_soil.n + alpha * n
    prev_soil.c = (1 - alpha) * prev_soil.c + alpha * c
    prev_soil.phi = (1 - alpha) * prev_soil.phi + alpha * phi
    prev_soil.k = (1 - alpha) * prev_soil.k + alpha * k

    return prev_soil


uwb_rows = load_csv(UWB_CSV)
grain_rows = load_csv(GRAIN_CSV)
vo_rows = load_csv(VO_CSV)

t_start = min(
    uwb_rows[0]["timestamp"] if uwb_rows else float("inf"),
    grain_rows[0]["timestamp"] if grain_rows else float("inf"),
    vo_rows[0]["timestamp"] if vo_rows else float("inf"),
)
t_end = max(
    uwb_rows[-1]["timestamp"] if uwb_rows else float("-inf"),
    grain_rows[-1]["timestamp"] if grain_rows else float("-inf"),
    vo_rows[-1]["timestamp"] if vo_rows else float("-inf"),
)

rover = Rover(200, 0.3, 0.2, 4)
soil = Soil(1500, 1500, 1.1, 300, 35, 0.03)
ekf = EKF(dt, rover, soil)

uwb_i = 0
grain_i = 0
vo_i = 0
t = t_start

log = {
    "t": [],
    "px": [],
    "py": [],
    "trace_P": [],
}


def replay_step():
    global t, uwb_i, grain_i, vo_i

    if t > t_end:
        return None

    window_end = t + dt

    latest_uwb = None
    while uwb_i < len(uwb_rows) and uwb_rows[uwb_i]["timestamp"] < window_end:
        latest_uwb = uwb_rows[uwb_i]
        uwb_i += 1

    latest_grain = None
    while grain_i < len(grain_rows) and grain_rows[grain_i]["timestamp"] < window_end:
        latest_grain = grain_rows[grain_i]
        grain_i += 1

    latest_vo = None
    while vo_i < len(vo_rows) and vo_rows[vo_i]["timestamp"] < window_end:
        latest_vo = vo_rows[vo_i]
        vo_i += 1

    if latest_grain is not None:
        features = np.array([
            latest_grain["D10"], latest_grain["D50"],
            latest_grain["D90"], latest_grain["Cu"]
        ])
        ekf.soil = soil_from_camera(features, ekf.soil)

    vo_result = None
    if latest_vo is not None:
        vo_result = {
            "dx": latest_vo["dx"],
            "dy": latest_vo["dy"],
            "dz": latest_vo["dz"],
            "dyaw": latest_vo["dyaw"],
        }

    if latest_uwb is not None:
        z_uwb = np.array([[latest_uwb["x"]], [latest_uwb["y"]]])
        ekf.step(wheel_omega, z_uwb, vo_result)
    else:
        ekf.predict(wheel_omega)
        if vo_result is not None:
            ekf.update_vo(vo_result["dx"], vo_result["dy"])

    log["t"].append(t)
    log["px"].append(ekf.x[0, 0])
    log["py"].append(ekf.x[1, 0])
    log["trace_P"].append(float(np.trace(ekf.P)))

    t += dt
    return True


fig, (ax_traj, ax_unc) = plt.subplots(1, 2, figsize=(13, 6))

ax_traj.set_title("Estimated trajectory")
ax_traj.set_xlabel("x (m)")
ax_traj.set_ylabel("y (m)")
ax_traj.axis("equal")
ax_traj.grid(True)
traj_line, = ax_traj.plot([], [], "-", color="tab:blue", linewidth=1, label="estimate")
current_point, = ax_traj.plot([], [], "o", color="tab:red", markersize=6)
ax_traj.legend()

ax_unc.set_title("trace(P) over time \u2014 EKF uncertainty")
ax_unc.set_xlabel("time (s)")
ax_unc.set_ylabel("trace(P)")
ax_unc.grid(True)
trace_line, = ax_unc.plot([], [], color="tab:purple", linewidth=1.5)

plt.tight_layout()


def update(frame):
    result = replay_step()
    if result is None:
        ani.event_source.stop()

    if not log["t"]:
        return traj_line, current_point, trace_line

    traj_line.set_data(log["px"], log["py"])
    current_point.set_data([log["px"][-1]], [log["py"][-1]])
    ax_traj.relim()
    ax_traj.autoscale_view()

    trace_line.set_data(log["t"], log["trace_P"])
    ax_unc.relim()
    ax_unc.autoscale_view()

    return traj_line, current_point, trace_line


ani = animation.FuncAnimation(fig, update, interval=10, cache_frame_data=False)
plt.show()

print(f"Replay finished, {len(log['t'])} steps processed")
if log["t"]:
    print(f"Final state: px={log['px'][-1]:.3f} py={log['py'][-1]:.3f}")
    print(f"Final trace(P): {log['trace_P'][-1]:.4f}")

with open("ekf_replay_output.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["t", "px", "py", "trace_P"])
    for i in range(len(log["t"])):
        writer.writerow([log["t"][i], log["px"][i], log["py"][i], log["trace_P"][i]])

print("Saved ekf_replay_output.csv")