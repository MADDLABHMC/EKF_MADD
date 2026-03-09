from ekf import EKF
from sim import elliptical_path, Rover, Soil, UWB_sensor
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    
    # STUFF TO MESS WITH
    dt = 0.1
    time = 625 # seconds sim runs
    x_pos = 1.0
    y_pos = 1.0
    x_velocity = 1.0  # slow starting velocity
    y_velocity = 0.5  # no initial y velocity
    x_accel = 0.1  # gave it baby accel for now
    y_accel = 0.15  # no y accel...for now

    ekf = EKF(dt)
    
    # rover and soil creation // practice parameters
    rover = Rover(mass=200, wheel_radius=0.3, wheel_width=0.2, num_wheels=4)
    soil = Soil(kc=1500, kphi=1500, n=1.1, c=300, phi=35, k=0.03)   
    
     
    # Create sensors here:
    s1 = UWB_sensor("Anchor_1", uncertainty=0.04, anchor_pos=(0, 0))
    s2 = UWB_sensor("Anchor_2", uncertainty=0.06, anchor_pos=(10, 0))
    s3 = UWB_sensor("Anchor_3", uncertainty=0.02, anchor_pos=(0, 10))
    s4 = UWB_sensor("Anchor_4", uncertainty=0.05, anchor_pos=(10, 10))

    # add sensors to EKF **IMPORTANT**
    ekf.addSensor(s1)
    ekf.addSensor(s2)
    ekf.addSensor(s3)
    ekf.addSensor(s4)
    #------------------------------------------------------------------------------------
    
    # Ground-truth state for testing: [px, py, vx, vy, ax, ay]
    ts = [x_pos, y_pos,x_velocity, y_velocity, x_accel, y_accel]
    true_state = np.array(ts)[:, np.newaxis]

    # for plotting later 
    true_x_history = []
    true_y_history = []
    pred_x_history = []
    pred_y_history = []
    uncertainty_history = []
    time_steps = []

    # Run simulation
    for step in range(time):
        # fake real motion without noise ***for testing only***
        t = step * dt

        ax, ay = brekkerWong_accel_model(rover, soil, ekf.x.flatten())
        
        u = np.array([ax, ay])
        
        # version for ellipse, line, circle etc
        true_px, true_py = elliptical_path(t, [3,3], 3, 2, .1)

        ekf.step(true_px, true_py, u)
        
        # True position
        true_x, true_y = true_px, true_py
        true_x_history.append(true_x)
        true_y_history.append(true_y)

        # Predicted position
        pred_x, pred_y = ekf.x[0,0], ekf.x[1,0]
        pred_x_history.append(pred_x)
        pred_y_history.append(pred_y)
        
        sigma_x = np.sqrt(ekf.P[0,0])
        sigma_y = np.sqrt(ekf.P[1,1])

        uncertainty_history.append((sigma_x, sigma_y))
        time_steps.append(t)

    # PLOTTING TRAJECTORY
    plt.figure(figsize=(8,8))

    # True trajectory
    plt.plot(true_x_history, true_y_history, label="True Position", linewidth=2)

    # EKF predicted trajectory
    plt.plot(pred_x_history, pred_y_history, label="EKF Predicted", linestyle="--")

    # Anchors
    for sensor in ekf.sensors:
        plt.scatter(sensor.anchor_pos[0], sensor.anchor_pos[1],
                    marker='X', s=100, label=sensor.name)

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Bot Trajectory: True vs EKF Predicted with Anchors & Uncertainty")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    
    # PLOTTING UNCERTAINTY OVER TIME
    plt.figure(figsize=(8,5))
    
    sigma_x_vals = [s[0] for s in uncertainty_history]
    sigma_y_vals = [s[1] for s in uncertainty_history]
    
    plt.plot(time_steps, sigma_x_vals, label="σ_x (X Uncertainty)")
    plt.plot(time_steps, sigma_y_vals, label="σ_y (Y Uncertainty)")

    plt.xlabel("Time (s)")
    plt.ylabel("Standard Deviation (m)")
    plt.title("EKF Position Uncertainty Over Time")

    plt.legend()
    plt.grid(True)

    plt.show()