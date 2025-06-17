import mujoco
import numpy as np
import time
from math import cos, sin, degrees
import argparse
from mujoco import viewer

class RoboconRobot:
    def __init__(self, model_path):
        try:
            self.model = mujoco.MjModel.from_xml_path(model_path)
            self.data = mujoco.MjData(self.model)
            
            print("Model loaded successfully!")
            print("Actuators:", [self.model.actuator(i).name for i in range(self.model.nu)])
            print("Joints:", [self.model.joint(i).name for i in range(self.model.njnt)])
            
            # Robot parameters (tune these!)
            self.WHEEL_RADIUS = 0.05  # meters
            self.WHEEL_BASE = 0.3     # meters
            self.MAX_RPM = 10         # Reduced from 60 to prevent instability
            
            # Odometry state
            self.pose = np.array([0.0, 0.0, 0.0])
            self.last_time = time.time()
            
            # Verify actuators exist
            actuator_names = [self.model.actuator(i).name for i in range(self.model.nu)]
            self.actuators_available = all(
                name in actuator_names 
                for name in ['left_wheel_velocity', 'right_wheel_velocity']
            )
            if not self.actuators_available:
                raise ValueError("Missing actuators in XML!")
                
        except Exception as e:
            print(f"Error during initialization: {e}")
            raise

    def set_wheel_velocities(self, left_rpm, right_rpm):
        """Set wheel speeds in RPM (clipped to MAX_RPM)"""
        if not self.actuators_available:
            return
            
        left_rads = np.clip(left_rpm, -self.MAX_RPM, self.MAX_RPM) * 2 * np.pi / 60
        right_rads = np.clip(right_rpm, -self.MAX_RPM, self.MAX_RPM) * 2 * np.pi / 60
        
        left_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_wheel_velocity")
        right_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_wheel_velocity")
        self.data.ctrl[left_id] = left_rads
        self.data.ctrl[right_id] = right_rads

    def update_odometry(self):
        """Update odometry with fixed timestep (0.01s) to avoid instability"""
        left_vel = self.data.qvel[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_wheel_joint")]
        right_vel = self.data.qvel[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_wheel_joint")]
        
        # Use a FIXED timestep (critical for stability)
        dt = 0.01  
        
        linear_vel = self.WHEEL_RADIUS * (left_vel + right_vel) / 2
        angular_vel = self.WHEEL_RADIUS * (right_vel - left_vel) / self.WHEEL_BASE
        
        self.pose[0] += linear_vel * cos(self.pose[2]) * dt
        self.pose[1] += linear_vel * sin(self.pose[2]) * dt
        self.pose[2] += angular_vel * dt
        
        return self.pose.copy()

    def run(self, duration=10.0):
        """Main loop with fixed timestep and stability checks"""
        try:
            with viewer.launch_passive(self.model, self.data) as v:
                print(f"Simulation running for {duration} seconds...")
                start_time = time.time()
                
                while time.time() - start_time < duration:
                    # Gentle control inputs (reduced RPM)
                    self.set_wheel_velocities(5.0, 4.0)  # Reduced from (30, 20)
                    
                    # Step simulation
                    mujoco.mj_step(self.model, self.data)
                    
                    # Update odometry
                    pose = self.update_odometry()
                    print(f"Position: X={pose[0]:.2f}m, Y={pose[1]:.2f}m, θ={degrees(pose[2]):.1f}°")
                    
                    # Sync viewer
                    v.sync()
                    time.sleep(max(0, 0.01 - (time.time() - start_time)))  # Real-time pacing
                    
        except KeyboardInterrupt:
            print("\nSimulation stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="hello.xml", help="MuJoCo model file")
    parser.add_argument("--time", type=float, default=10.0, help="Simulation duration (seconds)")
    args = parser.parse_args()
    
    robot = RoboconRobot(args.model)
    robot.run(args.time)