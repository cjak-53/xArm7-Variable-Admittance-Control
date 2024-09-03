import numpy as np

class AdmittanceController:
    def __init__(self, M, D_initial, dt):
        self.M = M  # Virtual inertia matrix (diagonal)
        self.D = D_initial  # Initial virtual damping matrix
        self.dt = dt  # Time step
        self.x_dot = np.zeros(2)  # Initial Cartesian velocity of the end-effector

    def update_damping(self, D_new):
        """ Update the virtual damping matrix based on ILC. """
        self.D = D_new

    def compute_velocity(self, f):
        """
        Compute the reference Cartesian velocity based on the external force.
        :param f: External force applied by the human operator.
        :return: Updated Cartesian velocity.
        """
        # Compute the acceleration (ẍ) from the admittance control equation
        x_ddot = np.linalg.inv(self.M) @ (f - self.D @ self.x_dot)
        
        # Update velocity using the computed acceleration
        self.x_dot += x_ddot * self.dt
        
        return self.x_dot

# Initialize parameters
M = np.diag([0.875, 0.875])  # Virtual inertia matrix
D_initial = np.diag([1.0, 1.0])  # Initial virtual damping matrix (can be modified)
dt = 0.001  # Time step (1 ms)

# Create the admittance controller
controller = AdmittanceController(M, D_initial, dt)

# Example force applied by the human operator (detected by the torque sensor)
f = np.array([1.0, 0.5])  # Force vector (example values)

# Update the damping matrix using ILC (example new damping values)
D_new = np.diag([1.2, 1.1])
controller.update_damping(D_new)

# Compute the reference Cartesian velocity of the end-effector
x_dot = controller.compute_velocity(f)

print(f"Reference Cartesian Velocity: {x_dot}")
