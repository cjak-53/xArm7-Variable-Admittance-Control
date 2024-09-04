import numpy as np

class DiscreteAdmittanceController:
    def __init__(self, M, D, dt):
        self.M_inv = np.linalg.inv(M)  # Inverse of the virtual inertia matrix
        self.D = D  # Virtual damping matrix
        self.dt = dt  # Time step
        self.I = np.eye(M.shape[0])  # Identity matrix of the same dimension as M
        self.x_dot = np.zeros(M.shape[0])  # Initial velocity of the end-effector

    def update(self, f):
        """
        Update the velocity of the end-effector based on external force and previous velocity.
        :param f: External force applied at current time step.
        :return: New velocity of the end-effector.
        """
        # Calculate the first part of the equation: Delta t * M^-1 * f
        part1 = self.dt * self.M_inv @ f
        
        # Calculate the second part of the equation: (I - Delta t * M^-1 * D) * x_dot(t-1)
        part2 = (self.I - self.dt * self.M_inv @ self.D) @ self.x_dot
        
        # Update velocity
        self.x_dot = part1 + part2
        
        return self.x_dot

# Parameters
M = np.diag([0.875, 0.875])  # Virtual inertia matrix
D = np.diag([50, 50])  # Virtual damping matrix
dt = 0.001  # Time step

# Initialize controller
controller = DiscreteAdmittanceController(M, D, dt)

# Time settings
T_start = 0  # Start time
T_end = 5  # End time
t_range = np.arange(T_start, T_end + dt, dt)  # Time range

# Run simulation over the time range
velocities = []
for t in t_range:
    # Compute the external force based on the given function
    f_value = -5/3 * (t - 5/2)**2
    f = np.array([f_value, f_value])  # Applying the same force in both directions
    
    # Update controller and get new velocity
    new_velocity = controller.update(f)
    velocities.append(new_velocity)

# Display the final updated velocities
print(f"Final Updated Velocity: {velocities[-1]}")
