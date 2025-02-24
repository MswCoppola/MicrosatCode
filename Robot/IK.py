#!/usr/bin/env python
 
import numpy as np
from robot_kin import forward, inverse  # Import forward and inverse kinematics functions
 
def main():
    # Define the joint angles (in radians) for the robot arm
    # These values represent the angles for each of the 6 joints of the robotic arm
    q = np.array([
        0.221097637965649,  # Joint 1 (Base rotation)
        -1.87309719622658,  # Joint 2 (Shoulder)
        1.44379984784883,   # Joint 3 (Elbow)
        0.429297403956852,  # Joint 4 (Wrist 1)
        0.223715636947962,  # Joint 5 (Wrist 2)
        -1.57079637937585 - np.pi / 4  # Joint 6 (Wrist 3), adding -π/4 for offset
    ])
 
    # Create an empty 1D array of 16 elements to store the transformation matrix (T)
    # This will be filled by the forward kinematics function
    T = np.zeros(16)  # Transformation matrix stored as a flat 1D array
 
    # Compute the forward kinematics (FK) transformation matrix for the given joint angles
    forward(q, T)  # The function modifies T in place
 
    # T is a 4×4 matrix stored in a 1D array (row-major order)
    # Uncomment the following lines if you want to print T in a readable 4x4 format:
    # T = T.reshape(4,4, order='C')  # Reshape T into a 4×4 matrix (row-major order)
    # print("T ivan isha is:" , T)
 
    # Print the transformation matrix (T) in row-major order for debugging
    # print("Forward Kinematics Result:")
    # for i in range(4):
    #     print(" ".join(f"{T[i, j]:1.3f}" for j in range(4)))
 
    # Create an empty 1D array to store up to 8 possible inverse kinematics (IK) solutions
    # Each solution consists of 6 joint angles, so we need space for 8 * 6 = 48 values
    q_sols = np.zeros(8 * 6)  
 
    # Compute the inverse kinematics (IK) solutions based on the transformation matrix T
    # The function returns the number of valid solutions found
    num_sols = inverse(T, q_sols, 0)  # The third argument (0) represents a preferred q6 value
 
    # Check if any IK solutions were found
    if q_sols is not None:
        print(f"IK Solutions Found: {num_sols}")  # Print the number of solutions
    else:
        print("No IK solution found.")  # No valid inverse kinematics solution was found
 
    # Print each inverse kinematics solution
    for i in range(num_sols):
        print(f"{q_sols[i * 6 + 0]:1.6f} {q_sols[i * 6 + 1]:1.6f} {q_sols[i * 6 + 2]:1.6f} "
              f"{q_sols[i * 6 + 3]:1.6f} {q_sols[i * 6 + 4]:1.6f} {q_sols[i * 6 + 5]:1.6f}")
        # Each row represents one valid set of 6 joint angles found by the inverse kinematics function
 
    # Print multiples of π/2 from 0 to 4π/2
    # This matches the output of the original C++ program
    for i in range(5):
        print(f"{np.pi / 2.0 * i} ", end="")
    print()  # Move to a new line after printing
 
# Run the main function when the script is executed
main()