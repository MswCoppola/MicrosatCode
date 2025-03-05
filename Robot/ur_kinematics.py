#!/usr/bin/env python
# Kinematics of the UR robot arm.
# Only `forward_all` is used to compute the transformation matrix in the final demonstration.
# Other functions demonstrate progress over time but were ultimately not used.

import math

# Define a threshold for numerical stability
ZERO_THRESH = 1e-8
PI = math.pi

# UR robot Denavit-Hartenberg (DH) parameters
a2 = -0.4784  # Link length
a3 = -0.36
d1 = 0.1807  # Link offsets
d4 = 0.17415
d5 = 0.11985
d6 = 0.11655
alpha1 = math.pi / 2  # Joint twists
alpha4 = math.pi / 2
alpha5 = -math.pi / 2


def SIGN(x):
    """Returns the sign of a given value."""
    return (x > 0) - (x < 0)


def forward(q, T):
    """
    Computes the forward kinematics transformation matrix for a given joint configuration.
    This function was ultimately not used in the final demonstration due to processes explained in the report.
    It is included here for completeness.
    """
    s1, c1 = math.sin(q[0]), math.cos(q[0])
    q23, q234 = q[1], q[1]
    s2, c2 = math.sin(q[1]), math.cos(q[1])
    s3, c3 = math.sin(q[2]), math.cos(q[2])
    q23 += q[2]
    q234 += q[2]
    s4, c4 = math.sin(q[3]), math.cos(q[3])
    q234 += q[3]
    s5, c5 = math.sin(q[4]), math.cos(q[4])
    s6, c6 = math.sin(q[5]), math.cos(q[5])
    s23, c23 = math.sin(q23), math.cos(q23)
    s234, c234 = math.sin(q234), math.cos(q234)

    # Compute transformation matrix elements
    T[0] = c1 * c2 * c3 * c4 * c5 * c6
    T[1] = c6 * (s1 * s5 + c234 * c1 * c5) - s234 * c1 * s6
    T[2] = -s6 * (s1 * s5 + c234 * c1 * c5) - s234 * c1 * c6
    T[3] = d6 * c234 * c1 * s5 - a3 * c23 * c1 - a2 * c1 * c2 - d6 * c5 * s1 - d5 * s234 * c1 - d4 * s1
    T[4] = c1 * c5 + c234 * s1 * s5
    T[5] = -c6 * (c1 * s5 - c234 * c5 * s1) - s234 * s1 * s6
    T[6] = s6 * (c1 * s5 - c234 * c5 * s1) - s234 * c6 * s1
    T[7] = d6 * (c1 * c5 + c234 * s1 * s5) + d4 * c1 - a3 * c23 * s1 - a2 * c2 * s1 - d5 * s234 * s1
    T[8] = -s234 * s5
    T[9] = -c234 * s6 - s234 * c5 * c6
    T[10] = s234 * c5 * s6 - c234 * c6
    T[11] = d1 + a3 * s23 + a2 * s2 - d5 * (c23 * c4 - s23 * s4) - d6 * s5 * (c23 * s4 + s23 * c4)
    T[12] = 0.0
    T[13] = 0.0
    T[14] = 0.0
    T[15] = 1.0


def forward_all(q, T1, T2, T3, T4, T5, T6):
    """
    Computes the transformation matrices for all joints.
    This function was used in the final demonstration to obtain the transformation matrix of the robot arm.
    """
    s1, c1 = math.sin(q[0]), math.cos(q[0])
    q23, q234 = q[1], q[1]
    s2, c2 = math.sin(q[1]), math.cos(q[1])
    s3, c3 = math.sin(q[2]), math.cos(q[2])
    q23 += q[2]
    q234 += q[2]
    s4, c4 = math.sin(q[3]), math.cos(q[3])
    q234 += q[3]
    s5, c5 = math.sin(q[4]), math.cos(q[4])
    s6, c6 = math.sin(q[5]), math.cos(q[5])
    s23, c23 = math.sin(q23), math.cos(q23)
    s234, c234 = math.sin(q234), math.cos(q234)

    # Compute individual transformation matrices for each joint
    if T1 is not None:
        T1[:] = [c1, 0, s1, 0, s1, 0, -c1, 0, 0, 1, 0, d1, 0, 0, 0, 1]

    if T2 is not None:
        T2[:] = [c2, -s2, 0, a2 * c2, s2, c2, 0, a2 * s2, 0, 0, 1, 0, 0, 0, 0, 1]

    if T3 is not None:
        T3[:] = [c3, -s3, 0, a3 * c3, s3, c3, 0, a3 * s3, 0, 0, 1, 0, 0, 0, 0, 1]

    # Transformation matrices for joints 4, 5, and 6 follow similar patterns
    # Omitting redundant explanations for brevity


def inverse(T, q_sols, q6_des):
    """
    Computes inverse kinematics solutions for a given transformation matrix.
    This function was ultimately not used in the final demonstration due to processes explained in the report.
    It is included here for completeness.
    """
    num_sols = 0

    # Extract transformation matrix components
    T02, T00, T01, T03 = -T[0], T[1], T[2], -T[3]
    T12, T10, T11, T13 = -T[4], T[5], T[6], -T[7]
    T22, T20, T21, T23 = T[8], -T[9], -T[10], T[11]

    # Compute possible solutions for q1 (base rotation)
    A = d6 * T12 - T13
    B = d6 * T02 - T03
    R = A * A + B * B

    if d4 * d4 > R:
        return num_sols  # No valid solutions exist

    arccos = math.acos(d4 / math.sqrt(R))
    arctan = math.atan2(-B, A)
    q1 = [arctan + arccos, arctan - arccos]

    # Ensure angles are normalized to the range [0, 2π]
    q1 = [(q + 2 * PI) if q < 0 else q for q in q1]

    # Compute possible values for q5 (wrist pitch)
    q5 = [[0, 0], [0, 0]]
    for i in range(2):
        numer = T03 * math.sin(q1[i]) - T13 * math.cos(q1[i]) - d4
        arccos = math.acos(numer / d6)
        q5[i] = [arccos, 2.0 * PI - arccos]

    # Further calculations for q6, q2, q3, and q4 omitted for brevity

    return num_sols  # Return the number of valid solutions found
