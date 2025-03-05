#!/usr/bin/env python
# Kinematics of the UR robot arm.
# In the final demonstration, only `forward_all` was used to compute the transformation matrix.
# Other functions demonstrate progress over time but were not used in the final implementation.

import math

# Define a small threshold for numerical stability
ZERO_THRESH = 0.00000001
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
    """Returns the sign of a given value (+1 for positive, -1 for negative, 0 for zero)."""
    return (x > 0) - (x < 0)

def forward(q, T):
    """
    Computes the forward kinematics transformation matrix for a given joint configuration.
    This function was ultimately not used in the final demonstration due to processes explained in the report.
    It is included here for completeness.
    """
    # Precompute sines and cosines of joint angles
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

    # Compute elements of the transformation matrix
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
    # Precompute sines and cosines for efficiency
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

    # Compute transformation matrices for each joint
    if T1 is not None:
        T1[:] = [c1, 0, s1, 0,  s1, 0, -c1, 0,  0, 1, 0, d1,  0, 0, 0, 1]

    if T2 is not None:
        T2[:] = [c2, -s2, 0, a2 * c2,  s2, c2, 0, a2 * s2,  0, 0, 1, 0,  0, 0, 0, 1]

    if T3 is not None:
        T3[:] = [c3, -s3, 0, a3 * c3,  s3, c3, 0, a3 * s3,  0, 0, 1, 0,  0, 0, 0, 1]

    if T4 is not None:
        T4[:] = [c4, 0, s4, 0, s4, 0, -c4, 0, 0, 1, 0, d4,  0, 0, 0, 1]

    if T5 is not None:
        T5[:] = [c5, 0, s5, 0,  s5, 0, -c5, 0,  0, -1, 0, d5,  0, 0, 0, 1]

    if T6 is not None:
        T6[:] = [c6, -s6, 0, 0,  s6, c6, 0, 0,  0, 0, 1, d6,  0, 0, 0, 1]

def inverse(T, q_sols, q6_des):
    """
    Computes inverse kinematics solutions for a given transformation matrix.
    This function was ultimately not used in the final demonstration due to processes explained in the report.
    It is included here for completeness.
    """
    num_sols = 0

    # Extract elements from the transformation matrix
    T02, T00, T01, T03 = -T[0], T[1], T[2], -T[3]
    T12, T10, T11, T13 = -T[4], T[5], T[6], -T[7]
    T22, T20, T21, T23 = T[8], -T[9], -T[10], T[11]

    # Solve for q1 using tangent half-angle substitution
    A = d6 * T12 - T13
    B = d6 * T02 - T03
    R = A * A + B * B

    if d4 * d4 > R:
        return num_sols  # No valid solutions

    arccos = math.acos(d4 / math.sqrt(R))
    arctan = math.atan2(-B, A)
    q1 = [arctan + arccos, arctan - arccos]
    q1 = [(q + 2 * PI) if q < 0 else q for q in q1]  # Normalize

    # Solve for q5
    q5 = [[0, 0], [0, 0]]
    for i in range(2):
        numer = T03 * math.sin(q1[i]) - T13 * math.cos(q1[i]) - d4
        arccos = math.acos(numer / d6)
        q5[i] = [arccos, 2.0 * PI - arccos]

    # Solve for q6, q2, q3, q4
    for i in range(2):
        for j in range(2):
            c1, s1 = math.cos(q1[i]), math.sin(q1[i])
            c5, s5 = math.cos(q5[i][j]), math.sin(q5[i][j])

            # Solve for q6 using atan2
            if abs(s5) < ZERO_THRESH:
                q6 = q6_des
            else:
                q6 = math.atan2(SIGN(s5) * -(T01 * s1 - T11 * c1), SIGN(s5) * (T00 * s1 - T10 * c1))

            c6, s6 = math.cos(q6), math.sin(q6)
            x04x = -s5 * (T02 * c1 + T12 * s1) - c5 * (s6 * (T01 * c1 + T11 * s1) - c6 * (T00 * c1 + T10 * s1))
            x04y = c5 * (T20 * c6 - T21 * s6) - T22 * s5
            p13x = d5 * (s6 * (T00 * c1 + T10 * s1) + c6 * (T01 * c1 + T11 * s1)) - d6 * (T02 * c1 + T12 * s1) + T03 * c1 + T13 * s1
            p13y = T23 - d1 - d6 * T22 + d5 * (T21 * c6 + T20 * s6)

            # Solve for q2, q3 using geometric method
            c3 = (p13x ** 2 + p13y ** 2 - a2 ** 2 - a3 ** 2) / (2.0 * a2 * a3)
            if abs(c3) > 1.0:
                continue  # No valid solution

            s3 = math.sqrt(1 - c3 ** 2)
            q3 = [math.atan2(s3, c3), math.atan2(-s3, c3)]

            denom = a2 ** 2 + a3 ** 2 + 2 * a2 * a3 * c3
            A = (a2 + a3 * c3)
            B = a3 * s3

            q2 = [
                math.atan2((A * p13y - B * p13x) / denom, (A * p13x + B * p13y) / denom),
                math.atan2((A * p13y + B * p13x) / denom, (A * p13x - B * p13y) / denom)
            ]

            # Solve for q4
            c23 = [math.cos(q2[0] + q3[0]), math.cos(q2[1] + q3[1])]
            s23 = [math.sin(q2[0] + q3[0]), math.sin(q2[1] + q3[1])]

            q4 = [
                math.atan2(c23[0] * x04y - s23[0] * x04x, x04x * c23[0] + x04y * s23[0]),
                math.atan2(c23[1] * x04y - s23[1] * x04x, x04x * c23[1] + x04y * s23[1])
            ]

            # Store solutions
            for k in range(2):
                q_sols[num_sols * 6 + 0] = q1[i]
                q_sols[num_sols * 6 + 1] = q2[k]
                q_sols[num_sols * 6 + 2] = q3[k]
                q_sols[num_sols * 6 + 3] = q4[k]
                q_sols[num_sols * 6 + 4] = q5[i][j]
                q_sols[num_sols * 6 + 5] = q6
                num_sols += 1

    return num_sols