# #!/usr/bin/env python
# kinematics of robot, in the end only forward_all is used to get the transformation matrix of the robot arm.
# the other functions show progress over time.

import math

ZERO_THRESH = 0.00000001
PI = math.pi

# UR robot DH parameters
a2 = -0.4784
a3 = -0.36
d1 = 0.1807
d4 = 0.17415
d5 = 0.11985
d6 = 0.11655
alpha1 = math.pi/2
alpha4 = math.pi/2
alpha5 = -math.pi/2

def SIGN(x):
    return (x > 0) - (x < 0)

def forward(q, T):
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

    T[0] = c1*c2*c3*c4*c5*c6
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

    if T1 is not None:
        T1[0] = c1; T1[1] = 0; T1[2] = s1; T1[3] = 0
        T1[4] = s1; T1[5] = 0; T1[6] = -c1; T1[7] = 0
        T1[8] = 0; T1[9] = 1; T1[10] = 0; T1[11] = d1
        T1[12] = 0; T1[13] = 0; T1[14] = 0; T1[15] = 1

    if T2 is not None:
        T2[0] = c2; T2[1] = -s2; T2[2] = 0; T2[3] = a2 * c2
        T2[4] = s2; T2[5] = c2; T2[6] = 0; T2[7] = a2 * s2
        T2[8] = 0; T2[9] = 0; T2[10] = 1; T2[11] = 0
        T2[12] = 0; T2[13] = 0; T2[14] = 0; T2[15] = 1

    if T3 is not None:
        T3[0] = c3; T3[1] = -s3; T3[2] = 0; T3[3] = a3 * c3
        T3[4] = s3; T3[5] = c3; T3[6] = 0; T3[7] = a3 * s3
        T3[8] = 0; T3[9] = 0; T3[10] = 1; T3[11] = 0
        T3[12] = 0; T3[13] = 0; T3[14] = 0; T3[15] = 1

    if T4 is not None:
        T4[0] = c4; T4[1] = 0; T4[2] = s4 * math.sin(alpha4); T4[3] = 0
        T4[4] = s4; T4[5] = 0; T4[6] = -c4 * math.sin(alpha4); T4[7] = 0
        T4[8] = 0; T4[9] = 1; T4[10] = 0; T4[11] = d4
        T4[12] = 0; T4[13] = 0; T4[14] = 0; T4[15] = 1

    if T5 is not None:
        T5[0] = c5; T5[1] = 0; T5[2] = s5 * math.sin(alpha5)
        T5[3] = 0
        T5[4] = s5; T5[5] = 0; T5[6] = -c5 * math.sin(alpha5)
        T5[7] = 0
        T5[8] = 0; T5[9] = -1; T5[10] = 0; T5[11] = d5
        T5[12] = 0; T5[13] = 0; T5[14] = 0; T5[15] = 1

    if T6 is not None:
        T6[0] = c6
        T6[1] = -s6 
        T6[2] = 0
        T6[3] = 0
        T6[4] = s6
        T6[5] = c6
        T6[6] = 0
        T6[7] = 0
        T6[8] = 0
        T6[9] = 0
        T6[10] = 1
        T6[11] = d6
        T6[12] = 0
        T6[13] = 0
        T6[14] = 0
        T6[15] = 1

def inverse(T, q_sols, q6_des):
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