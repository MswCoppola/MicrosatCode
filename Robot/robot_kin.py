import math

ZERO_THRESH = 0.00000001
PI = math.pi

# UR robot DH parameters
d1 = 0.187
a2 = -0.4784
a3 = -0.36
d4 = 0.17415
d5 = 0.11985
d6 = 0.11655

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

    T[0] = c234 * c1 * s5 - c5 * s1
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
        T2[0] = c1 * c2; T2[1] = -c1 * s2; T2[2] = s1; T2[3] = a2 * c1 * c2
        T2[4] = c2 * s1; T2[5] = -s1 * s2; T2[6] = -c1; T2[7] = a2 * c2 * s1
        T2[8] = s2; T2[9] = c2; T2[10] = 0; T2[11] = d1 + a2 * s2
        T2[12] = 0; T2[13] = 0; T2[14] = 0; T2[15] = 1

    if T3 is not None:
        T3[0] = c23 * c1; T3[1] = -s23 * c1; T3[2] = s1; T3[3] = c1 * (a3 * c23 + a2 * c2)
        T3[4] = c23 * s1; T3[5] = -s23 * s1; T3[6] = -c1; T3[7] = s1 * (a3 * c23 + a2 * c2)
        T3[8] = s23; T3[9] = c23; T3[10] = 0; T3[11] = d1 + a3 * s23 + a2 * s2
        T3[12] = 0; T3[13] = 0; T3[14] = 0; T3[15] = 1

    if T4 is not None:
        T4[0] = c234 * c1; T4[1] = s1; T4[2] = s234 * c1; T4[3] = c1 * (a3 * c23 + a2 * c2) + d4 * s1
        T4[4] = c234 * s1; T4[5] = -c1; T4[6] = s234 * s1; T4[7] = s1 * (a3 * c23 + a2 * c2) - d4 * c1
        T4[8] = s234; T4[9] = 0; T4[10] = -c234; T4[11] = d1 + a3 * s23 + a2 * s2
        T4[12] = 0; T4[13] = 0; T4[14] = 0; T4[15] = 1

    if T5 is not None:
        T5[0] = s1 * s5 + c234 * c1 * c5; T5[1] = -s234 * c1; T5[2] = c5 * s1 - c234 * c1 * s5
        T5[3] = c1 * (a3 * c23 + a2 * c2) + d4 * s1 + d5 * s234 * c1
        T5[4] = c234 * c5 * s1 - c1 * s5; T5[5] = -s234 * s1; T5[6] = -c1 * c5 - c234 * s1 * s5
        T5[7] = s1 * (a3 * c23 + a2 * c2) - d4 * c1 + d5 * s234 * s1
        T5[8] = s234 * c5; T5[9] = c234; T5[10] = -s234 * s5; T5[11] = d1 + a3 * s23 + a2 * s2 - d5 * c234
        T5[12] = 0; T5[13] = 0; T5[14] = 0; T5[15] = 1

    if T6 is not None:
        T6[0] = c6 * (s1 * s5 + c234 * c1 * c5) - s234 * c1 * s6
        T6[1] = -s6 * (s1 * s5 + c234 * c1 * c5) - s234 * c1 * c6
        T6[2] = c5 * s1 - c234 * c1 * s5
        T6[3] = d6 * (c5 * s1 - c234 * c1 * s5) + c1 * (a3 * c23 + a2 * c2) + d4 * s1 + d5 * s234 * c1
        T6[4] = -c6 * (c1 * s5 - c234 * c5 * s1) - s234 * s1 * s6
        T6[5] = s6 * (c1 * s5 - c234 * c5 * s1) - s234 * c6 * s1
        T6[6] = -c1 * c5 - c234 * s1 * s5
        T6[7] = s1 * (a3 * c23 + a2 * c2) - d4 * c1 - d6 * (c1 * c5 + c234 * s1 * s5) + d5 * s234 * s1
        T6[8] = c234 * s6 + s234 * c5 * c6
        T6[9] = c234 * c6 - s234 * c5 * s6
        T6[10] = -s234 * s5
        T6[11] = d1 + a3 * s23 + a2 * s2 - d5 * c234 - d6 * s234 * s5
        T6[12] = 0; T6[13] = 0; T6[14] = 0; T6[15] = 1

def inverse(T, q_sols, q6_des):
    num_sols = 0

    # Extract the transformation matrix elements
    T02 = -T[0]; T00 = T[1]; T01 = T[2]; T03 = -T[3]
    T12 = -T[4]; T10 = T[5]; T11 = T[6]; T13 = -T[7]
    T22 = T[8]; T20 = -T[9]; T21 = -T[10]; T23 = T[11]

    # T02, T00, T01, T03 = -T[0, 2], T[0, 0], T[0, 1], -T[0, 3]
    # T12, T10, T11, T13 = -T[1, 2], T[1, 0], T[1, 1], -T[1, 3]
    # T22, T20, T21, T23 = T[2, 2], -T[2, 0], -T[2, 1], T[2, 3]

    # Solve for q1
    q1 = [0, 0]
    A = d6 * T12 - T13
    B = d6 * T02 - T03
    R = A * A + B * B

    if abs(A) < ZERO_THRESH:
        if abs(abs(d4) - abs(B)) < ZERO_THRESH:
            div = -SIGN(d4) * SIGN(B)
        else:
            div = -d4 / B
        arcsin = math.asin(div)
        if abs(arcsin) < ZERO_THRESH:
            arcsin = 0.0
        if arcsin < 0.0:
            q1[0] = arcsin + 2.0 * PI
        else:
            q1[0] = arcsin
        q1[1] = PI - arcsin
    elif abs(B) < ZERO_THRESH:
        if abs(abs(d4) - abs(A)) < ZERO_THRESH:
            div = SIGN(d4) * SIGN(A)
        else:
            div = d4 / A
        arccos = math.acos(div)
        q1[0] = arccos
        q1[1] = 2.0 * PI - arccos
    elif d4 * d4 > R:
        return num_sols
    else:
        arccos = math.acos(d4 / math.sqrt(R))
        arctan = math.atan2(-B, A)
        pos = arccos + arctan
        neg = -arccos + arctan
        if abs(pos) < ZERO_THRESH:
            pos = 0.0
        if abs(neg) < ZERO_THRESH:
            neg = 0.0
        if pos >= 0.0:
            q1[0] = pos
        else:
            q1[0] = 2.0 * PI + pos
        if neg >= 0.0:
            q1[1] = neg
        else:
            q1[1] = 2.0 * PI + neg

    # Solve for q5
    q5 = [[0, 0], [0, 0]]
    for i in range(2):
        numer = (T03 * math.sin(q1[i]) - T13 * math.cos(q1[i]) - d4)
        if abs(abs(numer) - abs(d6)) < ZERO_THRESH:
            div = SIGN(numer) * SIGN(d6)
        else:
            div = numer / d6
        arccos = math.acos(div)
        q5[i][0] = arccos
        q5[i][1] = 2.0 * PI - arccos

    # Solve for q6, q2, q3, q4
    for i in range(2):
        for j in range(2):
            c1 = math.cos(q1[i])
            s1 = math.sin(q1[i])
            c5 = math.cos(q5[i][j])
            s5 = math.sin(q5[i][j])

            # Solve for q6
            if abs(s5) < ZERO_THRESH:
                q6 = q6_des
            else:
                q6 = math.atan2(SIGN(s5) * -(T01 * s1 - T11 * c1), SIGN(s5) * (T00 * s1 - T10 * c1))
                if abs(q6) < ZERO_THRESH:
                    q6 = 0.0

            # Solve for q2, q3, q4
            c6 = math.cos(q6)
            s6 = math.sin(q6)
            x04x = -s5 * (T02 * c1 + T12 * s1) - c5 * (s6 * (T01 * c1 + T11 * s1) - c6 * (T00 * c1 + T10 * s1))
            x04y = c5 * (T20 * c6 - T21 * s6) - T22 * s5
            p13x = d5 * (s6 * (T00 * c1 + T10 * s1) + c6 * (T01 * c1 + T11 * s1)) - d6 * (T02 * c1 + T12 * s1) + T03 * c1 + T13 * s1
            p13y = T23 - d1 - d6 * T22 + d5 * (T21 * c6 + T20 * s6)

            c3 = (p13x * p13x + p13y * p13y - a2 * a2 - a3 * a3) / (2.0 * a2 * a3)
            if abs(abs(c3) - 1.0) < ZERO_THRESH:
                c3 = SIGN(c3)
            elif abs(c3) > 1.0:
                continue  # No solution

            arccos = math.acos(c3)
            q3 = [arccos, 2.0 * PI - arccos]
            denom = a2 * a2 + a3 * a3 + 2 * a2 * a3 * c3
            s3 = math.sin(arccos)
            A = (a2 + a3 * c3)
            B = a3 * s3

            q2 = [
                math.atan2((A * p13y - B * p13x) / denom, (A * p13x + B * p13y) / denom),
                math.atan2((A * p13y + B * p13x) / denom, (A * p13x - B * p13y) / denom)
            ]

            c23_0 = math.cos(q2[0] + q3[0])
            s23_0 = math.sin(q2[0] + q3[0])
            c23_1 = math.cos(q2[1] + q3[1])
            s23_1 = math.sin(q2[1] + q3[1])

            q4 = [
                math.atan2(c23_0 * x04y - s23_0 * x04x, x04x * c23_0 + x04y * s23_0),
                math.atan2(c23_1 * x04y - s23_1 * x04x, x04x * c23_1 + x04y * s23_1)
            ]

            for k in range(2):
                if abs(q2[k]) < ZERO_THRESH:
                    q2[k] = 0.0
                if abs(q4[k]) < ZERO_THRESH:
                    q4[k] = 0.0
                elif q4[k] < 0.0:
                    q4[k] += 2.0 * PI

                q_sols[num_sols * 6 + 0] = q1[i]
                q_sols[num_sols * 6 + 1] = q2[k]
                q_sols[num_sols * 6 + 2] = q3[k]
                q_sols[num_sols * 6 + 3] = q4[k]
                q_sols[num_sols * 6 + 4] = q5[i][j]
                q_sols[num_sols * 6 + 5] = q6
                num_sols += 1

    return num_sols