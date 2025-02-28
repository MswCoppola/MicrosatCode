import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


def compute_slope(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return dy / dx if dx != 0 else np.inf


def match_vertices_series(series, threshold_factor=2):
    all_matches = []
    point_tracks = {i: [tuple(series[0][i])] for i in range(len(series[0]))}  # Initialize tracking
    active_tracks = {i: i for i in range(len(series[0]))}  # Track active indices

    for i in range(len(series) - 1):
        points1, points2 = series[i], series[i + 1]
        cost_matrix = np.linalg.norm(points1[:, np.newaxis, :] - points2[np.newaxis, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_pairs = list(zip(row_ind, col_ind))

        match_distances = [cost_matrix[i1, i2] for i1, i2 in matched_pairs]
        distance_threshold = threshold_factor * np.mean(match_distances)

        filtered_matches = []
        used_points2 = set()
        for i1, i2 in matched_pairs:
            slope = compute_slope(points1[i1], points2[i2])
            if cost_matrix[i1, i2] <= distance_threshold and abs(slope) <= 1 and i2 not in used_points2:
                filtered_matches.append((i1, i2))
                used_points2.add(i2)
            else:
                alternative_matches = sorted(
                    [(j, cost_matrix[i1, j]) for j in range(len(points2)) if j != i2 and j not in used_points2],
                    key=lambda x: x[1])
                for j, _ in alternative_matches:
                    new_slope = compute_slope(points1[i1], points2[j])
                    if abs(new_slope) <= 1:
                        filtered_matches.append((i1, j))
                        used_points2.add(j)
                        break

        all_matches.append(filtered_matches)

        new_tracks = {}
        for i1, i2 in filtered_matches:
            if i1 in active_tracks:
                track_id = active_tracks[i1]
                point_tracks[track_id].append(tuple(points2[i2]))
                new_tracks[i2] = track_id
            else:
                track_id = max(point_tracks.keys()) + 1
                point_tracks[track_id] = [tuple(points2[i2])]
                new_tracks[i2] = track_id

        active_tracks = new_tracks

    tracked_series = [track for track in point_tracks.values() if len(track) >= 5]

    return tracked_series, all_matches


def plot_matching_series(series, all_matches):
    for i in range(len(series) - 1):
        points1, points2 = series[i], series[i + 1]
        matches = all_matches[i]

        plt.figure(figsize=(6, 6))
        for j, (x, y) in enumerate(points1):
            plt.scatter(x, y, color='red', label="Frame " + str(i) if j == 0 else "", s=50)
            plt.text(x, y, f"{j}", fontsize=12, color='red', verticalalignment='bottom')
        for j, (x, y) in enumerate(points2):
            plt.scatter(x, y, color='blue', label="Frame " + str(i + 1) if j == 0 else "", s=50)
            plt.text(x, y, f"{j}", fontsize=12, color='blue', verticalalignment='bottom')
        for i1, i2 in matches:
            x_values = [points1[i1, 0], points2[i2, 0]]
            y_values = [points1[i1, 1], points2[i2, 1]]
            plt.plot(x_values, y_values, 'gray', linestyle='dashed', alpha=0.7)
        plt.legend()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Vertex Correspondences Between Frame {i} and Frame {i + 1}")
        plt.gca().invert_yaxis()
        plt.show()


def grouping():
    series = [
        np.array([(536, 270), (513, 301), (553, 328), (573, 293)]),
        np.array([(497, 296), (459, 332), (513, 441), (546, 410), (467, 270), (429, 301)]),
        np.array([(385, 332), (557, 447), (591, 412), (423, 298), (419, 273)]),
        np.array([(362, 321), (581, 448), (616, 420), (385, 295), (393, 271), (620, 390), (497, 350)]),
        np.array([(367, 289), (601, 424), (623, 398), (389, 269)]),
        np.array([(612, 405), (579, 431), (583, 460), (615, 433), (381, 289), (410, 270)]),
        np.array([(577, 408), (537, 433), (553, 466), (594, 437), (409, 280), (561, 408), (426, 271)]),
        np.array([(556, 412), (514, 438), (537, 468), (579, 441)]),
        np.array([(526, 409), (484, 437), (512, 469), (554, 443), (471, 277), (550, 428), (490, 296)]),
        np.array([(515, 408), (482, 439), (506, 469), (546, 438), (482, 274), (507, 297)])
    ]
    tracked_series, all_matches = match_vertices_series(series)
    plot_matching_series(series, all_matches)
    print("Tracked Point Series:")
    print(tracked_series)


if __name__ == "__main__":
    grouping()


