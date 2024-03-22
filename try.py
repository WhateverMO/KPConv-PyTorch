def read_pts_file(file_path,lable = True):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    points = []
    lables = []
    for line in lines:
        parts = line.split()
        point = [float(it) for it in parts[:3]]
        points.append(point)
        if lable:
            lables.append(int(parts[-1]))
    return points,lables

points,lables = read_pts_file('../../Data/ISPRS3D/Vaihingen3D_Traininig.pts')
input('done!')

from collections import Counter

def calculate_label_distribution(labels):
    counter = Counter(labels)
    total_count = len(labels)
    distribution = {label: count / total_count for label, count in counter.items()}
    return distribution

distribution = calculate_label_distribution(lables)
for label, percentage in distribution.items():
    print(f"Label {label}: {percentage * 100:.2f}%")