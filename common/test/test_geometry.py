from common.geometry.point import Point
from common.geometry.geometry import convex_hull

if __name__ == "__main__":
    points = []
    points.append(Point(0, 3, 0))
    points.append(Point(2, 2, 0))
    points.append(Point(1, 1, 0))
    points.append(Point(2, 1, 0))
    points.append(Point(3, 0, 0))
    points.append(Point(0, 0, 0))
    points.append(Point(3, 3, 0))

    hull = convex_hull(points)
    for point in hull:
        print(point)
