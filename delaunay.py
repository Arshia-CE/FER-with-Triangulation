import cv2
import imutils
import numpy as np
from face_recognition import detect_landmarks


# Check if a point is inside a rectangle
def rect_contains(rectangle, point):
    if point[0] < rectangle[0]:
        return False
    elif point[1] < rectangle[1]:
        return False
    elif point[0] > rectangle[2]:
        return False
    elif point[1] > rectangle[3]:
        return False
    return True


# Draw a point
def draw_point(img, p, color):
    cv2.circle(img, p, 2, color, cv2.FILLED, cv2.LINE_AA, 0)


# Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color):
    triangle_list = subdiv.getTriangleList()
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangle_list:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)


def delaunay_triangulation(img, points):
    # Create an instance of Subdiv2D
    rect = (0, 0, img.shape[1], img.shape[0])
    subdiv = cv2.Subdiv2D(rect)

    # Insert points into subdiv
    for p in points:
        if 0 <= p[0] < img.shape[1] and 0 <= p[1] < img.shape[0]:
            subdiv.insert(p)

    return subdiv


def compute_barycenter(triangles):
    barycenters = []
    for triangle in triangles:
        # Get coordinates of triangle vertices
        p1 = (triangle[0], triangle[1])
        p2 = (triangle[2], triangle[3])
        p3 = (triangle[4], triangle[5])
        # Compute barycenter
        barycenter = np.mean(np.array([p1, p2, p3]), axis=0)
        barycenters.append(barycenter)

    return barycenters


def combine_points_with_barycenter(triangles, barycenters):
    combined_points = []
    for triangle, barycenter in zip(triangles, barycenters):
        # Combine points of triangle with barycenter
        combined_triangle = np.concatenate((triangle, barycenter))
        combined_points.append(combined_triangle)
    return combined_points


def extract_features(img):
    _, landmarks = detect_landmarks(img)
    # Compute triangulated face
    subdiv = delaunay_triangulation(img, landmarks)

    #  Compute barycenter of each triangle
    barycenters = compute_barycenter(triangles=subdiv.getTriangleList())

    # Combine points of triangles with barycenters
    combined_points = combine_points_with_barycenter(subdiv.getTriangleList(), barycenters)

    # flatten for our models
    combined_points = np.array(combined_points).flatten()
    # print(combined_points.shape)
    return combined_points, landmarks


if __name__ == '__main__':
    # Read the input image
    img = cv2.imread("dataset/fear/S506_004_00000038.png", 1)

    # Resize image keeping aspect-ratio to ensure no overflow in visualization
    img = imutils.resize(img, width=800)

    # Draw face Bounding box

    # If l28 only use 28 of the 68 landmarks provided by dlib shape-predictor # l28 = args.l28 #f l28: mask = [0, 2,
    # 4, 6, 8, 10, 12, 14, 16, 18, 20, 21, 22, 23, 25, 27, 29, 30, 31, 35, 36, 39, 42, 45, 48, 51, 54, 57] landmarks
    # = [landmarks[i] for i in mask]
    _, landmarks = detect_landmarks(img)
    
    # Compute triangulated face
    subdiv = delaunay_triangulation(img, landmarks)

    # Allocate space for the triangulated face
    triangulated_face = np.zeros_like(img)
    # Draw delaunay triangles
    draw_delaunay(triangulated_face, subdiv, (125, 125, 255))
    
    # Show original image and triangulated face
    cv2.imshow('Original Image', img)
    cv2.imshow('Triangulated Face', triangulated_face)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
