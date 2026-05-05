import cv2
import numpy as np

def find_average_oval_area(image_path, show=False):
    image = cv2.imread(image_path)
    output = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas = []

    for cnt in contours:
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (x, y), (major_axis, minor_axis), angle = ellipse

            # Filter out junk detections
            if major_axis > 20 and minor_axis > 20:
                area = np.pi * (major_axis / 2) * (minor_axis / 2)
                areas.append(area)

                if show:
                    cv2.ellipse(output, ellipse, (0, 255, 0), 2)

    # Compute average
    if len(areas) > 0:
        avg_area = sum(areas) / len(areas)
    else:
        avg_area = 0

    print(f"Detected {len(areas)} ovals")
    print(f"Average area: {avg_area:.2f} px^2")

    if show:
        cv2.imshow("Detected Ovals", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return avg_area


# Example usage
avg = find_average_oval_area("image1.jpg", show=True)