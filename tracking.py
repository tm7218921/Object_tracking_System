import cv2

def initialize_tracker(tracking_algorithm="CSRT"):
    if tracking_algorithm == "CSRT":
        return cv2.TrackerCSRT_create()
    elif tracking_algorithm == "KCF":
        return cv2.TrackerKCF_create()
    else:
        raise ValueError("Unsupported tracking algorithm")
