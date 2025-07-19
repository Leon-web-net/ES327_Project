def convert_bbox_to_yolo_format(bbox):
    """
    Convert bounding box from (x_min, y_min, x_max, y_max) format to YOLO format.
    
    Parameters:
        x_min (int/float): Minimum x-coordinate of the bounding box.
        y_min (int/float): Minimum y-coordinate of the bounding box.
        x_max (int/float): Maximum x-coordinate of the bounding box.
        y_max (int/float): Maximum y-coordinate of the bounding box.
        img_width (int/float): Width of the image.
        img_height (int/float): Height of the image.
    
    Returns:
        tuple: (x_center, y_center, bbox_width, bbox_height) in YOLO format (normalized).
    """
    img_width = 1600
    img_height = 900
    x_min, y_min, x_max, y_max = bbox
    # Compute bounding box width and height
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min

    # Ensure bounding box coordinates are within image boundaries
    x_center = max(0, min(x_min + bbox_width / 2, img_width))
    y_center = max(0, min(y_min + bbox_height / 2, img_height))
    
    # Ensure width and height are not negative and do not exceed image size
    bbox_width = max(0, min(bbox_width, img_width))
    bbox_height = max(0, min(bbox_height, img_height))

    # Normalize values by image dimensions
    x_center /= img_width
    y_center /= img_height
    bbox_width /= img_width
    bbox_height /= img_height

    # Round values to 6 decimal places
    x_center = round(x_center, 6)
    y_center = round(y_center, 6)
    bbox_width = round(bbox_width, 6)
    bbox_height = round(bbox_height, 6)

    return x_center, y_center, bbox_width, bbox_height

# Example usage
bbox = [900, 369, 931, 479]
  # Example image dimensions
yolo_bbox = convert_bbox_to_yolo_format(bbox)
print("YOLO Format Bounding Box:", yolo_bbox)