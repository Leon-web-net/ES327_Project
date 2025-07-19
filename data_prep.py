from nuimages import NuImages
import os 
import shutil
from collections import defaultdict
from pascal_to_yolo import convert_bbox_to_yolo_format

Labels = {
    "pedestrian":0,
    "car":1,
    "bus":2,
    "bicycle":3,
    "truck":4,
    "motorcycle":5,
          }

# Define minimum number of instances required for each class
min_instances_per_class = {
    0: 400,  # Pedestrian
    1: 400,  # Car
    2: 150,   # Bus
    3: 150,   # Bicycle
    4: 150,   # Truck
    5: 150,   # Motorcycle
}


def Return_class_idx(category, Labels = Labels):

     #pedestrian must be before car to avoid human.pedestrian.police_officer misclassification
    if "pedestrian" in category:   
        return Labels["pedestrian"]
    elif "car" in category or "police" in category:
        return Labels["car"]
    elif "bus" in category:
        return Labels["bus"]
    elif "bicycle" in category:
        return Labels["bicycle"]
    elif "truck" in category or "ambulance" in category:
        return Labels["truck"]
    elif "motorcycle" in category:
        return Labels["motorcycle"]
    else:
        return None  # Skip categories that don't fit

NEW_DATASET_DIR = "./Nuim_dataset_Yv8_test"
NUIM_DIR = "./Nuscenes_img"


os.makedirs(f"{NEW_DATASET_DIR}/images/train", exist_ok=True)
os.makedirs(f"{NEW_DATASET_DIR}/images/val", exist_ok=True)

os.makedirs(f"{NEW_DATASET_DIR}/labels/train", exist_ok=True)
os.makedirs(f"{NEW_DATASET_DIR}/labels/val", exist_ok=True)

nuim = NuImages(dataroot="./Nuscenes_img",version="v1.0-train")

train_class_counts = defaultdict(int)
val_class_counts = defaultdict(int)

# Initialize counters
idx = 20000
file_count = 1
train_count = 0
val_count = 0
is_train = True

# Ratio for splitting into training and validation sets
train_ratio = 1

# Function to check if a class has exceeded its minimum count
def class_exceeds_min_count(class_num, train_counts, val_counts, min_instances):
    total_instances = train_counts.get(class_num, 0) + val_counts.get(class_num, 0)
    return total_instances >= min_instances

# Function to check if at least one class in the image is still below its minimum count
def image_has_underrepresented_classes(object_tokens, train_counts, val_counts, min_instances):
    for obj_token in object_tokens:
        object_ann = nuim.get("object_ann", obj_token)
        category = nuim.get("category", object_ann["category_token"])["name"]
        class_num = Return_class_idx(category)
        if class_num is None:
            continue
        if not class_exceeds_min_count(class_num, train_counts, val_counts, min_instances[class_num]):
            return True
    return False

# Process samples until all classes meet the minimum instance requirement
while True:
    sample_0 = nuim.sample[idx]
    token = sample_0["token"]
    sample_data = nuim.get("sample_data", sample_0["key_camera_token"])
    filename = sample_data["filename"]
    # print(f"\n\n filename: {filename}")

    object_tokens, _ = nuim.list_anns(token, verbose=False)
    obj_len = len(object_tokens)

    if obj_len > 0:
        # Check if the image contains at least one class that hasn't met its minimum count
        if image_has_underrepresented_classes(object_tokens, train_class_counts, val_class_counts, min_instances_per_class):
            label_content = ""
            for obj_token in object_tokens:
                object_ann = nuim.get("object_ann", obj_token)
                category = nuim.get("category", object_ann["category_token"])["name"]

                class_num = Return_class_idx(category)
                if class_num is None:
                    continue

                bbox = object_ann["bbox"]
                yolo_bbox = convert_bbox_to_yolo_format(bbox)
                label_content += f"{class_num} {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}\n"

            if label_content:
                file_num = str(file_count).zfill(8)
                new_image_name = f"{file_num}.jpg"
                new_label_name = f"{file_num}.txt"

                # Determine if the image goes to train or val
                if train_count <= int((train_count + val_count) * train_ratio):
                    image_dir = f"{NEW_DATASET_DIR}/images/train"
                    label_dir = f"{NEW_DATASET_DIR}/labels/train"
                    is_train = True  # Flag to indicate this image is for training
                else:
                    image_dir = f"{NEW_DATASET_DIR}/images/val"
                    label_dir = f"{NEW_DATASET_DIR}/labels/val"
                    is_train = False  # Flag to indicate this image is for validation

                # Copy image and save label
                shutil.copy(os.path.join(NUIM_DIR, filename), f"{image_dir}/{new_image_name}")
                with open(f"{label_dir}/{new_label_name}", "w") as f:
                    f.write(label_content)

                # Increment class counts
                for obj_token in object_tokens:
                    object_ann = nuim.get("object_ann", obj_token)
                    category = nuim.get("category", object_ann["category_token"])["name"]
                    class_num = Return_class_idx(category)
                    if class_num is None:
                        continue
                    if is_train:
                                train_class_counts[class_num] += 1  # Increment train_class_counts
                    else:
                        val_class_counts[class_num] += 1  # Increment val_class_counts

                # Increment train_count or val_count only once per image
                if is_train:
                    train_count += 1
                else:
                    val_count += 1


                if file_count % 250 == 0:
                    print(f"Processed {file_count} files. Current class counts:")
                    print("Training dataset class counts:")
                    for class_num in sorted(train_class_counts.keys()):  # Display in order
                        print(f"Class {class_num}: {train_class_counts[class_num]} instances")
                    print("Validation dataset class counts:")
                    for class_num in sorted(val_class_counts.keys()):  # Display in order
                        print(f"Class {class_num}: {val_class_counts[class_num]} instances")
                    print("\n")

                file_count += 1

    idx += 1

    # Stop if we've exhausted the dataset or all classes meet their minimum counts
    if idx >= len(nuim.sample) or all(
        class_exceeds_min_count(class_num, train_class_counts, val_class_counts, min_instances)
        for class_num, min_instances in min_instances_per_class.items()
    ):
        break

# Print final class counts in order
print("Final training dataset class counts:")
for class_num in sorted(train_class_counts.keys()):  # Display in order
    print(f"Class {class_num}: {train_class_counts[class_num]} instances")

print("\nFinal validation dataset class counts:")
for class_num in sorted(val_class_counts.keys()):  # Display in order
    print(f"Class {class_num}: {val_class_counts[class_num]} instances")

# Check if all classes meet the minimum instance requirement
for class_num, min_instances in min_instances_per_class.items():
    total_instances = train_class_counts.get(class_num, 0) + val_class_counts.get(class_num, 0)
    if total_instances < min_instances:
        print(f"Warning: Class {class_num} has only {total_instances} instances (less than {min_instances}).")
    else:
        print(f"Class {class_num} has {total_instances} instances (meets the minimum requirement).")

# Generate YOLO .yaml file
dataset_relative_path = f"datasets/Nuim_dataset_Yv8"
yaml_content = f"""
# Path to the dataset root directory (relative or absolute)
path: {dataset_relative_path}  # Relative path to the dataset (when in MyDrive)

# Training and validation images directories (relative to 'path')
train: images/train  # Path to the training images
val: images/val      # Path to the validation images

# Class names (mapped to their respective indices)
names:
  0: pedestrian
  1: car
  2: bus
  3: bicycle
  4: truck
  5: motorcycle
"""

# Save the .yaml file
yaml_file_path = os.path.join(NEW_DATASET_DIR, "nuim_dataset.yaml")
with open(yaml_file_path, "w") as yaml_file:
    yaml_file.write(yaml_content)

print(f"YOLO .yaml file saved at: {yaml_file_path}")
print("Dataset split and renamed successfully!")
