import cv2
import mediapipe as mp
import os

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.5)

input_root = "dataset"
output_root = "eye_dataset"

for category in ["closed", "open"]:
    input_folder = os.path.join(input_root, category)
    output_folder = os.path.join(output_root, category)

    os.makedirs(output_folder, exist_ok=True)

    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)

        image = cv2.imread(img_path)
        if image is None:
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_image)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = image.shape

                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                face = image[y:y+height, x:x+width]

                if face.size == 0:
                    continue

                output_path = os.path.join(output_folder, img_name)
                cv2.imwrite(output_path, face)

print("Face Cropping Completed!")
