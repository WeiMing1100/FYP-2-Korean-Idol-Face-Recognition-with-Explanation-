from retinaface import RetinaFace
import cv2
from PIL import Image
from torchvision import transforms

def extract_faces(img_path):
    faces = RetinaFace.extract_faces(img_path=img_path)

    if faces is not None and len(faces) > 0:  # Check if at least one face is detected and aligned
        # Use the first face (index 0) from the detected faces
        aligned_face = faces[0]  # Only keep the first face

        # Resize the aligned face to 256x256
        resized_face = cv2.resize(aligned_face, (256, 256))

        # Convert from BGR to RGB (RetinaFace returns images in BGR format)
        aligned_face_rgb = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)

        return aligned_face_rgb
    else:
      return None


def preprocess_image(img_input):
    if isinstance(img_input, str):
        img = Image.open(img_input).convert("RGB")
    else:
        # NumPy array
        img = Image.fromarray(img_input)

    preprocess = transforms.Compose([
        transforms.Resize(size=(160, 160)),  # Adjust as needed (follow what you did during training phase (for test dataloaders))
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img_tensor = preprocess(img).unsqueeze(0)
    return img_tensor


# def load_custom_labels():
#     dataset_path = ''
#     # Replace with your custom class names
#     class_names = [
#         folder_name for folder_name in os.listdir(dataset_path)
#         if os.path.isdir(os.path.join(dataset_path, folder_name))
#     ]
#     class_names.sort(key=lambda x: x.strip().lower()) # REMEMBER to sort the classes!
#     return class_names


