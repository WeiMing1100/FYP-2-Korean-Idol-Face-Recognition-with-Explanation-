from retinaface import RetinaFace
import cv2
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1
import streamlit as st

@st.cache_resource
def load_model():
    # Load the PTM(pre-trained model) with its pre-trained weights base on facial recognition dataset
    model = InceptionResnetV1(pretrained='vggface2')

    # Must be set to True to be used in classification
    model.classify = True

    # Freeze the layers in our PTM
    for param in model.parameters():
        param.requires_grad = False

    # Change the last layer named logits to the ff layer with 4 output channels
    model.logits = nn.Sequential(
        nn.Linear(512*1*1, 512),  # Extract embeddings
        nn.ReLU(),                # Add a ReLU activation
        nn.BatchNorm1d(512),  # Batch norm can improve convergence
        nn.Dropout(0.2),           # Add regularization
        nn.Linear(512, 25),       # change to how many idols
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Transfer the model to device (GPU) for faster training if GPU available
    model = model.to(device)

    # Load trained model weights from the file
    model.load_state_dict(torch.load('model_weights (phase 3).pth', map_location=device))
    model.eval()

    return model, device

model, device = load_model()

def get_embedding(img_input):
    img_tensor = preprocess_image(img_input).to(device)

    with torch.no_grad():
        embedding = model(img_tensor)

    return embedding.squeeze(0).cpu().numpy()


def extract_faces(img_path):
    faces = RetinaFace.extract_faces(img_path=img_path)

    landmark_detections = RetinaFace.detect_faces(img_path)


    if faces is not None and len(faces) > 0:  # Check if at least one face is detected and aligned
        # Use the first face (index 0) from the detected faces
        aligned_face = faces[0]  # Only keep the first face

        # Resize the aligned face to 256x256
        resized_face = cv2.resize(aligned_face, (256, 256))

        # Convert from BGR to RGB (RetinaFace returns images in BGR format)
        aligned_face_rgb = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)

        # Get bounding box of the first face
        facial_area = landmark_detections["face_1"]["facial_area"]  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = facial_area
        orig_w, orig_h = x2 - x1, y2 - y1

        # Scale landmarks into 256x256
        original_landmarks = landmark_detections["face_1"]["landmarks"]
        aligned_landmarks = {}
        for key, (lx, ly) in original_landmarks.items():
            new_x = int((lx - x1) * (256 / orig_w))
            new_y = int((ly - y1) * (256 / orig_h))
            aligned_landmarks[key] = (new_x, new_y)

        return aligned_face_rgb, aligned_landmarks
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


