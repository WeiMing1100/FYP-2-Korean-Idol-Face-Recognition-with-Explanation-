import torch
import torch.nn as nn
# from torchvision import models
# from torch.autograd import Variable
from facenet_pytorch import InceptionResnetV1
from gradcam import GradCAM, visualize_gradCAM_results
from visualize_results import *
from utils import *
import numpy as np


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


# I don't want to put this function in utils.py cuz don't want to import model
def get_embedding(img_input):
    img_tensor = preprocess_image(img_input).to(device)

    with torch.no_grad():
        embedding = model(img_tensor)

    return embedding.squeeze(0).cpu().numpy()


query_image_path = '/content/drive/MyDrive/FYP2/test_moonbyul.png' # PLACEHOLDER FOR NOW
grad_cam = GradCAM(model)
extracted_face = extract_faces(query_image_path)
# I have already called preprocess_image in apply_grad_cam (which is called inside visualize results)
visualize_gradCAM_results(query_image_path, extracted_face, model, grad_cam)

idol_embeddings = np.load("idol_embeddings.npy")
idol_labels = np.load("idol_labels.npy", allow_pickle=True)  # if labels are strings
query_embeddings = get_embedding(extracted_face)

# visualize embeddings
best_idx, similarity_score = visualize_embeddings(idol_embeddings, query_embeddings, idol_labels)

predicted_label = idol_labels[best_idx]
print("Predicted idol:", predicted_label)
print("Similarity:", similarity_score)

similar_img_path = '/content/drive/MyDrive/FYP2/dataset (phase 3)/phase_3/lee chaeyeon/chaeyeon (14).png' # PLACEHOLDER FOR NOW
visualize_similar_images(query_image_path, similar_img_path, predicted_label, np.round(similarity_score, decimals=4))







