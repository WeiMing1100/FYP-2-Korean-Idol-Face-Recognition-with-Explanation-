# from torchvision import models
# from torch.autograd import Variable
from gradcam import GradCAM, visualize_gradCAM_results
from visualize_results import *
from utils import *
import numpy as np
import streamlit as st
import cv2

# requirements.txt
# facenet-pytorch==2.6.0
# retina-face==0.0.17
# numpy==1.26.4
# streamlit==1.49.0

st.title("Korean Idol Face Recognition with Explanation")

uploaded_image = st.file_uploader("Upload an image of a Korean Idol or Yourself (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"],
                                         accept_multiple_files=False)


if uploaded_image is not None:
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)




    # query_image_path = '/content/drive/MyDrive/FYP2/test_moonbyul.png' # PLACEHOLDER FOR NOW
    grad_cam = GradCAM(model)
    extracted_face = extract_faces(img_rgb)
    # I have already called preprocess_image in apply_grad_cam (which is called inside visualize results)
    is_kpop_idol = visualize_gradCAM_results(img_rgb, extracted_face, model, grad_cam)


    if not is_kpop_idol:
        idol_embeddings = np.load("idol_embeddings.npy")
        idol_labels = np.load("idol_labels.npy", allow_pickle=True)  # if labels are strings
        query_embeddings = get_embedding(extracted_face)

        # visualize embeddings
        best_idx, similarity_score = visualize_embeddings(idol_embeddings, query_embeddings, idol_labels)

        predicted_label = idol_labels[best_idx]
        print("Predicted idol:", predicted_label)
        print("Similarity:", similarity_score)

        # similar_img_path = '/content/drive/MyDrive/FYP2/dataset (phase 3)/phase_3/lee chaeyeon/chaeyeon (14).png' # PLACEHOLDER FOR NOW
        # visualize_similar_images(uploaded_image, similar_img_path, predicted_label, np.round(similarity_score, decimals=4))







