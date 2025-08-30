# from torchvision import models
# from torch.autograd import Variable
from gradcam import GradCAM, visualize_gradCAM_results
from visualize_results import *
from utils import *
import numpy as np
import streamlit as st
import cv2
import os
import math

st.title("Korean Idol Face Recognition with Explanation")


st.header("List of Korean Idols")

# Items
items = list(range(1, 26))
items_per_page = 5
item_width = 120  # must match CSS width
gap = 10

# Number of pages (ceil division)
total_pages = math.ceil(len(items) / items_per_page)

# Session state
if "page" not in st.session_state:
    st.session_state.page = 0

# Navigation
col1, spacer, col2 = st.columns([1, 9, 1])  # middle col is just empty space
with col1:
    st.button("⬅️", on_click=lambda: st.session_state.update(
        {"page": max(0, st.session_state.page - 1)}
    ))
with col2:
    st.button("➡️", on_click=lambda: st.session_state.update(
        {"page": min(total_pages - 1, st.session_state.page + 1)}
    ))


# Container width = exactly 5 items
container_width = items_per_page * (item_width + gap)

# shift by whole container width per page
shift = -(st.session_state.page * container_width)

# Carousel (scroll illusion)
carousel = f"""
<div style="width: {container_width}px; overflow: hidden; margin: auto;">
  <div style="
      display: flex;
      transform: translateX({shift}px);
      transition: transform 0.6s ease;
  ">
    <div style="min-width: 120px; height: 100px; margin-right: 10px;
        background-color: lightblue; border: 2px solid #333; border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-weight: bold; font-size: 20px;">1</div>
    <div style="min-width: 120px; height: 100px; margin-right: 10px;
        background-color: lightblue; border: 2px solid #333; border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-weight: bold; font-size: 20px;">2</div>
    <div style="min-width: 120px; height: 100px; margin-right: 10px;
        background-color: lightblue; border: 2px solid #333; border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-weight: bold; font-size: 20px;">3</div>
    <div style="min-width: 120px; height: 100px; margin-right: 10px;
        background-color: lightblue; border: 2px solid #333; border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-weight: bold; font-size: 20px;">4</div>
    <div style="min-width: 120px; height: 100px; margin-right: 10px;
        background-color: lightblue; border: 2px solid #333; border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-weight: bold; font-size: 20px;">5</div>
    <div style="min-width: 120px; height: 100px; margin-right: 10px;
        background-color: lightblue; border: 2px solid #333; border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-weight: bold; font-size: 20px;">6</div>
    <div style="min-width: 120px; height: 100px; margin-right: 10px;
        background-color: lightblue; border: 2px solid #333; border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-weight: bold; font-size: 20px;">7</div>
    <div style="min-width: 120px; height: 100px; margin-right: 10px;
        background-color: lightblue; border: 2px solid #333; border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-weight: bold; font-size: 20px;">8</div>
    <div style="min-width: 120px; height: 100px; margin-right: 10px;
        background-color: lightblue; border: 2px solid #333; border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-weight: bold; font-size: 20px;">9</div>
    <div style="min-width: 120px; height: 100px; margin-right: 10px;
        background-color: lightblue; border: 2px solid #333; border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-weight: bold; font-size: 20px;">10</div>
    <div style="min-width: 120px; height: 100px; margin-right: 10px;
        background-color: lightblue; border: 2px solid #333; border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-weight: bold; font-size: 20px;">11</div>
    <div style="min-width: 120px; height: 100px; margin-right: 10px;
        background-color: lightblue; border: 2px solid #333; border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-weight: bold; font-size: 20px;">12</div>
    <div style="min-width: 120px; height: 100px; margin-right: 10px;
        background-color: lightblue; border: 2px solid #333; border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-weight: bold; font-size: 20px;">13</div>
    <div style="min-width: 120px; height: 100px; margin-right: 10px;
        background-color: lightblue; border: 2px solid #333; border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-weight: bold; font-size: 20px;">14</div>
    <div style="min-width: 120px; height: 100px; margin-right: 10px;
        background-color: lightblue; border: 2px solid #333; border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-weight: bold; font-size: 20px;">15</div>
    <div style="min-width: 120px; height: 100px; margin-right: 10px;
        background-color: lightblue; border: 2px solid #333; border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-weight: bold; font-size: 20px;">16</div>
    <div style="min-width: 120px; height: 100px; margin-right: 10px;
        background-color: lightblue; border: 2px solid #333; border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-weight: bold; font-size: 20px;">17</div>
    <div style="min-width: 120px; height: 100px; margin-right: 10px;
        background-color: lightblue; border: 2px solid #333; border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-weight: bold; font-size: 20px;">18</div>
    <div style="min-width: 120px; height: 100px; margin-right: 10px;
        background-color: lightblue; border: 2px solid #333; border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-weight: bold; font-size: 20px;">19</div>
    <div style="min-width: 120px; height: 100px; margin-right: 10px;
        background-color: lightblue; border: 2px solid #333; border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-weight: bold; font-size: 20px;">20</div>
    <div style="min-width: 120px; height: 100px; margin-right: 10px;
        background-color: lightblue; border: 2px solid #333; border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-weight: bold; font-size: 20px;">21</div>
    <div style="min-width: 120px; height: 100px; margin-right: 10px;
        background-color: lightblue; border: 2px solid #333; border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-weight: bold; font-size: 20px;">22</div>
    <div style="min-width: 120px; height: 100px; margin-right: 10px;
        background-color: lightblue; border: 2px solid #333; border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-weight: bold; font-size: 20px;">23</div>
    <div style="min-width: 120px; height: 100px; margin-right: 10px;
        background-color: lightblue; border: 2px solid #333; border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-weight: bold; font-size: 20px;">24</div>
    <div style="min-width: 120px; height: 100px; margin-right: 10px;
        background-color: lightblue; border: 2px solid #333; border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-weight: bold; font-size: 20px;">25</div>
  </div>
</div>
"""

# Render
st.markdown(carousel, unsafe_allow_html=True)

st.write(f"Page {st.session_state.page + 1} of {total_pages}")

st.divider()

uploaded_image = st.file_uploader("Upload an image of a Korean Idol or Yourself (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"],
                                         accept_multiple_files=False)



if uploaded_image is not None:
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)



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
        similarity_score_percent = np.round(similarity_score, decimals=4)*100
        print("Predicted idol:", predicted_label)
        print("Similarity:", similarity_score)

        idol_image_pair = {
            "an yujin": os.path.join("pairs", "an yujin", "yujin.png"),
            "chaeryeong": os.path.join("pairs", "chaeryeong", "chaeryeong.png"),
            "hani": os.path.join("pairs", "hani", "hani.png"),
            "heejin": os.path.join("pairs", "heejin", "heejin.png"),
            "irene": os.path.join("pairs", "irene", "irene.png"),
            "kwon eunbi": os.path.join("pairs", "kwon eunbi", "eunbi.png"),
            "lee chaeyeon": os.path.join("pairs", "lee chaeyeon", "chaeyeon.png"),
            "lisa": os.path.join("pairs", "lisa", "lisa.png"),
            "mina": os.path.join("pairs", "mina", "mina.png"),
            "momo": os.path.join("pairs", "momo", "momo.png"),
            "moonbyul": os.path.join("pairs", "moonbyul", "moonbyul.png"),
            "nayeon": os.path.join("pairs", "nayeon", "nayeon.png"),
            "rose": os.path.join("pairs", "rose", "rose.png"),
            "ryujin": os.path.join("pairs", "ryujin", "ryujin.png"),
            "seulgi": os.path.join("pairs", "seulgi", "seulgi.png"),
            "sinB": os.path.join("pairs", "sinB", "sinb.png"),
            "soojin": os.path.join("pairs", "soojin", "soojin.png"),
            "soyeon": os.path.join("pairs", "soyeon", "soyeon.png"),
            "tzuyu": os.path.join("pairs", "tzuyu", "tzuyu.png"),
            "wheein": os.path.join("pairs", "wheein", "wheein.png"),
            "yeji": os.path.join("pairs", "yeji", "yeji.png"),
            "yena": os.path.join("pairs", "yena", "yena.png"),
            "yuna": os.path.join("pairs", "yuna", "yuna.png"),
            "yuqi": os.path.join("pairs", "yuqi", "yuqi.png"),
            "yves": os.path.join("pairs", "yves", "yves.png"),
        }

        similar_image = idol_image_pair[predicted_label]
        visualize_similar_images(uploaded_image, similar_image, predicted_label)

        st.text(f"Your submitted image is probably not a Kpop Idol. This person looks the most similar to :orange[{predicted_label}], with similarity score of {similarity_score_percent}%!")





