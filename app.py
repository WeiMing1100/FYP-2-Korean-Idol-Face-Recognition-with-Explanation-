from gradcam import GradCAM, visualize_gradCAM_results, generate_textual_explanation_using_mediapipe_landmarks
from visualize_results import *
from utils import *
import numpy as np
import streamlit as st
import cv2
import os
import math
from css_code import bg_img_2
import pandas as pd

st.set_page_config(initial_sidebar_state="collapsed", layout="centered")
st.markdown(bg_img_2, unsafe_allow_html=True)
st.title("Korean Idol Face Recognition with Explanation")

st.subheader("List of Korean Idols")
st.markdown("""
<div class="custom-markdown-class">
    Top 25 best female dancers in KPOP based on 
    <a href="https://www.ranker.com/list/best-kpop-female-dancers-right-now/ranker-music?" target="_blank">ranker.com</a> (as of 31/8/25)</div>
""", unsafe_allow_html=True)

# Items
items = list(range(1, 26))
items_per_page = 5
item_width = 120
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

# Carousel (scroll illusion) (keep here instead of css_code.py for easier integration)
carousel = f"""
<div style="width: {container_width}px; overflow: hidden; margin: auto;">
    <div style="
      display: flex;
      transform: translateX({shift}px);
      transition: transform 0.6s ease;
    ">
        <div style="min-width: 120px; height: 100px; margin-right: 10px;
            background-color: #b0c4de; border: 2px solid #333; border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            font-weight: bold; font-size: 20px; text-align: center;">
            1. Lisa<br>(Blackpink)
        </div>
        <div style="min-width: 120px; height: 100px; margin-right: 10px;
            background-color: #b4c9df; border: 2px solid #333; border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            font-weight: bold; font-size: 20px; text-align: center;">
            2. Yeji<br>(ITZY)
        </div>
        <div style="min-width: 120px; height: 100px; margin-right: 10px;
            background-color: #b8cfe0; border: 2px solid #333; border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            font-weight: bold; font-size: 20px; text-align: center;">
            3. Momo<br>(TWICE)
        </div><div style="min-width: 120px; height: 100px; margin-right: 10px;
            background-color: #bcd4e1; border: 2px solid #333; border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            font-weight: bold; font-size: 20px; text-align: center;">
            4. Chaeryeong<br>(ITZY)
        </div><div style="min-width: 120px; height: 100px; margin-right: 10px;
            background-color: #b0e0e6; border: 2px solid #333; border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            font-weight: bold; font-size: 20px; text-align: center;">
            5. Lee Chaeyeon<br>(Soloist)
        </div><div style="min-width: 120px; height: 100px; margin-right: 10px;
            background-color: #b0e0e6; border: 2px solid #333; border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            font-weight: bold; font-size: 20px; text-align: center;">
            6. Seulgi<br>(Red Velvet)
        </div><div style="min-width: 120px; height: 100px; margin-right: 10px;
            background-color: #bcd4e1; border: 2px solid #333; border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            font-weight: bold; font-size: 20px; text-align: center;">
            7. Ryujin<br>(ITZY)
        </div><div style="min-width: 120px; height: 100px; margin-right: 10px;
            background-color: #b8cfe0; border: 2px solid #333; border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            font-weight: bold; font-size: 20px; text-align: center;">
            8. Rosé<br>(Blackpink)
        </div><div style="min-width: 120px; height: 100px; margin-right: 10px;
            background-color: #b4c9df; border: 2px solid #333; border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            font-weight: bold; font-size: 20px; text-align: center;">
            9. Yuna<br>(ITZY)
        </div><div style="min-width: 120px; height: 100px; margin-right: 10px;
            background-color: #b0c4de; border: 2px solid #333; border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            font-weight: bold; font-size: 20px; text-align: center;">
            10. Mina<br>(TWICE)
        </div><div style="min-width: 120px; height: 100px; margin-right: 10px;
            background-color: #b0c4de; border: 2px solid #333; border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            font-weight: bold; font-size: 20px; text-align: center;">
            11. Soojin<br>(I-DLE)
        </div><div style="min-width: 120px; height: 100px; margin-right: 10px;
            background-color: #b4c9df; border: 2px solid #333; border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            font-weight: bold; font-size: 20px; text-align: center;">
            12. Moonbyul<br>(Mamamoo)
        </div><div style="min-width: 120px; height: 100px; margin-right: 10px;
            background-color: #b8cfe0; border: 2px solid #333; border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            font-weight: bold; font-size: 20px; text-align: center;">
            13. Wheein<br>(Mamamoo)
        </div><div style="min-width: 120px; height: 100px; margin-right: 10px;
            background-color: #bcd4e1; border: 2px solid #333; border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            font-weight: bold; font-size: 20px; text-align: center;">
            14. Irene<br>(Red Velvet)
        </div><div style="min-width: 120px; height: 100px; margin-right: 10px;
            background-color: #b0e0e6; border: 2px solid #333; border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            font-weight: bold; font-size: 20px; text-align: center;">
            15. Kwon Eunbi<br>(Soloist)
        </div><div style="min-width: 120px; height: 100px; margin-right: 10px;
            background-color: #b0e0e6; border: 2px solid #333; border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            font-weight: bold; font-size: 20px; text-align: center;">
            16. SinB<br>(Gfriend)
        </div><div style="min-width: 120px; height: 100px; margin-right: 10px;
            background-color: #bcd4e1; border: 2px solid #333; border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            font-weight: bold; font-size: 20px; text-align: center;">
            17. Yuqi<br>(I-DLE)
        </div><div style="min-width: 120px; height: 100px; margin-right: 10px;
            background-color: #b8cfe0; border: 2px solid #333; border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            font-weight: bold; font-size: 20px; text-align: center;">
            18. Soyeon<br>(I-DLE)
        </div><div style="min-width: 120px; height: 100px; margin-right: 10px;
            background-color: #b4c9df; border: 2px solid #333; border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            font-weight: bold; font-size: 20px; text-align: center;">
            19. Nayeon<br>(TWICE)
        </div><div style="min-width: 120px; height: 100px; margin-right: 10px;
            background-color: #b0c4de; border: 2px solid #333; border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            font-weight: bold; font-size: 20px; text-align: center;">
            20. An Yujin<br>(IVE)
        </div><div style="min-width: 120px; height: 100px; margin-right: 10px;
            background-color: #b0c4de; border: 2px solid #333; border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            font-weight: bold; font-size: 20px; text-align: center;">
            21. Yves<br>(Soloist)
        </div><div style="min-width: 120px; height: 100px; margin-right: 10px;
            background-color: #b4c9df; border: 2px solid #333; border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            font-weight: bold; font-size: 20px; text-align: center;">
            22. Hani<br>(EXID)
        </div><div style="min-width: 120px; height: 100px; margin-right: 10px;
            background-color: #b8cfe0; border: 2px solid #333; border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            font-weight: bold; font-size: 20px; text-align: center;">
            23. HeeJin<br>(ARTMS)
        </div><div style="min-width: 120px; height: 100px; margin-right: 10px;
            background-color: #bcd4e1; border: 2px solid #333; border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            font-weight: bold; font-size: 20px; text-align: center;">
            24. Yena<br>(Soloist)
        </div><div style="min-width: 120px; height: 100px; margin-right: 10px;
            background-color: #b0e0e6; border: 2px solid #333; border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            font-weight: bold; font-size: 20px; text-align: center;">
            25. Tzuyu<br>(TWICE)
        </div>
    </div>
</div>
"""

st.markdown(carousel, unsafe_allow_html=True)

st.write(f"Page {st.session_state.page + 1} of {total_pages}")

st.divider()


uploaded_image = st.file_uploader("Upload an image of a Korean Idol or Yourself (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"],
                                         accept_multiple_files=False)


# @st.cache_resource # dont cache cuz every GradCAM will have different image input
def get_grad_cam():
    return GradCAM(model)

if uploaded_image is None:
    st.markdown("""
    <div class="custom-markdown-class">
        <i>Try uploading an image of a Korean Idol or Yourself to see how similar you are to a female Korean Idol!</i>
    </div>
    """, unsafe_allow_html=True)
elif uploaded_image is not None:
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)



    grad_cam = get_grad_cam()
    extracted_face= extract_faces(img_rgb)
    if extracted_face is None:
        st.text("No face found in the image")
    elif extracted_face is not None:
        # I have already called preprocess_image in apply_grad_cam (which is called inside visualize_gradCAM_results)
        is_kpop_idol, cam, cam_heatmap, overlaid_image = visualize_gradCAM_results(img_rgb, extracted_face, model, grad_cam)

        if is_kpop_idol:
            st.divider()
            # text_explanation = generate_textual_explanation(cam, retinaface_landmarks, overlaid_image)
            top_score, sorted_regions, regions_text = generate_textual_explanation_using_mediapipe_landmarks(cam, extracted_face, overlaid_image)

            # dataframe for sorted_regions
            df_scores = pd.DataFrame(sorted_regions, columns=["Region", "Score"])
            styled_df = df_scores.style.background_gradient(cmap="Blues")
            st.dataframe(styled_df, width="stretch", height="auto", hide_index=True)
            st.caption("*Note: Score indicates how much the model focuses for each region*")

            if top_score < 0.2: # weak focus
                st.markdown(f"""
                <div class="custom-markdown-class">
                    The model distributed attention across the whole face.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="custom-markdown-class">
                    The model focused mostly on the 
                    <span style="color: orange;">{regions_text}</span>
                    when identifying this idol!
                </div>
                """, unsafe_allow_html=True)

        elif not is_kpop_idol:
            idol_embeddings = np.load("idol_embeddings.npy")
            idol_labels = np.load("idol_labels.npy", allow_pickle=True)  # if labels are strings
            query_embeddings = get_embedding(extracted_face)

            # visualize embeddings
            best_idx, similarity_score = visualize_embeddings(idol_embeddings, query_embeddings, idol_labels)

            st.caption(
                "Note: The 2D t-SNE plot is only a simplified view of the 512-dimensional embedding space."
                "Points that look far apart here may still be very close in the original high-dimensional space,"
                "so global distances(distance between each cluster) in this plot may not always reflect the true similarity."
            )

            st.divider()

            predicted_label = idol_labels[best_idx]
            similarity_score_percent = str(np.round(similarity_score, decimals=4)*100)[:5]
            # print("Predicted idol:", predicted_label)
            # print("Similarity:", similarity_score)

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
            capitalized_predicted_label = predicted_label.capitalize()
            visualize_similar_images(uploaded_image, similar_image, capitalized_predicted_label)

            st.markdown(f"""
            <div class="custom-markdown-class">
                Your submitted image is probably not in the list of Kpop Idols. 
                This person looks the most similar to 
                <span style="color: orange;">{capitalized_predicted_label}</span>, 
                with similarity score of 
                <span style="color: orange;">{similarity_score_percent}%</span>!
            </div>
            """, unsafe_allow_html=True)






