# Korean Idol Face Recognition with Explanation
This project focuses on building a **Korean Idol Face Recognition 
web application** using deep learning's Convolutional Neural Network(CNN) and have 
explanation using Explainable Artificial Intelligence(XAI)

The list of Korean Idols covered is based on the **Top 25 best female 
dancers in KPOP** from 
[ranker.com](https://www.ranker.com/list/best-kpop-female-dancers-right-now/ranker-music) (as of 31/8/25)

---

## 🧠 Model
- **Framework:** PyTorch
- **Base Model:** InceptionResNetV1 pretrained on VGGFace2  
- **Dataset:** 25 Korean idols (~50 images per idol, augmented)  
- **Face Detection & Alignment:** RetinaFace 
- **Facial Landmarks Extraction for Explainability:** MediaPipe FaceMesh  
- **Explainability:** Grad-CAM with textual explanation of key facial regions  

---

## 🌐 Web Application
- **Framework:** Streamlit  
- **Features:**
  - Upload an image of a Korean idol:
    - Get **predicted idol name**  
    - View **heatmap (Grad-CAM)** highlighting attention regions  
    - Receive a **textual explanation** of the model’s focus  
    - Display confidence scores across different face regions and show the top face region the model focuses on
  - If image is not a Korean Idol found in the list: 
    - Display the **most similar idol** with similarity score  
    - Display **t-SNE plot** of similarity scores

---

The web app is deployed in Streamlit and can be found in the link below:
<br>
[Korean Idol Face Recognition with Explanation](https://korean-idol-face-rec.streamlit.app/)




