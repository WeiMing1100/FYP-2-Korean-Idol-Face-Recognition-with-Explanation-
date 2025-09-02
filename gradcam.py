import torch
import numpy as np
import cv2
from utils import preprocess_image
import matplotlib.pyplot as plt
from utils import device
import streamlit as st
import mediapipe as mp

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.model.eval()

    def save_gradient(self, grad):
        self.gradients = grad

    def forward(self, x):
        self.model.zero_grad()
        x.requires_grad_()
        out = self.model(x)
        return out

    def generate_cam(self, image_tensor, target_class):
        output = self.forward(image_tensor)

        self.model.zero_grad()
        one_hot_output = torch.zeros((1, output.size()[-1]), dtype=torch.float).to(device)
        one_hot_output[0][target_class] = 1
        output.backward(gradient=one_hot_output)

        gradients = self.gradients.detach().cpu().numpy()
        feature_maps = self.model.feature_maps.detach().cpu().numpy()

        cam_weights = np.mean(gradients, axis=(2, 3))[0, :]
        cam = np.zeros(feature_maps.shape[2:], dtype=np.float32)

        for i, weight in enumerate(cam_weights):
            cam += weight * feature_maps[0, i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (160, 160))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        return cam


def register_hooks(model, grad_cam):
    def forward_hook(module, input, output):
        # print("Forward hook triggered!")
        grad_cam.model.feature_maps = output

    def backward_hook(module, grad_input, grad_output):
        # print("Backward hook triggered!")
        grad_cam.save_gradient(grad_output[0])

    target_module = None

    # finding the last conv2d layer, in this case, it's model.block8.conv2d
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.conv.Conv2d):
            target_module = module

    target_module.register_forward_hook(forward_hook)
    target_module.register_full_backward_hook(backward_hook)


def apply_grad_cam(image_input, model, grad_cam):
    # Preprocess image (works with file path or NumPy array)
    image_tensor = preprocess_image(image_input).to(device)
    image_tensor.requires_grad_()

    # Register hooks for Grad-CAM
    register_hooks(model, grad_cam)

    # Forward pass
    output = model(image_tensor)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    top_prob, top_class = probabilities.topk(1, dim=1)

    # Generate Grad-CAM
    cam = grad_cam.generate_cam(image_tensor, top_class.item())

    # Prepare original image for overlay
    if isinstance(image_input, str):
        original_image = cv2.imread(image_input)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    else:
        # NumPy array (BGR or RGB)
        if image_input.shape[2] == 3:  # assume BGR from OpenCV
            original_image = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        else:
            original_image = image_input

    original_image = cv2.resize(original_image, (160, 160))

    # Create heatmap and overlay
    cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
    overlaid_image = cv2.addWeighted(original_image, 0.5, cam_heatmap, 0.5, 0)

    return top_class.item(), top_prob.item(), cam, cam_heatmap, overlaid_image


def visualize_gradCAM_results(original_image, image_input, model, grad_cam):
    custom_labels = ['an yujin', 'chaeryeong', 'hani', 'heejin', 'irene', 'kwon eunbi',
                     'lee chaeyeon', 'lisa', 'mina', 'momo', 'moonbyul', 'nayeon', 'rose',
                     'ryujin', 'seulgi', 'sinB', 'soojin', 'soyeon', 'tzuyu', 'wheein',
                     'yeji', 'yena', 'yuna', 'yuqi', 'yves']

    top_class, top_prob, cam, cam_heatmap, overlaid_image = apply_grad_cam(image_input, model, grad_cam)
    class_label = custom_labels[top_class]

    if top_prob <= 0.55: # TEST THIS THRESHOLD
        return False, cam, cam_heatmap, overlaid_image
    elif top_prob > 0.55:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        ax[0].imshow(original_image)
        ax[0].axis('off')
        ax[0].set_title('Original Image')

        ax[1].imshow(cam_heatmap)
        ax[1].axis('off')
        ax[1].set_title('Grad-CAM Heatmap')

        ax[2].imshow(overlaid_image)
        # if aligned_landmarks is not None:
        #     for name, (x, y) in aligned_landmarks.items():
        #         ax[2].scatter(x, y, c="cyan", s=30, edgecolors="black")
        #         ax[2].text(x+2, y-2, name, color="white", fontsize=8)
        ax[2].axis('off')
        ax[2].set_title(f'Overlaid Image (Class: {class_label}, Prob: {top_prob:.4f})')

        st.pyplot(fig)
    return True, cam, cam_heatmap, overlaid_image


def generate_textual_explanation_using_mediapipe_landmarks(cam, aligned_face_rgb, overlaid_image, dst_size=160):
    # Use mediapipe to detect facial landmarks
    mp_face_mesh = mp.solutions.face_mesh
    face_rgb = aligned_face_rgb.copy()

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        results = face_mesh.process(face_rgb)

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]

        h, w, _ = face_rgb.shape
        landmarks = {}
        for i, lm in enumerate(face_landmarks.landmark):
            x, y = int(lm.x * w), int(lm.y * h)
            landmarks[i] = (x, y)

        # Map key points for textual explanation
        key_landmarks = {
            "left_eye": landmarks[33],
            "right_eye": landmarks[263],
            "nose": landmarks[1],
            "mouth_left": landmarks[61],
            "mouth_right": landmarks[291]
        }

        scale_x = dst_size / w
        scale_y = dst_size / h
        scaled_landmarks = {k: (int(x * scale_x), int(y * scale_y)) for k, (x, y) in key_landmarks.items()}

        # Compute region scores from Grad-CAM ---
        region_scores = {}
        for name, (x, y) in scaled_landmarks.items():
            patch = cam[max(0, y - 7):y + 7, max(0, x - 7):x + 7]
            region_scores[name] = patch.max() if patch.size > 0 else 0

        sorted_regions = sorted(region_scores.items(), key=lambda x: x[1], reverse=True)
        top_region, top_score = sorted_regions[0]

        st.write(f"Sorted Region: {sorted_regions}")
        st.write(f"Top Region: {top_region}")
        st.write(f"Top Score: {top_score}")

        # plot figures
        fig, ax = plt.subplots(1, 2, figsize=(6, 6))
        ax[0].imshow(cv2.resize(aligned_face_rgb, (dst_size, dst_size)))
        for name, (x, y) in scaled_landmarks.items():
            ax[0].scatter(x, y, c="cyan", s=30, edgecolors="black")
            ax[0].text(x, y-3, name, color="white", fontsize=6, ha="center", va="bottom")
        ax[0].axis("off")
        ax[0].set_title("Face with Key Landmarks", fontsize=10)

        ax[1].imshow(cv2.resize(overlaid_image, (dst_size, dst_size)))
        for name, (x, y) in scaled_landmarks.items():
            ax[1].scatter(x, y, c="cyan", s=30, edgecolors="black")
            ax[1].text(x, y-3, name, color="white", fontsize=6, ha="center", va="bottom")
        ax[1].axis("off")
        ax[1].set_title("Overlaid Face with Key Landmarks", fontsize=10)

        st.pyplot(fig)

        # Generate textual explanation
        if top_score < 0.2:  # weak focus
            return "The model distributed attention across the whole face."
        else:
            region_map = {
                "left_eye": "left eye",
                "right_eye": "right eye",
                "nose": "nose region",
                "mouth_left": "mouth (left side)",
                "mouth_right": "mouth (right side)"
            }
            return f"The model focused mostly on the {region_map[top_region]} when identifying this idol."


# def generate_textual_explanation_using_retinaface_landmarks(cam, retinaface_landmarks, overlaid_image, src_size=256, dst_size=160):
#     scale_x = dst_size / src_size
#     scale_y = dst_size / src_size  # square, so same
#     scaled_landmarks = {
#         key: (int(x * scale_x), int(y * scale_y))
#         for key, (x, y) in retinaface_landmarks.items()
#     }
#
#     region_scores = {}
#     for name, coordinates in scaled_landmarks.items():
#         x, y = int(coordinates[0]), int(coordinates[1])
#         patch = cam[max(0, y-7):y+7, max(0, x-7):x+7]
#         region_scores[name] = patch.max() if patch.size > 0 else 0
#
#     sorted_regions = sorted(region_scores.items(), key=lambda x: x[1], reverse=True)
#     top_region, top_score = sorted_regions[0]
#
#     st.write(f"Sorted Region: {sorted_regions}")
#     st.write(f"Top Region: {top_region}")
#     st.write(f"Top Score: {top_score}")
#
#     fig, ax = plt.subplots(1, 1, figsize=(15, 5))
#
#     ax.imshow(overlaid_image)
#     if scaled_landmarks is not None:
#         for name, (x, y) in scaled_landmarks.items():
#             ax.scatter(x, y, c="cyan", s=30, edgecolors="black")
#             ax.text(x+2, y-2, name, color="white", fontsize=8)
#     ax.axis('off')
#     ax.set_title('Overlaid Image with Facial Landmarks')
#
#     st.pyplot(fig)
#
#
#
#
#     if top_score < 0.2:  # threshold for weak focus
#         return "The model distributed attention across the whole face."
#     else:
#         region_map = {
#             "left_eye": "left eye",
#             "right_eye": "right eye",
#             "nose": "nose region",
#             "mouth_left": "mouth (left side)",
#             "mouth_right": "mouth (right side)"
#         }
#         return f"The model focused mostly on the {region_map[top_region]} when identifying this idol."