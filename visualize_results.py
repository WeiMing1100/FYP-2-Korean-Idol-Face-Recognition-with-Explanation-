import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from adjustText import adjust_text
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import streamlit as st
from threading import RLock


_lock = RLock()


def visualize_embeddings(idol_embeddings, query_embeddings, idol_labels):
    # compare cosine similarity
    sims = cosine_similarity([query_embeddings], idol_embeddings)
    best_idx = np.argmax(sims)  # index of most similar embedding

    # reduce embeddings to 2D for visualization
    all_embeddings = np.vstack([idol_embeddings, query_embeddings])  # include query
    emb_2d = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(all_embeddings)

    # split back
    idol_emb_2d = emb_2d[:-1]
    query_emb_2d = emb_2d[-1]

    # find top-k similar idols
    k = 5
    top_k_idx = np.argsort(-sims[0])[:k]

    with _lock:
        fig, ax = plt.subplots(figsize=(12, 9))

        # plot all idols (grey)
        sns.scatterplot(x=idol_emb_2d[:,0], y=idol_emb_2d[:,1],
                        color="lightgrey", s=40, alpha=0.5, label="Other Idols")

        # highlight top-k similar idols
        sns.scatterplot(x=idol_emb_2d[top_k_idx,0], y=idol_emb_2d[top_k_idx,1],
                        color="blue", s=80, label="Top-K Similar Idols")

        # highlight the best match
        sns.scatterplot(x=[idol_emb_2d[best_idx,0]], y=[idol_emb_2d[best_idx,1]],
                        color="red", s=150, label=f"Best Match: {idol_labels[best_idx]}")

        # plot query embedding
        sns.scatterplot(x=[query_emb_2d[0]], y=[query_emb_2d[1]],
                        color="green", s=80, marker="X", label="Query Image")

        # annotate names for top-k idols
        texts = []
        for idx in top_k_idx:
            texts.append(
                ax.text(idol_emb_2d[idx,0], idol_emb_2d[idx,1], idol_labels[idx],
                         fontsize=10, color="blue")
            )

        # Automatically adjust positions to reduce overlap
        adjust_text(texts, arrowprops=dict(arrowstyle="->", color='blue'))


        ax.set_title(f"Top-{k} Similar Idols")
        ax.legend()
        st.pyplot(fig)

    similarity_score = sims[0, best_idx].item()

    return best_idx, similarity_score


def visualize_similar_images(image_input, similar_idol_image, similar_idol, probability):
    with _lock:
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))

        ax[0].imshow(Image.open(image_input))
        ax[0].axis('off')
        ax[0].set_title('Original Image')

        ax[1].imshow(Image.open(similar_idol_image))
        ax[1].axis('off')
        ax[1].set_title(f"Similar Idol: {similar_idol}")
        ax[1].text(0.5, -0.1, f"Your submitted image is probably not a Kpop Idol. This person looks the most similar to {similar_idol}, with similarity score of {probability*100}%!",
                   fontsize=14, ha='center', va='top', transform=ax[1].transAxes)

        st.pyplot(fig)