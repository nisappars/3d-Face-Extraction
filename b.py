import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)


# Function to estimate age and gender based on facial landmarks
def estimate_demographics(face_landmarks):
    """
    Simple demographic estimation based on facial landmark characteristics
    Note: This is a VERY rough estimation and should not be considered accurate
    """
    # Convert landmarks to numpy array for easier manipulation
    landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])

    # Calculate some basic measurements
    face_width = np.max(landmarks_array[:, 0]) - np.min(landmarks_array[:, 0])
    face_height = np.max(landmarks_array[:, 1]) - np.min(landmarks_array[:, 1])

    # Very basic gender estimation based on facial structure
    # This is EXTREMELY simplistic and NOT scientifically accurate
    gender = "Male" if face_width > face_height * 1.2 else "Female"

    # Very basic age range estimation
    # Again, this is EXTREMELY simplistic
    landmarks_count = len(landmarks_array)
    if landmarks_count < 400:
        age_range = "Child (0-12)"
    elif 400 <= landmarks_count < 450:
        age_range = "Teenager (13-19)"
    elif 450 <= landmarks_count < 470:
        age_range = "Young Adult (20-35)"
    else:
        age_range = "Adult (36+)"

    return gender, age_range


# Function to extract 3D facial landmarks
def get_face_landmarks(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    if results.multi_face_landmarks:
        return np.array([[lm.x, lm.y, lm.z] for lm in results.multi_face_landmarks[0].landmark]), \
            results.multi_face_landmarks[0]
    return None, None


# Function to extract the face using the landmarks
def extract_face(image, landmarks):
    # Get the bounding box coordinates based on landmarks
    face_landmarks = np.array([[lm.x, lm.y] for lm in landmarks.landmark])
    x_min = np.min(face_landmarks[:, 0]) * image.shape[1]
    x_max = np.max(face_landmarks[:, 0]) * image.shape[1]
    y_min = np.min(face_landmarks[:, 1]) * image.shape[0]
    y_max = np.max(face_landmarks[:, 1]) * image.shape[0]

    # Add some padding
    padding = 0.1
    width = x_max - x_min
    height = y_max - y_min
    x_min = max(0, int(x_min - width * padding))
    x_max = min(image.shape[1], int(x_max + width * padding))
    y_min = max(0, int(y_min - height * padding))
    y_max = min(image.shape[0], int(y_max + height * padding))

    # Crop the face from the image
    face_image = image[int(y_min):int(y_max), int(x_min):int(x_max)]
    return face_image


# New functions for advanced landmark analysis
def calculate_landmark_distances(landmarks):
    """
    Calculate pairwise distances between facial landmarks
    """
    # Compute pairwise Euclidean distances
    distances = pdist(landmarks[:, :2])  # Use 2D coordinates
    distance_matrix = squareform(distances)
    return distance_matrix


def create_distance_distribution_plots(landmarks):
    """
    Create distribution plots for landmark distances
    """
    distance_matrix = calculate_landmark_distances(landmarks)

    # Flatten the upper triangle of distance matrix (excluding diagonal)
    distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]

    # Create distribution plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Histogram
    sns.histplot(distances, kde=True, ax=ax1)
    ax1.set_title('Distribution of Landmark Distances')
    ax1.set_xlabel('Distance')
    ax1.set_ylabel('Frequency')

    # Box plot
    sns.boxplot(x=distances, ax=ax2)
    ax2.set_title('Boxplot of Landmark Distances')
    ax2.set_xlabel('Distance')

    plt.tight_layout()
    st.pyplot(fig)


def create_landmark_heatmap(landmarks):
    """
    Create a heatmap of landmark distances
    """
    distance_matrix = calculate_landmark_distances(landmarks)

    plt.figure(figsize=(10, 8))
    sns.heatmap(distance_matrix, cmap='YlGnBu',
                xticklabels=False,
                yticklabels=False,
                annot=False)
    plt.title('Facial Landmark Distance Heatmap')
    plt.tight_layout()
    st.pyplot(plt)


def create_landmark_clustering_plot(landmarks):
    """
    Create a clustering visualization of landmarks
    """
    # Reduce dimensionality for visualization
    pca = PCA(n_components=2)
    landmarks_2d = pca.fit_transform(landmarks[:, :2])

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(landmarks_2d[:, 0], landmarks_2d[:, 1],
                          c=landmarks[:, 2], cmap='viridis')
    plt.colorbar(scatter, label='Depth')
    plt.title('Landmark Clustering (PCA)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.tight_layout()
    st.pyplot(plt)


def calculate_facial_symmetry(landmarks):
    """
    Calculate facial symmetry metrics
    """
    # Split landmarks into left and right sides
    mid_point = np.median(landmarks[:, 0])
    left_landmarks = landmarks[landmarks[:, 0] < mid_point]
    right_landmarks = landmarks[landmarks[:, 0] >= mid_point]

    # Calculate average distances for left and right sides
    left_distances = pdist(left_landmarks[:, :2])
    right_distances = pdist(right_landmarks[:, :2])

    # Compute symmetry score (lower is more symmetric)
    symmetry_score = np.abs(np.mean(left_distances) - np.mean(right_distances))

    return symmetry_score


# Streamlit configuration
st.set_page_config(layout="wide")

# Streamlit UI
st.title("Advanced Facial Landmark Analysis")

# File upload for the image
uploaded_image = st.file_uploader("Upload Image with Face", type=["jpg", "png", "jpeg"])

if uploaded_image:
    # Load the uploaded image
    image = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Extract and store face landmarks from the uploaded image
    face_landmarks, face_landmarks_obj = get_face_landmarks(image)

    if face_landmarks is not None:
        # Extract face
        extracted_face = extract_face(image, face_landmarks_obj)

        # Estimate demographics
        gender, age_range = estimate_demographics(face_landmarks_obj)

        # Display results in columns
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with col2:
            st.subheader("Extracted Face")
            st.image(extracted_face, caption="Extracted Face", use_container_width=True)

        # Demographics Estimation
        st.subheader("Facial Analysis (Estimated)")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Estimated Gender", gender)
        with col2:
            st.metric("Estimated Age Range", age_range)
        with col3:
            # Calculate and display symmetry score
            symmetry_score = calculate_facial_symmetry(face_landmarks)
            st.metric("Facial Symmetry Score", f"{symmetry_score:.4f}")

        # Warning about estimation
        st.warning(
            "⚠️ Note: These are VERY rough estimations based on facial landmarks and should not be considered accurate.")

        # 2D Landmark Visualization with Annotations
        st.subheader("2D Facial Landmarks Visualization")
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        landmark_scatter = ax.scatter(face_landmarks[:, 0] * image.shape[1],
                                      face_landmarks[:, 1] * image.shape[0],
                                      c=face_landmarks[:, 2], cmap='viridis', alpha=0.7,
                                      label="Facial Landmarks")
        ax.set_title("2D Facial Landmarks Overlay with Depth")
        plt.colorbar(landmark_scatter, label='Depth')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

        # 3D Landmark Visualization
        st.subheader("3D Facial Landmarks Visualization")
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(face_landmarks[:, 0],
                             face_landmarks[:, 1],
                             face_landmarks[:, 2],
                             c=face_landmarks[:, 2],
                             cmap='plasma',
                             marker="o")
        ax.set_title("3D Facial Landmarks")
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        plt.colorbar(scatter, ax=ax, label='Depth (Z-axis)')
        plt.tight_layout()
        st.pyplot(fig)

        # Additional Landmark Analyses
        st.header("Advanced Landmark Analyses")

        # Landmark Distance Distribution
        st.subheader("Landmark Distance Distribution")
        create_distance_distribution_plots(face_landmarks)

        # Landmark Distance Heatmap
        st.subheader("Landmark Distance Heatmap")
        create_landmark_heatmap(face_landmarks)

        # Landmark Clustering Visualization
        st.subheader("Landmark Clustering Visualization")
        create_landmark_clustering_plot(face_landmarks)

    else:
        st.write("No face detected in the uploaded image.")
else:
    st.write("Please upload an image to analyze facial landmarks.")