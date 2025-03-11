import os
import cv2
import numpy as np
import streamlit as st
import face_recognition
import dlib
import mediapipe as mp
import segno
import datetime
import base64
from typing import List, Dict, Optional
from PIL import Image
import hashlib


class AdvancedFaceVerification:
    def __init__(self, first_name: str, last_name: str):
        """
        Initialize advanced face verification system

        Args:
            first_name (str): User's first name
            last_name (str): User's last name
        """
        self.first_name = first_name
        self.last_name = last_name
        self.base_folder = r"C:\Users\tnish\PycharmProjects\NISHANTH\biometric_captures"

        # Initialize MediaPipe face mesh for detailed landmark detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )

        # Initialize dlib for additional face landmark detection
        self.dlib_detector = dlib.get_frontal_face_detector()
        self.dlib_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    def compress_and_encode_image(self, image_path: str, max_size_kb: int = 20) -> str:
        """
        Compress and encode image to base64 with size limitations

        Args:
            image_path (str): Path to the image to compress
            max_size_kb (int): Maximum allowed size in kilobytes

        Returns:
            Base64 encoded compressed image or empty string
        """
        try:
            # Read the image
            img = cv2.imread(image_path)

            # Initial compression attempts
            compression_levels = [50, 30, 10]  # JPEG quality levels
            scale_factors = [1.0, 0.75, 0.5, 0.25]

            for quality in compression_levels:
                for scale in scale_factors:
                    # Resize image
                    resized_img = cv2.resize(
                        img,
                        None,
                        fx=scale,
                        fy=scale,
                        interpolation=cv2.INTER_AREA
                    )

                    # Compression parameters
                    compression_params = [
                        cv2.IMWRITE_JPEG_QUALITY, quality,
                        cv2.IMWRITE_JPEG_PROGRESSIVE, 1
                    ]

                    # Encode image
                    success, encoded_img = cv2.imencode('.jpg', resized_img, compression_params)

                    if not success:
                        continue

                    # Check size
                    img_size_kb = len(encoded_img) / 1024

                    if img_size_kb <= max_size_kb:
                        # Convert to base64
                        return base64.b64encode(encoded_img).decode('utf-8')

            st.warning("Could not compress image to desired size")
            return ""

        except Exception as e:
            st.error(f"Image compression error: {e}")
            return ""

    def generate_verification_qr_code(self, match_score: float, test_image_path: str) -> Optional[str]:
        """
        Generate a comprehensive QR code with verification details and compressed image data

        Args:
            match_score (float): The confidence score of face verification
            test_image_path (str): Path to the test image

        Returns:
            Path to the generated QR code image or None if generation fails
        """
        try:
            # Get current timestamp
            verification_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Compress and encode the test image
            compressed_image = self.compress_and_encode_image(test_image_path)

            # Generate image hash for verification
            image_hash = self.get_image_signature(test_image_path)

            # Create comprehensive verification data
            verification_data = (
                f"VERIFICATION CERTIFICATE\n"
                f"Name: {self.first_name.upper()} {self.last_name.upper()}\n"
                f"Verification Score: {match_score * 100:.2f}%\n"
                f"Timestamp: {verification_timestamp}\n"
                f"Image Hash: {image_hash}\n"
                f"Status: {'VERIFIED' if match_score > 0.7 else 'PARTIALLY VERIFIED'}"
            )

            # Generate QR code
            qr = segno.make(verification_data)

            # Save the QR code
            qr_code_path = os.path.join(
                self.base_folder,
                f"Verification_{self.first_name}_{self.last_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )

            # Save with specific parameters
            qr.save(
                qr_code_path,
                scale=10,
                dark='darkblue',
                light='white'
            )

            # Optional: Log verification details
            with open(qr_code_path + "_verification_log.txt", "w") as f:
                f.write(f"Test Image Path: {test_image_path}\n")
                f.write(f"Image Signature: {image_hash}\n")
                if compressed_image:
                    f.write(f"Compressed Image Preview: {compressed_image[:50]}...\n")

            return qr_code_path

        except Exception as e:
            st.error(f"QR Code generation error: {e}")
            return None

    def get_image_signature(self, image_path: str) -> str:
        """
        Generate a secure hash of the image

        Args:
            image_path (str): Path to the image

        Returns:
            SHA-256 hash of the image
        """
        try:
            with open(image_path, "rb") as f:
                image_hash = hashlib.sha256(f.read()).hexdigest()
            return image_hash
        except Exception as e:
            st.error(f"Image hash generation error: {e}")
            return ""
    def get_reference_images(self) -> List[str]:
        """
        Retrieve reference images for the specific user

        Returns:
            List of paths to reference images
        """
        # Pattern for matching user-specific images
        image_patterns = [
            f"{self.first_name}_{self.last_name}_straight_posture.jpg",
            f"{self.first_name}_{self.last_name}_left_turn_posture.jpg",
            f"{self.first_name}_{self.last_name}_right_turn_posture.jpg",
            f"{self.first_name}_{self.last_name}_up_tilt_posture.jpg",
            f"{self.first_name}_{self.last_name}_down_tilt_posture.jpg",
            f"{self.first_name}_{self.last_name}passport_photo.jpg",
            f"{self.first_name}_{self.last_name}extractedface.jpg"
        ]

        # Print all potential image paths for debugging
        print("Searching for images in:", self.base_folder)
        print("Potential image patterns:", image_patterns)

        reference_images = [
            os.path.join(self.base_folder, img)
            for img in image_patterns
            if os.path.exists(os.path.join(self.base_folder, img))
        ]

        print("Found reference images:", reference_images)

        if not reference_images:
            st.warning(f"No reference images found for {self.first_name} {self.last_name}")

        return reference_images

    def get_reference_landmarks(self) -> List[np.ndarray]:
        """
        Retrieve pre-computed landmark files for the user

        Returns:
            List of numpy arrays containing landmarks
        """
        landmark_patterns = [
            f"{self.first_name}_{self.last_name}_straight_landmarks.npy",
            f"{self.first_name}_{self.last_name}_left_turn_landmarks.npy",
            f"{self.first_name}_{self.last_name}_right_turn_landmarks.npy",
            f"{self.first_name}_{self.last_name}_up_tilt_landmarks.npy",
            f"{self.first_name}_{self.last_name}_down_tilt_landmarks.npy"
        ]

        reference_landmarks = []
        for landmark_file in landmark_patterns:
            landmark_path = os.path.join(self.base_folder, landmark_file)
            print(f"Checking landmark file: {landmark_path}")
            if os.path.exists(landmark_path):
                try:
                    # Use allow_pickle=True to load object arrays
                    landmark = np.load(landmark_path, allow_pickle=True)
                    reference_landmarks.append(landmark)
                    print(f"Successfully loaded {landmark_file}")
                except Exception as e:
                    print(f"Error loading landmark file {landmark_file}: {e}")
                    st.warning(f"Error loading landmark file {landmark_file}: {e}")

        print(f"Total reference landmarks loaded: {len(reference_landmarks)}")
        return reference_landmarks

    def capture_test_image(self) -> Optional[str]:
        """
        Capture a test image using the camera

        Returns:
            Path to the captured test image or None if capture fails
        """
        # Open camera
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Unable to open camera")
            return None

        # Capture frame
        ret, frame = cap.read()
        cap.release()

        if not ret:
            st.error("Failed to capture image")
            return None

        # Save the captured image
        test_image_path = os.path.join(self.base_folder, f"{self.first_name}_{self.last_name}_test_image.jpg")
        cv2.imwrite(test_image_path, frame)

        return test_image_path

    def extract_advanced_face_landmarks(self, image_path: str) -> Dict:
        """
        Extract advanced facial landmarks using multiple techniques

        Args:
            image_path (str): Path to the input image

        Returns:
            Dict containing various landmark features
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            st.error(f"Failed to read image: {image_path}")
            return {}

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # MediaPipe face mesh landmarks
        results = self.face_mesh.process(image_rgb)

        # Convert to grayscale for dlib
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.dlib_detector(gray)

        landmarks_data = {
            'mediapipe_landmarks': [],
            'dlib_landmarks': [],
            'face_recognition_encoding': None,
            'additional_features': {}
        }

        # Extract MediaPipe landmarks
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks_data['mediapipe_landmarks'] = [
                    (int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]))
                    for landmark in face_landmarks.landmark
                ]

        # Extract Dlib landmarks
        if faces:
            shape = self.dlib_predictor(gray, faces[0])
            landmarks_data['dlib_landmarks'] = [
                (shape.part(n).x, shape.part(n).y) for n in range(68)
            ]

        # Face recognition encoding
        try:
            face_encodings = face_recognition.face_encodings(
                face_recognition.load_image_file(image_path)
            )
            if face_encodings:
                landmarks_data['face_recognition_encoding'] = face_encodings[0]
        except Exception as e:
            st.error(f"Error extracting face encoding: {e}")

        return landmarks_data

    def multi_view_verification(self, reference_images: List[str], test_image: str,
                                reference_landmarks: List[np.ndarray]) -> Dict:
        # Stricter verification parameters
        def strict_face_verification(encoding1, encoding2, tolerance=0.45):
            # Lower tolerance for stricter matching
            distance = face_recognition.face_distance([encoding1], encoding2)[0]
            return distance < tolerance

        verification_results = {
            'overall_match_score': 0.0,
            'match_details': [],
            'strict_match': False
        }

        # More rigorous verification checks
        encoding_matches = []
        landmark_consistencies = []

        for ref_image in reference_images:
            ref_landmark = self.extract_advanced_face_landmarks(ref_image)
            test_landmark = self.extract_advanced_face_landmarks(test_image)

            # Strict encoding verification
            if (ref_landmark.get('face_recognition_encoding') is not None and
                    test_landmark.get('face_recognition_encoding') is not None):
                strict_match = strict_face_verification(
                    ref_landmark['face_recognition_encoding'],
                    test_landmark['face_recognition_encoding']
                )

                landmark_similarity = self._advanced_landmark_comparison(
                    ref_landmark.get('mediapipe_landmarks', []),
                    test_landmark.get('mediapipe_landmarks', [])
                )

                encoding_matches.append(strict_match)
                landmark_consistencies.append(landmark_similarity)

        # Enhanced match scoring with stricter criteria
        verification_results['strict_match'] = (
                np.mean(encoding_matches) > 0.8 and
                np.mean(landmark_consistencies) > 0.7
        )

        verification_results['overall_match_score'] = (
                np.mean(encoding_matches) * 0.6 +
                np.mean(landmark_consistencies) * 0.4
        )

        return verification_results

    def _advanced_landmark_comparison(self, landmarks1, landmarks2):
        # More sophisticated landmark comparison
        def normalize_landmarks(landmarks):
            if not landmarks:
                return []

            x_coords = [point[0] for point in landmarks]
            y_coords = [point[1] for point in landmarks]

            center_x, center_y = np.mean(x_coords), np.mean(y_coords)
            scale = max(np.std(x_coords), np.std(y_coords))

            return [
                ((x - center_x) / scale, (y - center_y) / scale)
                for x, y in landmarks
            ]

        norm_landmarks1 = normalize_landmarks(landmarks1)
        norm_landmarks2 = normalize_landmarks(landmarks2)

        if len(norm_landmarks1) != len(norm_landmarks2):
            return 0.0

        # Advanced distance metric with geometric constraints
        distances = [
            np.sqrt((l1[0] - l2[0]) * 2 + (l1[1] - l2[1]) * 2)
            for l1, l2 in zip(norm_landmarks1, norm_landmarks2)
        ]

        # Strict similarity calculation
        similarity = 1 - np.mean(distances)
        return max(0, min(1, similarity))

    def _compare_landmarks(self, landmarks1, landmarks2) -> float:
        """
        Enhanced landmark comparison with more robust similarity calculation
        """
        if not landmarks1 or not landmarks2:
            return 0.0

        # Normalize landmark sizes
        def normalize_landmarks(landmarks):
            x_coords = [point[0] for point in landmarks]
            y_coords = [point[1] for point in landmarks]

            # Normalize to center and scale
            center_x, center_y = np.mean(x_coords), np.mean(y_coords)
            scale = np.std(x_coords) + np.std(y_coords)

            normalized = [
                ((x - center_x) / scale, (y - center_y) / scale)
                for x, y in landmarks
            ]
            return normalized

        # Normalize both landmark sets
        norm_landmarks1 = normalize_landmarks(landmarks1)
        norm_landmarks2 = normalize_landmarks(landmarks2)

        # More robust distance calculation
        max_landmarks = min(len(norm_landmarks1), len(norm_landmarks2))
        distances = [
            np.sqrt((l1[0] - l2[0]) * 2 + (l1[1] - l2[1]) * 2)
            for l1, l2 in zip(norm_landmarks1[:max_landmarks], norm_landmarks2[:max_landmarks])
        ]

        # Use multiple similarity metrics
        mean_distance = np.mean(distances)
        similarity = 1 / (1 + mean_distance)

        return max(0, min(1, similarity))

    def _calculate_match_score(self, verification_results: Dict) -> float:
        """
        Enhanced match score calculation with more robust weighting
        """
        if not verification_results['match_details']:
            return 0.0

        # Extract metrics
        encoding_matches = [
            1.0 if detail['encoding_match'] else 0.0
            for detail in verification_results['match_details']
        ]
        face_distances = [
            1 - detail['face_distance']
            for detail in verification_results['match_details']
        ]
        landmark_similarities = [
            detail['landmark_similarity']
            for detail in verification_results['match_details']
        ]

        # Weighted combination with more balanced approach
        weights = [0.4, 0.3, 0.3]  # Encoding, Face Distance, Landmark Similarity
        weighted_scores = [
            np.mean(encoding_matches) * weights[0],
            np.mean(face_distances) * weights[1],
            np.mean(landmark_similarities) * weights[2]
        ]

        overall_score = np.sum(weighted_scores)
        return max(0, min(1, overall_score))
def main():
    st.title("Advanced Multi-View Face Verification")

    # User input for first and last name
    col1, col2 = st.columns(2)
    with col1:
        first_name = st.text_input("Enter First Name")
    with col2:
        last_name = st.text_input("Enter Last Name")

    if not first_name or not last_name:
        st.error("Please enter both First Name and Last Name")
        return

    # Initialize verification system
    verifier = AdvancedFaceVerification(first_name, last_name)

    # Get reference images from the reference folder
    reference_images = verifier.get_reference_images()
    reference_landmarks = verifier.get_reference_landmarks()

    if not reference_images:
        st.error(f"Please add reference images for {first_name} {last_name}")
        return

    # Display number of reference images
    st.write(f"Number of reference images found: {len(reference_images)}")
    st.write(f"Number of reference landmark files: {len(reference_landmarks)}")

    # Capture test image button
    if st.button("Capture Test Image and Verify"):
        # Capture test image
        test_image_path = verifier.capture_test_image()

        if test_image_path:
            # Display captured test image
            test_image = cv2.imread(test_image_path)
            st.image(test_image, channels="BGR", use_container_width=True)

            # Perform verification
            try:
                results = verifier.multi_view_verification(
                    reference_images,
                    test_image_path,
                    reference_landmarks
                )

                # Display results
                st.write("Verification Results:")
                st.write(f"Overall Match Score: {results['overall_match_score']:.4f}")

                # Confidence interpretation with QR code generation
                if results['overall_match_score'] > 0.7:
                    st.success("High Confidence Match ✅")

                    # Generate and display QR code with the match score and compressed test image
                    qr_code_path = verifier.generate_verification_qr_code(
                        results['overall_match_score'],
                        test_image_path
                    )
                    if qr_code_path:
                        st.write("Verification QR Code:")
                        st.image(qr_code_path)
                        st.write(f"QR Code saved at: {qr_code_path}")

                elif results['overall_match_score'] > 0.5:
                    st.warning("Moderate Confidence Match ⚠️")
                    # Optional: Generate QR code for partial verification with compressed test image
                    qr_code_path = verifier.generate_verification_qr_code(
                        results['overall_match_score'],
                        test_image_path
                    )
                else:
                    st.error("Low Confidence Match ❌")

            except Exception as e:
                st.error(f"Verification failed: {e}")

if __name__ == "__main__":
    main()