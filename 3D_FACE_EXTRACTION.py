import streamlit as st
import cv2
import numpy as np
import os
import uuid
import re
from deepface import DeepFace
import pandas as pd
import dlib
import scipy.spatial
import json
import math

class BiometricVerification:
    def __init__(self):
        # Ensure base directory path is absolute
        base_path = os.path.abspath('biometric_captures')
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

        # Define more relaxed postures with broader angle requirements
        self.postures = [
            {
                "name": "straight",
                "instruction": "Look directly at the camera",
                "angle_check": self.check_straight_face,
                "is_strict": False,  # Make this less strict
                "yaw_range": (-20, 20),  # increased from (-10, 10)
                "pitch_range": (-15, 15),  # increased from (-10, 10)
                "roll_range": (-15, 15)  # increased from (-10, 10)
            },
            {
                "name": "left_turn",
                "instruction": "Turn your head 30-45 degrees to the left",
                "angle_check": self.check_left_turn_face,
                "is_strict": False,  # Make less strict
                "yaw_range": (20, 55),  # broadened range
                "pitch_range": (-15, 15),  # more tolerance in vertical angle
                "roll_range": (-15, 15)  # more tolerance in side tilt
            },
            {
                "name": "right_turn",
                "instruction": "Turn your head 30-45 degrees to the right",
                "angle_check": self.check_right_turn_face,
                "is_strict": False,  # Make less strict
                "yaw_range": (-55, -20),  # broadened range
                "pitch_range": (-15, 15),  # more tolerance in vertical angle
                "roll_range": (-15, 15)  # more tolerance in side tilt
            },
            {
                "name": "up_tilt",
                "instruction": "Tilt your head up 15-20 degrees",
                "angle_check": self.check_up_tilt_face,
                "is_strict": False,  # Make less strict
                "yaw_range": (-20, 20),  # increased from (-10, 10)
                "pitch_range": (10, 25),  # broadened range
                "roll_range": (-15, 15)  # increased tolerance
            },
            {
                "name": "down_tilt",
                "instruction": "Tilt your head down 15-20 degrees",
                "angle_check": self.check_down_tilt_face,
                "is_strict": False,  # Make less strict
                "yaw_range": (-20, 20),  # increased from (-10, 10)
                "pitch_range": (-25, -10),  # broadened range
                "roll_range": (-15, 15)  # increased tolerance
            }
        ]

        # Load face detection and landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    def extract_face(self, image_path, output_path, size=(300, 300)):
        """
        Extract and save face from an image
        """
        # Read image
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.detector(gray)

        if len(faces) > 0:
            # Get the first face
            face = faces[0]

            # Get face coordinates
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()

            # Crop face
            face_img = img[y1:y2, x1:x2]

            # Resize face
            face_img_resized = cv2.resize(face_img, size)

            # Save extracted face
            cv2.imwrite(output_path, face_img_resized)
            return True
        return False

    def create_unique_filename(self, prefix, first_name='', last_name='', extension='.jpg'):
        """
        Create a unique filename within the base directory
        """
        # Ensure base path is used
        base_filename = f"{first_name}_{last_name}_{prefix}" if first_name and last_name else f"{prefix}_{uuid.uuid4()}"
        filename = f"{base_filename}{extension}"
        full_path = os.path.join(self.base_path, filename)
        return full_path
    def run_verification_process(self):
        # Initialize session state variables if not already set
        if 'stage' not in st.session_state:
            st.session_state.stage = 'initial_info'

        # Add other initial session state variables as needed
        if 'current_posture_index' not in st.session_state:
            st.session_state.current_posture_index = 0
        if 'captured_postures' not in st.session_state:
            st.session_state.captured_postures = {}
        if 'posture_landmarks' not in st.session_state:
            st.session_state.posture_landmarks = {}

        # Main routing logic based on current stage
        if st.session_state.stage == 'initial_info':
            self.initial_info_stage()
        elif st.session_state.stage == 'passport_photo':
            self.passport_photo_stage()
        elif st.session_state.stage == 'posture_capture':
            self.posture_capture_stage()
        elif st.session_state.stage == 'verification':
            self.verification_stage()

    def initial_info_stage(self):
        """
        Collect initial personal information
        """
        st.header("Personal Information")

        # Collect first name
        st.session_state.first_name = st.text_input("First Name")
        st.session_state.last_name = st.text_input("Last Name")

        if st.button("Next"):
            if st.session_state.first_name and st.session_state.last_name:
                st.session_state.stage = 'passport_photo'
                st.rerun()
            else:
                st.error("Please enter both first and last name")

    def calculate_head_pose_angles(self, landmarks):
        """
        Calculate head pose angles using facial landmarks
        Returns (yaw, pitch, roll) in degrees
        """
        if landmarks is None:
            return None

        # Key landmark points
        left_eye = np.mean(landmarks[36:42], axis=0)
        right_eye = np.mean(landmarks[42:48], axis=0)
        nose_tip = landmarks[30]
        left_mouth = landmarks[48]
        right_mouth = landmarks[54]

        # Yaw (horizontal rotation)
        eye_midpoint = (left_eye + right_eye) / 2
        yaw = np.degrees(np.arctan2(nose_tip[0] - eye_midpoint[0], eye_midpoint[1] - nose_tip[1]))

        # Pitch (vertical tilt)
        mouth_midpoint = (left_mouth + right_mouth) / 2
        pitch = np.degrees(np.arctan2(nose_tip[1] - mouth_midpoint[1], mouth_midpoint[0] - nose_tip[0]))

        # Roll (side tilt) - use eye line angle
        roll = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))

        return (yaw, pitch, roll)

    def is_angle_in_range(self, angle, angle_range):
        """
        Check if an angle is within a specified range
        """
        return angle_range[0] <= angle <= angle_range[1]
    def detect_face_landmarks(self, image_path):
        """
        Detect face landmarks using dlib
        Returns landmarks or None if detection fails
        """
        # Read image
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.detector(gray)

        if len(faces) == 0:
            return None

        # Get landmarks for the first face
        landmarks = self.predictor(gray, faces[0])

        # Convert to numpy array
        landmarks_np = np.array([(p.x, p.y) for p in landmarks.parts()])

        return landmarks_np

    def check_straight_face(self, landmarks):
        """
        Verify if face is relatively straight with more tolerance
        """
        angles = self.calculate_head_pose_angles(landmarks)
        if angles is None:
            return False

        yaw, pitch, roll = angles
        return (
                abs(yaw) <= 20 and
                abs(pitch) <= 15 and
                abs(roll) <= 15
        )

    def check_left_turn_face(self, landmarks):
        """
        Check for left turn with broader acceptance range
        """
        angles = self.calculate_head_pose_angles(landmarks)
        if angles is None:
            return False

        yaw, pitch, roll = angles
        return (
                20 <= yaw <= 55 and
                abs(pitch) <= 15 and
                abs(roll) <= 15
        )

    def check_right_turn_face(self, landmarks):
        """
        Check for right turn with broader acceptance range
        """
        angles = self.calculate_head_pose_angles(landmarks)
        if angles is None:
            return False

        yaw, pitch, roll = angles
        return (
                -55 <= yaw <= -20 and
                abs(pitch) <= 15 and
                abs(roll) <= 15
        )

    def check_up_tilt_face(self, landmarks):
        """
        Check for upward head tilt with more tolerance
        """
        angles = self.calculate_head_pose_angles(landmarks)
        if angles is None:
            return False

        yaw, pitch, roll = angles
        return (
                abs(yaw) <= 20 and
                10 <= pitch <= 25 and
                abs(roll) <= 15
        )

    def check_down_tilt_face(self, landmarks):
        """
        Check for downward head tilt with more tolerance
        """
        angles = self.calculate_head_pose_angles(landmarks)
        if angles is None:
            return False

        yaw, pitch, roll = angles
        return (
                abs(yaw) <= 20 and
                -25 <= pitch <= -10 and
                abs(roll) <= 15
        )
    def extract_demographic_info(self, img_path):
        """
        Safely extract demographic information
        """
        try:
            # Analyze demographics using DeepFace
            demographic_analysis = DeepFace.analyze(
                img_path=img_path,
                actions=['age', 'gender', 'race'],
                enforce_detection=True
            )

            # Handle potential list or dict result
            if isinstance(demographic_analysis, list):
                demographic_analysis = demographic_analysis[0]

            # Extract gender (most confident prediction)
            gender_dict = demographic_analysis.get('gender', {})
            if isinstance(gender_dict, dict):
                gender = max(gender_dict, key=gender_dict.get)
            else:
                gender = gender_dict

            return {
                'age': demographic_analysis.get('age', 'N/A'),
                'gender': gender,
                'dominant_race': demographic_analysis.get('dominant_race', 'N/A')
            }
        except Exception as e:
            st.warning(f"Demographic analysis error: {str(e)}")
            return {
                'age': 'N/A',
                'gender': 'N/A',
                'dominant_race': 'N/A'
            }


    def passport_photo_stage(self):
        """
        Upload passport photo stage with face extraction
        """
        st.header("Passport Photo Upload")

        uploaded_file = st.file_uploader("Upload your passport photo", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            # Save the uploaded file
            first_name = st.session_state.first_name
            last_name = st.session_state.last_name

            file_path = self.create_unique_filename(
                "passport_photo",
                first_name,
                last_name,
                extension=os.path.splitext(uploaded_file.name)[1]
            )

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Display the uploaded image
            st.image(uploaded_file, caption="Uploaded Passport Photo")

            # Extract and save face
            extracted_face_path = self.create_unique_filename(
                "extractedface",
                first_name,
                last_name,
                '.png'
            )

            face_extracted = self.extract_face(file_path, extracted_face_path)

            if face_extracted:
                st.image(extracted_face_path, caption="Extracted Face")
                st.session_state.passport_photo_path = file_path
                st.session_state.extracted_face_path = extracted_face_path

                if st.button("Confirm Photo"):
                    st.session_state.stage = 'posture_capture'
                    st.rerun()
            else:
                st.error("No face detected in the image. Please upload a clear passport photo.")

    def posture_capture_stage(self):
        """Enhanced posture capture with landmark-based posture validation"""
        # Get current posture
        current_index = st.session_state.current_posture_index
        current_posture = self.postures[current_index]
        first_name = st.session_state.first_name
        last_name = st.session_state.last_name

        st.header(f"Capture {current_posture['name'].replace('_', ' ').title()} Posture")
        st.write(current_posture['instruction'])

        try:
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                st.error("Unable to access webcam. Please check your camera connection.")
                return

            # Create placeholders
            frame_placeholder = st.empty()
            capture_button = st.empty()

            # Capture frame
            ret, frame = cap.read()
            if ret:
                # Convert color space for Streamlit display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Display frame
                frame_placeholder.image(frame_rgb, channels="RGB")

                # Capture button
                if capture_button.button(f"Capture {current_posture['name'].replace('_', ' ').title()} Posture"):
                    # Save frame
                    posture_path = self.create_unique_filename(
                        f"{current_posture['name']}_posture",
                        first_name,
                        last_name
                    )
                    cv2.imwrite(posture_path, frame)

                    try:
                        # Detect facial landmarks
                        landmarks = self.detect_face_landmarks(posture_path)

                        # Validate posture based on current posture's requirements
                        posture_valid = current_posture['angle_check'](landmarks)

                        if posture_valid or not current_posture['is_strict']:
                            # Store posture
                            st.session_state.captured_postures[current_posture['name']] = posture_path
                            st.session_state.posture_landmarks[current_posture['name']] = landmarks

                            # Move to next posture or verification
                            st.session_state.current_posture_index += 1

                            if st.session_state.current_posture_index >= len(self.postures):
                                st.session_state.stage = 'verification'

                            cap.release()
                            st.rerun()
                        else:
                            st.error(
                                f"Invalid {current_posture['name']} posture. Please follow the instructions carefully.")
                            os.remove(posture_path)

                    except Exception as e:
                        st.error(f"Posture capture failed: {e}. Adjust your position and try again.")
                        os.remove(posture_path)  # Remove invalid capture
            else:
                st.error("Failed to capture frame from webcam.")

            cap.release()

        except Exception as e:
            st.error(f"An error occurred during posture capture: {e}")

    def verification_stage(self):
        """Enhanced verification stage with comprehensive stats and demographics"""
        st.header("Verification Results")

        # Verification models
        models = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace']

        # Verification inputs
        passport_photo = st.session_state.passport_photo_path
        straight_posture = st.session_state.captured_postures['straight']

        # Personal info
        first_name = st.session_state.first_name
        last_name = st.session_state.last_name

        try:
            # Demographic Analysis with improved error handling
            st.subheader("Demographic Analysis")
            demographic_analysis = self.extract_demographic_info(passport_photo)

            # Create columns for demographic info
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Predicted Age", demographic_analysis['age'])

            with col2:
                st.metric("Predicted Gender", demographic_analysis['gender'])

            with col3:
                st.metric("Dominant Race", demographic_analysis['dominant_race'])

            # Verification Models Performance
            verification_results = {}
            verification_details = []
            verification_successful = False

            # Perform verification for each model
            for model in models:
                try:
                    result = DeepFace.verify(
                        img1_path=passport_photo,
                        img2_path=straight_posture,
                        model_name=model,
                        detector_backend='mtcnn'
                    )
                    verification_results[model] = result

                    # Prepare detailed results
                    verification_details.append({
                        "Model": model,
                        "Verified": result.get('verified', False),
                        "Distance": round(result.get('distance', 0), 4),
                        "Threshold": round(result.get('threshold', 0), 4)
                    })

                    # If any model verifies successfully
                    if result.get('verified', False):
                        verification_successful = True

                except Exception as inner_e:
                    verification_results[model] = {"error": str(inner_e)}
                    verification_details.append({
                        "Model": model,
                        "Verified": False,
                        "Distance": "Error",
                        "Threshold": "Error"
                    })

            # Create a styled dataframe for results
            results_df = pd.DataFrame(verification_details)

            # Styling the results
            st.subheader("Verification Models Performance")
            styled_results = results_df.style.apply(
                lambda x: ['background-color: green' if v else 'background-color: red'
                           for v in x == True],
                subset=['Verified']
            )
            st.dataframe(styled_results)

            # Advanced stats and file saving
            if verification_successful:
                # Combine demographic and verification data
                full_results = {
                    **demographic_analysis,
                    "verification_models": verification_details
                }

                # Save comprehensive results
                verification_stats_path = self.create_unique_filename(
                    "verification_stats",
                    first_name,
                    last_name,
                    '.json'
                )

                # Save results as JSON for more comprehensive storage
                import json
                with open(verification_stats_path, 'w') as f:
                    json.dump(full_results, f, indent=4)

                # Save landmark and posture files
                for posture_name, posture_path in st.session_state.captured_postures.items():
                    # Save landmarks
                    landmarks = st.session_state.posture_landmarks[posture_name]
                    landmarks_save_path = self.create_unique_filename(
                        f"{posture_name}_landmarks",
                        first_name,
                        last_name,
                        '.npy'
                    )
                    np.save(landmarks_save_path, landmarks)

                st.success("Verification successful! Biometric data saved.")
            else:
                # Clean up temporary files if verification fails
                for posture_path in st.session_state.captured_postures.values():
                    try:
                        os.remove(posture_path)
                    except Exception:
                        pass

                st.error("Verification failed. Please try again.")

        except Exception as e:
            st.error(f"Verification process failed: {e}")

def main():
    st.title("Biometric Verification System")
    verification_app = BiometricVerification()
    verification_app.run_verification_process()

if __name__ == "__main__":
    main()