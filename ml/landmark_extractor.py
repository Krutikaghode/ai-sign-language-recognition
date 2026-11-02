import cv2
import mediapipe as mp
import numpy as np

class HandLandmarkExtractor:
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def extract_landmarks(self, image):
        """
        Extracts 21 3D landmarks for up to 2 hands (total 126 values: 2 * 21 * 3).
        If a hand is missing, fills with zeros.
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        landmarks_vector = np.zeros(126, dtype=np.float32)
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if idx >= 2:
                    break
                coords = []
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                
                coords = np.array(coords, dtype=np.float32).reshape(21, 3)
                wrist = coords[0]
                normalized_coords = coords - wrist
                
                start_idx = idx * 63
                landmarks_vector[start_idx:start_idx + 63] = normalized_coords.flatten()
                
        return landmarks_vector, results

    def draw_landmarks(self, image, results):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
        return image
