from typing import Optional
from dataclasses import dataclass
import time
import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

class EyeControl:
    ...


from config import LEFT_EYE, RIGHT_EYE

class FaceMesh:
    def __init__(self):
        self.loop_time_sec = 0
        self.close_left_eye_timer = 0
        self.close_right_eye_timer = 0
        self.open_left_eye_timer = 0
        self.open_right_eye_timer = 0
        self.left_eye_close = False
        self.right_eye_close = False

    def get_face_mesh(self, image):
        """Get facemesh coords powered by mediapipe"""
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:
            results = face_mesh.process(image)
            image.flags.writeable = True
            return results

    @staticmethod
    def create_blank_image(height=1080, width=1920, color=None):
        """Return blank image"""
        blank = np.zeros((height, width, 3), np.uint8)
        if color:
            blank += np.array(color, dtype=np.uint8)
        return blank

    @staticmethod
    def distance(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def detect_close_eye(self, results):
        if self.loop_time_sec > 0:
            self.loop_time_sec = time.perf_counter() - self.loop_time_sec

        for face_landmarks in results.multi_face_landmarks:
            marks = face_landmarks.landmark
            base_distance = self.distance((marks[144].x, marks[144].y), (marks[145].x, marks[145].y))
            left_eye_distance = self.distance((marks[374].x, marks[374].y), (marks[386].x, marks[386].y))
            right_eye_distance = self.distance((marks[145].x, marks[145].y), (marks[159].x, marks[159].y))

            if base_distance > left_eye_distance:
                self.close_left_eye_timer += self.loop_time_sec
                self.open_left_eye_timer = 0
            else:
                self.open_left_eye_timer += self.loop_time_sec
                self.close_left_eye_timer = 0

            if base_distance > right_eye_distance:
                self.close_right_eye_timer += self.loop_time_sec
                self.open_right_eye_timer = 0
            else:
                self.open_right_eye_timer += self.loop_time_sec
                self.close_right_eye_timer = 0

        if self.close_left_eye_timer > 0.5:
            self.left_eye_close = True

        if self.close_right_eye_timer > 0.5:
            self.right_eye_close = True

        if self.open_left_eye_timer > 0.5:
            self.left_eye_close = False

        if self.open_right_eye_timer > 0.5:
            self.right_eye_close = False

        self.loop_time_sec = time.perf_counter()

        return self.left_eye_close, self.right_eye_close

    def draw(self, image):
        results = self.get_face_mesh(image)
        if results.multi_face_landmarks:
            left_close_eye, right_eye_close = self.detect_close_eye(results)
            # close_eye = False
            left_color = (0, 255, 255) if left_close_eye else (0, 255, 0)
            right_color = (0, 255, 255) if right_eye_close else (0, 255, 0)
            mesh_image = self.draw_results(results, right_color, left_color)
        else:
            # No faces found.
            mesh_image = self.create_blank_image(color=[14, 17, 23])

        mesh_image = cv2.resize(mesh_image, dsize=(640, 640))
        blank_image = self.create_blank_image(color=[14, 17, 23], width=640, height=640)
        mesh_image = cv2.hconcat([mesh_image, blank_image])

        return mesh_image

    @staticmethod
    def draw_mesh(image, results):
        """Return image drawn a face mesh"""
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
            )
        return image

    @staticmethod
    def draw_eyes_contour(image, results, right_color, left_color):
        """Return image drawn eyes contour line"""
        height, width = image.shape[:2]
        for face_landmarks in results.multi_face_landmarks:
            # Add eyes information
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            contours, _ = cv2.findContours(
                image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            marks = face_landmarks.landmark
            image += np.array([14, 17, 23], dtype=np.uint8)
            thickness = 2
            for mark in LEFT_EYE:
                s_mark, e_mark = mark
                sp = (int(marks[s_mark].x * width), int(marks[s_mark].y * height))
                ep = (int(marks[e_mark].x * width), int(marks[e_mark].y * height))
                cv2.line(image, sp, ep, left_color, thickness)
            for mark in RIGHT_EYE:
                s_mark, e_mark = mark
                sp = (int(marks[s_mark].x * width), int(marks[s_mark].y * height))
                ep = (int(marks[e_mark].x * width), int(marks[e_mark].y * height))
                cv2.line(image, sp, ep, right_color, thickness)

            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            padding = 100
            x -= padding//2
            y -= padding//2
            w += padding
            h += padding
            x = max(0, min(width, x))
            y = max(0, min(height, y))
            w = max(0, min(width, w))
            h = max(0, min(height, h))
            image = image[y : y + h, x : x + w]
        return image

    def draw_results(self, results, right_color, left_color):
        """Return image drawn a face mesh and eyes contour line"""
        blank_image = self.create_blank_image()
        # Draw face mesh
        mesh_image = self.draw_mesh(blank_image, results)
        # Draw eyes contour line
        mesh_image = self.draw_eyes_contour(
            mesh_image, results, right_color, left_color
        )
        return mesh_image


class GestureApp:
    def __init__(self):
        self.face_mesh = FaceMesh()
