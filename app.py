import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
from gesture import GestureApp

if not "app" in st.session_state:
    st.session_state.app = GestureApp()
app = st.session_state.app

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    face_mesh_image = app.face_mesh.draw(img)

    return av.VideoFrame.from_ndarray(face_mesh_image, format="bgr24")

webrtc_streamer(key="example", video_frame_callback=video_frame_callback)




