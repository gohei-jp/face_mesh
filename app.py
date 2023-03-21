import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
from gesture import GestureApp

if not "app" in st.session_state:
    st.session_state.app = GestureApp()
app = st.session_state.app

def video_frame_callback(frame):
    image = frame.to_ndarray(format="bgr24")
    image = app.update(image)
    return av.VideoFrame.from_ndarray(image, format="bgr24")

webrtc_streamer(key="example", video_frame_callback=video_frame_callback)