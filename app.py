import streamlit as st
import sqlite3
import hashlib
import cv2
import numpy as np
import tempfile
import os
from glob import glob
from PIL import Image
from ultralytics import YOLO
from norfair import Detection, Tracker

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Semen Analysis AI", layout="wide", page_icon="🧬")

# ---------------- DARK UI ----------------
st.markdown("""
<style>
html, body, [class*="css"]  {
    background-color: #0e1117;
    color: #ffffff;
}
.stButton>button {
    background-color: #00ADB5;
    color: white;
    border-radius: 10px;
    border: none;
}
.card {
    background-color: #161b22;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
}
.metric-box {
    background-color: #1f2933;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
}
h1, h2, h3 { text-align: center; }
img { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ---------------- DB ----------------
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
)
""")
conn.commit()


def hash_password(p):
    return hashlib.sha256(p.encode()).hexdigest()


def add_user(u, p):
    try:
        c.execute("INSERT INTO users VALUES (?, ?)", (u, hash_password(p)))
        conn.commit()
        return True
    except:
        return False


def login_user(u, p):
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (u, hash_password(p)))
    return c.fetchone()


# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""


# ---------------- AUTH UI ----------------
def auth_page():
    st.title("🔐 Semen Analysis AI – Login")

    tab1, tab2 = st.tabs(["Login", "Signup"])

    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            if login_user(u, p):
                st.session_state.logged_in = True
                st.session_state.username = u
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        new_u = st.text_input("New Username")
        new_p = st.text_input("New Password", type="password")
        if st.button("Create Account"):
            if add_user(new_u, new_p):
                st.success("Account created! Please login.")
            else:
                st.error("Username already exists")


if not st.session_state.logged_in:
    auth_page()
    st.stop()

# ---------------- LOGOUT ----------------
st.sidebar.write(f"👤 {st.session_state.username}")
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# ---------------- CONFIG ----------------
DEFECT_MODEL_PATH = "models/defect_best.pt"
MOTILITY_MODEL_PATH = "models/motility_best.pt"

DIST_THRESHOLD = 45
IMMOTILE_VEL = 5
PROGRESS_RATIO = 0.7


# -------- Utility --------
def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def classify(traj, fps):
    path = sum(dist(traj[i], traj[i + 1]) for i in range(len(traj) - 1))
    disp = dist(traj[0], traj[-1])
    vel = path / (len(traj) / fps)
    ratio = disp / path if path > 0 else 0

    if vel < IMMOTILE_VEL:
        return "IM"
    elif ratio > PROGRESS_RATIO:
        return "PR"
    else:
        return "NP"


@st.cache_resource
def load_defect_model():
    return YOLO(DEFECT_MODEL_PATH)


@st.cache_resource
def load_motility_model():
    return YOLO(MOTILITY_MODEL_PATH)


# ---------------- SIDEBAR ----------------
st.sidebar.title("🧬 Semen Analysis AI")
page = st.sidebar.radio("Navigation", ["Defect Analysis", "Motility Analysis", "EDA Dashboard"])

# =========================================================
# 🧬 DEFECT ANALYSIS
# =========================================================
if page == "Defect Analysis":

    st.markdown("## 🧬 Semen Defect Analysis")
    model = load_defect_model()

    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original", use_container_width=True)

        if st.button("Run Detection"):
            img_np = np.array(image)
            results = model(img_np)
            plotted = results[0].plot()
            plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

            with col2:
                st.image(plotted_rgb, caption="Result", use_container_width=True)

# =========================================================
# 🧪 MOTILITY ANALYSIS
# =========================================================
if page == "Motility Analysis":

    st.markdown("## 🧪 Sperm Motility Analysis")

    MODEL_PATH = "motility/runs/detect/train/weights/best.pt"

    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_video:

        st.markdown("### 🎬 Uploaded Video Preview")
        st.video(uploaded_video)

        if st.button("Run Motility Tracking"):

            # ---- Load YOLO model (same as script) ----
            model = YOLO(MODEL_PATH)

            # ---- Norfair tracker (same as script) ----
            tracker = Tracker(
                distance_function="euclidean",
                distance_threshold=DIST_THRESHOLD
            )

            # ---- Save uploaded video to temp ----
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            VIDEO_PATH = tfile.name

            cap = cv2.VideoCapture(VIDEO_PATH)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(3))
            height = int(cap.get(4))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            output_path = tempfile.NamedTemporaryFile(
                suffix=".mp4",
                delete=False
            ).name

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            trajectories = {}

            stframe = st.empty()
            progress_bar = st.progress(0)
            status_text = st.empty()

            frame_count = 0

            st.info("Tracking started...")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # ---- YOLO Detection (same as script) ----
                results = model(frame, imgsz=640, verbose=False)[0]

                detections = []
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]

                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    detections.append(
                        Detection(points=np.array([[cx.item(), cy.item()]]))
                    )

                # ---- Norfair Tracking (same as script) ----
                tracked_objects = tracker.update(detections=detections)

                for obj in tracked_objects:
                    tid = obj.id
                    x, y = obj.estimate[0]

                    trajectories.setdefault(tid, []).append(
                        np.array([x, y])
                    )

                    cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                    cv2.putText(
                        frame,
                        f"ID {tid}",
                        (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 0),
                        1
                    )

                out.write(frame)

                # ---- Streamlit live display ----
                stframe.image(frame, channels="BGR")

                frame_count += 1
                progress = frame_count / total_frames
                progress_bar.progress(min(progress, 1.0))
                status_text.text(
                    f"Processing frame {frame_count}/{total_frames} "
                    f"({progress * 100:.1f}%)"
                )

            cap.release()
            out.release()

            st.success("Tracking video saved")

            # -------- Motility Calculation (same as script) --------
            st.info("Calculating motility...")

            results_dict = {}
            for tid, traj in trajectories.items():
                if len(traj) > fps:
                    results_dict[tid] = classify(traj, fps)

            counts = {"PR": 0, "NP": 0, "IM": 0}
            for r in results_dict.values():
                counts[r] += 1

            total = sum(counts.values())

            st.markdown("### 📊 Motility Summary")

            for k, v in counts.items():
                percent = (v / total * 100) if total > 0 else 0
                st.write(f"{k}: {v} ({percent:.1f}%)")




# =========================================================
# 📊 EDA DASHBOARD
# =========================================================
# =========================================================
# 📊 EDA DASHBOARD
# =========================================================
if page == "EDA Dashboard":

    st.markdown("## 📊 Training Metrics Dashboard")

    tab1, tab2 = st.tabs(["🧬 Defect Analysis", "🧪 Motility Analysis"])

    # -----------------------------------------------------
    # 🧬 DEFECT EDA (SEGMENTATION)
    # -----------------------------------------------------
    with tab1:

        st.markdown("### 🧬 Defect Model Metrics")

        defect_path = r"C:\Users\loges\PycharmProjects\sperm\runs\segment\train"

        if not os.path.exists(defect_path):
            st.error("Defect path not found.")
        else:

            plot_patterns = [
                "results.png",
                "confusion_matrix.png",
                "P_curve.png",
                "R_curve.png",
                "PR_curve.png",
                "F1_curve.png"
            ]

            plot_paths = []
            for pattern in plot_patterns:
                plot_paths.extend(glob(os.path.join(defect_path, pattern)))

            if plot_paths:
                idx = st.slider(
                    "Slide through plots",
                    0,
                    len(plot_paths) - 1,
                    0,
                    key="defect_slider"
                )

                st.image(plot_paths[idx], use_container_width=True)

                with st.expander("Show All Plots"):
                    cols = st.columns(2)
                    for i, img_path in enumerate(plot_paths):
                        cols[i % 2].image(img_path, use_container_width=True)
            else:
                st.warning("No plots found in defect train folder.")

    # -----------------------------------------------------
    # 🧪 MOTILITY EDA (DETECTION)
    # -----------------------------------------------------
    with tab2:

        st.markdown("### 🧪 Motility Model Metrics")

        motility_path = r"C:\Users\loges\PycharmProjects\sperm\motility\runs\detect\train"

        if not os.path.exists(motility_path):
            st.error("Motility path not found.")
        else:

            plot_patterns = [
                "results.png",
                "confusion_matrix.png",
                "P_curve.png",
                "R_curve.png",
                "PR_curve.png",
                "F1_curve.png"
            ]

            plot_paths = []
            for pattern in plot_patterns:
                plot_paths.extend(glob(os.path.join(motility_path, pattern)))

            if plot_paths:
                idx = st.slider(
                    "Slide through plots",
                    0,
                    len(plot_paths) - 1,
                    0,
                    key="motility_slider"
                )

                st.image(plot_paths[idx], use_container_width=True)

                with st.expander("Show All Plots"):
                    cols = st.columns(2)
                    for i, img_path in enumerate(plot_paths):
                        cols[i % 2].image(img_path, use_container_width=True)
            else:
                st.warning("No plots found in motility train folder.")
