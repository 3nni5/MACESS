import cv2
import numpy as np
import mediapipe as mp
import PySimpleGUI as sg

def extract_lips_landmarks(frame, results):
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        lips_idx = list(mp.solutions.face_mesh.FACEMESH_LIPS)
        lips = np.ravel(lips_idx)
        lips_coordinates = [(int(pt.x * frame.shape[1]), int(pt.y * frame.shape[0]))
                            for i, pt in enumerate(face_landmarks.landmark) if i in lips]
        return lips_coordinates
    else:
        return None

def create_lips_mask(frame, lips_coordinates, lips_color, alpha):
    mask = np.zeros_like(frame, dtype=np.uint8)
    convex_hull = cv2.convexHull(np.array(lips_coordinates))
    mask = cv2.fillConvexPoly(mask, convex_hull, lips_color)
    return mask

def main():
    face_mesh = mp.solutions.face_mesh
    face_detector = face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)

    # 利用可能な唇の色のプリセット
    lip_presets = {
        "Red": (0, 0, 255),
        "Blue": (255, 0, 0),
        "Green": (0, 255, 0),
        "Purple": (128, 0, 128),
        "Orange": (0, 165, 255),
        "Pink": (203, 192, 255),
        "Yellow": (0, 255, 255),
        "Brown": (42, 42, 165),
        "Gray": (128, 128, 128),
        "White": (255, 255, 255),
    }

    selected_preset = "Red"  # 初期の唇の色
    alpha = 0.5  # 初期の透明度

    # もともとの唇の色
    original_lips_color = (140, 166, 243)
    original_lips_alpha = 0.5  # 固定の透明度

    # PySimpleGUIのウィンドウレイアウト
    layout = [
        [sg.Image(filename="", key="-IMAGE-")],
        [sg.Text("Choose Lips Color:"), sg.Combo(list(lip_presets.keys()), default_value=selected_preset, key="-COLOR-"),
         sg.Text("Adjust Lips Intensity:"), sg.Slider(range=(0, 1), default_value=alpha, resolution=0.01, orientation="h", key="-ALPHA-")],
        [sg.Exit()]
    ]

    window = sg.Window("MACESS", layout, finalize=True, resizable=True, element_justification='c')

    while cap.isOpened():
        event, values = window.read(timeout=20)

        if event == sg.WINDOW_CLOSED or event == "Exit":
            break

        selected_preset = values["-COLOR-"]
        lips_color = lip_presets[selected_preset]
        alpha = values["-ALPHA-"]

        ret, frame = cap.read()

        if not ret:
            sg.popup_error("Error reading from camera.")
            break

        results = face_detector.process(frame)
        lips_coordinates = extract_lips_landmarks(frame, results)

        if lips_coordinates:
            # 1つ目のレイヤー: もともとの唇の色
            original_lips_mask = create_lips_mask(frame, lips_coordinates, original_lips_color, original_lips_alpha)

            # 2つ目のレイヤー: ユーザが選択した唇の色
            user_lips_mask = create_lips_mask(frame, lips_coordinates, lips_color, alpha)

            # ベースのレイヤー: カメラ映像
            resulting_image = cv2.addWeighted(frame, 1, original_lips_mask, original_lips_alpha, 0)
            resulting_image = cv2.addWeighted(resulting_image, 1, user_lips_mask, alpha, 0)

            # OpenCVの画像をPySimpleGUIのウィンドウに描画
            imgbytes = cv2.imencode(".png", resulting_image)[1].tobytes()
            window["-IMAGE-"].update(data=imgbytes)
        else:
            imgbytes = cv2.imencode(".png", frame)[1].tobytes()
            window["-IMAGE-"].update(data=imgbytes)

    cap.release()
    window.close()

if __name__ == "__main__":
    main()
