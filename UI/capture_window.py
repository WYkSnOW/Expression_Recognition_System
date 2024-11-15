import cv2
import time
from helper.face_detection import detect_faces, draw_face_boxes
from helper.emotion_model import predict_emotion
from helper.utils import convert_frame_to_image
from helper.face_mesh import FaceMeshDetector  # 引入 FaceMesh 模块


def click_event(event, x, y, flags, param):
    """
    鼠标点击事件处理函数，用于检测按钮的点击。
    """
    global running, button_position, face_mesh_button_position, face_box_button_position
    global show_face_mesh, show_face_box, button_size
    if event == cv2.EVENT_LBUTTONDOWN:
        # 检查鼠标是否点击退出按钮
        if button_position and \
           button_position[0] <= x <= button_position[0] + button_size[0] and \
           button_position[1] <= y <= button_position[1] + button_size[1]:
            print("Exit button clicked. Exiting...")
            running = False
        # 检查鼠标是否点击 FaceMesh 按钮
        elif face_mesh_button_position and \
             face_mesh_button_position[0] <= x <= face_mesh_button_position[0] + button_size[0] and \
             face_mesh_button_position[1] <= y <= face_mesh_button_position[1] + button_size[1]:
            show_face_mesh = not show_face_mesh
            print(f"FaceMesh display toggled: {'ON' if show_face_mesh else 'OFF'}")
        # 检查鼠标是否点击人脸方框按钮
        elif face_box_button_position and \
             face_box_button_position[0] <= x <= face_box_button_position[0] + button_size[0] and \
             face_box_button_position[1] <= y <= face_box_button_position[1] + button_size[1]:
            show_face_box = not show_face_box
            print(f"Face Box display toggled: {'ON' if show_face_box else 'OFF'}")


def show_loading_window():
    """
    显示加载窗口，显示“Loading”字样。
    """
    loading_window = "Loading"
    cv2.namedWindow(loading_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(loading_window, 400, 200)
    loading_frame = 255 * (cv2.getStructuringElement(cv2.MORPH_RECT, (400, 200))).astype("uint8")
    cv2.putText(loading_frame, "Loading", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    cv2.imshow(loading_window, loading_frame)
    cv2.waitKey(1)
    return loading_window


def main():
    # 显示加载窗口
    loading_window = show_loading_window()

    # 初始化 FaceMeshDetector
    detector = FaceMeshDetector()

    # 初始化状态变量
    global running, button_position, face_mesh_button_position, face_box_button_position
    global show_face_mesh, show_face_box, button_size
    show_face_mesh = True  # 默认显示 FaceMesh
    show_face_box = True   # 默认显示人脸框
    running = True

    # 设置检测间隔和初始变量
    detection_interval = 1  # in seconds
    current_label = "No Face Detected"
    last_detection_time = time.time()

    # 设置按钮尺寸
    button_size = (100, 40)

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        cv2.destroyWindow(loading_window)
        return

    # 关闭加载窗口并设置主窗口
    cv2.destroyWindow(loading_window)
    cv2.namedWindow("Real-Time FaceMesh & Emotion")
    cv2.setMouseCallback("Real-Time FaceMesh & Emotion", click_event)

    try:
        while running:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            # 计算按钮的位置（退出按钮在右上角，其下方为其他按钮）
            frame_width = frame.shape[1]
            button_position = (frame_width - button_size[0] - 10, 10)  # 退出按钮
            face_mesh_button_position = (frame_width - button_size[0] - 10, 60)  # FaceMesh 按钮
            face_box_button_position = (frame_width - button_size[0] - 10, 110)  # 人脸框按钮

            # 检测是否存在人脸并绘制面部网格
            frame, faces_mesh, face_detected = detector.find_face_mesh(frame, draw=show_face_mesh)

            # 始终运行检测逻辑，无论显示状态如何
            face_boxes = detect_faces(frame)

            # 如果需要，绘制人脸方框
            if show_face_box:
                draw_face_boxes(frame, face_boxes)

            # 表情预测逻辑
            if face_detected and face_boxes:
                # 每隔一段时间对检测到的区域进行表情预测
                current_time = time.time()
                if current_time - last_detection_time >= detection_interval:
                    x, y, x_end, y_end = face_boxes[0]  # 只检测第一个人脸
                    face_image = convert_frame_to_image(frame, x, y, x_end, y_end)
                    current_label = predict_emotion(face_image)
                    last_detection_time = current_time
            else:
                current_label = "No Face Detected"

            # 显示预测结果
            cv2.putText(frame, f"Predicted: {current_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 绘制退出按钮
            cv2.rectangle(frame, button_position,
                          (button_position[0] + button_size[0], button_position[1] + button_size[1]),
                          (0, 0, 255), -1)
            cv2.putText(frame, "Exit", (button_position[0] + 10, button_position[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # 绘制 FaceMesh 控制按钮
            cv2.rectangle(frame, face_mesh_button_position,
                          (face_mesh_button_position[0] + button_size[0], face_mesh_button_position[1] + button_size[1]),
                          (0, 255, 0), -1)
            cv2.putText(frame, "Mesh", (face_mesh_button_position[0] + 10, face_mesh_button_position[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # 绘制人脸框控制按钮
            cv2.rectangle(frame, face_box_button_position,
                          (face_box_button_position[0] + button_size[0], face_box_button_position[1] + button_size[1]),
                          (255, 0, 0), -1)
            cv2.putText(frame, "Box", (face_box_button_position[0] + 10, face_box_button_position[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # 显示实时帧
            cv2.imshow("Real-Time FaceMesh & Emotion", frame)

            # 检测关闭窗口或按下Escape退出
            if cv2.getWindowProperty("Real-Time FaceMesh & Emotion", cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed. Exiting...")
                break

            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                print("Exiting...")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
