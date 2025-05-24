from gtts import gTTS
from playsound import playsound
import asyncio
import cv2
import numpy as np
import os
import uuid


# Đường dẫn các tệp YOLO
weights_path = "yolov3.weights"
config_path = "yolov3.cfg"
classes_path = "coco.names"

# Đọc tên các lớp
with open(classes_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load YOLO model
net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Màu cho bounding box
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Dictionary lưu trạng thái các đối tượng đã phát hiện
tracked_objects = {}  # {id: {label, center, distance, frames_since_seen}}
FRAMES_TO_FORGET = 30  # Số khung hình để quên một đối tượng không còn xuất hiện
DISTANCE_THRESHOLD = 50  # Ngưỡng khoảng cách để so khớp đối tượng
# Tiêu cự đã hiệu chỉnh
FOCAL_LENGTH = 470.59

REAL_HEIGHT = {
    "person": 170,            # Chiều cao trung bình người lớn
    "bicycle": 100,           # Chiều cao xe đạp
    "car": 150,               # Chiều cao trung bình xe hơi
    "motorbike": 110,         # Chiều cao xe máy
    "aeroplane": 1200,        # Chiều cao trung bình máy bay (khi đậu)
    "bus": 330,               # Chiều cao xe buýt
    "train": 400,             # Chiều cao toa tàu
    "truck": 350,             # Chiều cao xe tải
    "boat": 250,              # Chiều cao trung bình tàu thuyền nhỏ
    "traffic light": 300,     # Chiều cao đèn giao thông
    "fire hydrant": 90,       # Chiều cao vòi cứu hỏa
    "stop sign": 200,         # Chiều cao biển dừng
    "parking meter": 150,     # Chiều cao đồng hồ đỗ xe
    "bench": 45,              # Chiều cao ghế băng
    "bird": 20,               # Chiều cao chim (ví dụ bồ câu)
    "cat": 25,                # Chiều cao mèo
    "dog": 50,                # Chiều cao chó
    "horse": 160,             # Chiều cao ngựa
    "sheep": 60,              # Chiều cao cừu
    "cow": 140,               # Chiều cao bò
    "elephant": 300,          # Chiều cao voi
    "bear": 200,              # Chiều cao gấu (đứng thẳng)
    "zebra": 140,             # Chiều cao ngựa vằn
    "giraffe": 500,           # Chiều cao hươu cao cổ
    "backpack": 50,           # Chiều cao balo
    "umbrella": 90,           # Chiều cao ô (dù)
    "handbag": 40,            # Chiều cao túi xách
    "tie": 60,                # Chiều dài cà vạt
    "suitcase": 60,           # Chiều cao vali
    "frisbee": 3,             # Đường kính đĩa ném
    "skis": 150,              # Chiều dài ván trượt tuyết
    "snowboard": 150,         # Chiều dài ván trượt tuyết
    "sports ball": 22,        # Đường kính bóng
    "kite": 100,              # Chiều dài diều
    "baseball bat": 80,       # Chiều dài gậy bóng chày
    "baseball glove": 25,     # Chiều cao găng tay
    "skateboard": 20,         # Chiều cao ván trượt
    "surfboard": 200,         # Chiều dài ván lướt sóng
    "tennis racket": 70,      # Chiều dài vợt tennis
    "bottle": 30,             # Chiều cao chai nước
    "wine glass": 25,         # Chiều cao ly rượu
    "cup": 15,                # Chiều cao cốc
    "fork": 20,               # Chiều dài nĩa
    "knife": 20,              # Chiều dài dao
    "spoon": 18,              # Chiều dài thìa
    "bowl": 10,               # Chiều cao bát
    "banana": 20,             # Chiều dài chuối
    "apple": 8,               # Đường kính táo
    "sandwich": 5,            # Chiều cao sandwich
    "orange": 7,              # Đường kính cam
    "broccoli": 25,           # Chiều cao súp lơ
    "carrot": 20,             # Chiều dài cà rốt
    "hot dog": 15,            # Chiều dài hot dog
    "pizza": 30,              # Đường kính pizza
    "donut": 10,              # Đường kính donut
    "cake": 15,               # Chiều cao bánh ngọt
    "chair": 90,              # Chiều cao ghế
    "sofa": 100,              # Chiều cao sofa
    "pottedplant": 40,        # Chiều cao cây cảnh
    "bed": 70,                # Chiều cao giường
    "diningtable": 75,        # Chiều cao bàn ăn
    "toilet": 45,             # Chiều cao bồn cầu
    "tvmonitor": 50,          # Chiều cao màn hình TV
    "laptop": 2,              # Chiều cao laptop (dày nhất)
    "mouse": 5,               # Chiều cao chuột máy tính
    "remote": 15,             # Chiều dài điều khiển
    "keyboard": 5,            # Chiều cao bàn phím
    "cell phone": 15,         # Chiều dài điện thoại
    "microwave": 30,          # Chiều cao lò vi sóng
    "oven": 50,               # Chiều cao lò nướng
    "toaster": 25,            # Chiều cao máy nướng bánh mì
    "sink": 25,               # Chiều cao chậu rửa
    "refrigerator": 180,      # Chiều cao tủ lạnh
    "book": 20,               # Chiều cao sách
    "clock": 30,              # Đường kính đồng hồ
    "vase": 30,               # Chiều cao bình hoa
    "scissors": 15,           # Chiều dài kéo
    "teddy bear": 50,         # Chiều cao gấu bông
    "hair drier": 20,         # Chiều cao máy sấy tóc
    "toothbrush": 18          # Chiều dài bàn chải đánh răng
}


# Hàm phát âm thanh không đồng bộ
async def speak_text(text):
    if text.strip():
        tts = gTTS(text=text, lang='vi')
        tts.save("output.mp3")
        playsound("output.mp3")
        os.remove("output.mp3")

# Ánh xạ tên đối tượng sang tiếng Việt
object_name_translation = {
    "person": "Con người",
    "bicycle": "Xe đạp",
    "car": "Xe hơi",
    "motorbike": "Xe máy",
    "aeroplane": "Máy bay",
    "bus": "Xe buýt",
    "train": "Toa tàu",
    "truck": "Xe tải",
    "boat": "Tàu thuyền",
    "traffic light": "Đèn giao thông",
    "fire hydrant": "Vòi cứu hỏa",
    "stop sign": "Biển dừng",
    "parking meter": "Đồng hồ đỗ xe",
    "bench": "Ghế băng",
    "bird": "Chim",
    "cat": "Mèo",
    "dog": "Chó",
    "horse": "Ngựa",
    "sheep": "Cừu",
    "cow": "Bò",
    "elephant": "Voi",
    "bear": "Gấu",
    "zebra": "Ngựa vằn",
    "giraffe": "Hươu cao cổ",
    "backpack": "Balo",
    "umbrella": "Ô",
    "handbag": "Túi xách",
    "tie": "Cà vạt",
    "suitcase": "Vali",
    "frisbee": "Đĩa ném",
    "skis": "Ván trượt tuyết",
    "snowboard": "Ván trượt tuyết",
    "sports ball": "Bóng thể thao",
    "kite": "Diều",
    "baseball bat": "Gậy bóng chày",
    "baseball glove": "Găng tay bóng chày",
    "skateboard": "Ván trượt",
    "surfboard": "Ván lướt sóng",
    "tennis racket": "Vợt tennis",
    "bottle": "Chai nước",
    "wine glass": "Ly rượu",
    "cup": "Cốc",
    "fork": "Nĩa",
    "knife": "Dao",
    "spoon": "Thìa",
    "bowl": "Bát",
    "banana": "Chuối",
    "apple": "Táo",
    "sandwich": "Sandwich",
    "orange": "Cam",
    "broccoli": "Súp lơ",
    "carrot": "Cà rốt",
    "hot dog": "Hot dog",
    "pizza": "Pizza",
    "donut": "Donut",
    "cake": "Bánh ngọt",
    "chair": "Ghế",
    "sofa": "Sofa",
    "pottedplant": "Cây cảnh",
    "bed": "Giường",
    "diningtable": "Bàn ăn",
    "toilet": "Bồn cầu",
    "tvmonitor": "Màn hình TV",
    "laptop": "Laptop",
    "mouse": "Chuột máy tính",
    "remote": "Điều khiển",
    "keyboard": "Bàn phím",
    "cell phone": "Điện thoại",
    "microwave": "Lò vi sóng",
    "oven": "Lò nướng",
    "toaster": "Máy nướng bánh mì",
    "sink": "Chậu rửa",
    "refrigerator": "Tủ lạnh",
    "book": "Sách",
    "clock": "Đồng hồ",
    "vase": "Bình hoa",
    "scissors": "Kéo",
    "teddy bear": "Gấu bông",
    "hair drier": "Máy sấy tóc",
    "toothbrush": "Bàn chải đánh răng"
}


# Đọc video từ camera
cap = cv2.VideoCapture("sample.mp4")
#cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Tiền xử lý ảnh
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # Danh sách lưu kết quả phát hiện
    boxes, confidences, class_ids, distances = [], [], [], []
    detected_objects_within_range = []

    LEFT_REGION, RIGHT_REGION = width // 3, 2 * width // 3

    # Xử lý kết quả đầu ra
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)
                x, y = int(center_x - w / 2), int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indexes) > 0:
        indexes = indexes.flatten()
    else:
        indexes = []

    # Xác định vùng
    width_third = width // 3
    height_third = height // 3

    # Xử lý kết quả đầu ra
    for i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])

        # Chuyển tên đối tượng sang tiếng Việt
        label_vn = object_name_translation.get(label, label)  # Nếu không có trong từ điển thì giữ nguyên tên

        confidence = confidences[i]
        color = colors[class_ids[i]]

        center_x = x + w // 2
        center_y = y + h // 2

        # Xác định khu vực
        if center_x < width_third:
            if center_y < height_third:
                region = "phía trên bên trái"
            elif center_y < 2 * height_third:
                region = "bên trái"
            else:
                region = "phía dưới bên trái"
        elif center_x < 2 * width_third:
            if center_y < height_third:
                region = "phía trên"
            elif center_y < 2 * height_third:
                region = "chính giữa"
            else:
                region = "phía dưới"
        else:
            if center_y < height_third:
                region = "phía trên bên phải"
            elif center_y < 2 * height_third:
                region = "bên phải"
            else:
                region = "phía dưới bên phải"

        if label in REAL_HEIGHT:
            distance = (FOCAL_LENGTH * REAL_HEIGHT[label]) / h
            distance_meters = distance / 100  # Chuyển từ cm sang mét

            # Thêm cơ chế theo dõi đối tượng mới
            matched = False
            for obj_id, tracked_obj in tracked_objects.items():
                tracked_center = tracked_obj["center"]
                distance_to_tracked = np.linalg.norm(np.array(tracked_center) - np.array((center_x, center_y)))

                # Nếu đối tượng khớp với đối tượng đã theo dõi
                if distance_to_tracked < DISTANCE_THRESHOLD and tracked_obj["label"] == label_vn:
                    tracked_objects[obj_id]["center"] = (center_x, center_y)
                    tracked_objects[obj_id]["distance"] = distance_meters
                    tracked_objects[obj_id]["frames_since_seen"] = 0
                    matched = True
                    break

            # Nếu không khớp, thêm đối tượng mới và phát âm thanh
            if not matched:
                obj_id = str(uuid.uuid4())
                tracked_objects[obj_id] = {
                    "label": label_vn,
                    "center": (center_x, center_y),
                    "distance": distance_meters,
                    "frames_since_seen": 0
                }
                detected_objects_within_range.append((label_vn, distance_meters, region))

            cv2.putText(frame, f"{label_vn} {distance_meters:.2f} m {region}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Xóa đối tượng không còn xuất hiện sau FRAMES_TO_FORGET
    for obj_id in list(tracked_objects.keys()):
        tracked_objects[obj_id]["frames_since_seen"] += 1
        if tracked_objects[obj_id]["frames_since_seen"] > FRAMES_TO_FORGET:
            del tracked_objects[obj_id]

    # Phát âm thanh cho đối tượng mới
    if detected_objects_within_range:
        detected_texts = [
            f"{obj[0]} ở {obj[2]} cách bạn {obj[1]:.2f} m"
            for obj in detected_objects_within_range
        ]
        text_to_speak = ", ".join(detected_texts)
        print(detected_texts)

        # Phát âm thanh không đồng bộ
        asyncio.run(speak_text(text_to_speak))

    cv2.imshow("Image", frame)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()