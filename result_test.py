import cv2
import gdown
import torch
import numpy as np
from pathlib import Path
from torchvision import transforms
from ultralytics import YOLO

from boxmot import BotSort
from boxmot.utils.ops import letterbox

# Preprocessing pipeline
preprocess = transforms.Compose([transforms.ToTensor()])
device = torch.device('cuda')
yolox_model = 'yolox_s.pt'
yolox_model_path = Path(yolox_model)

# Initialize YOLOX model
exp = get_exp(None, 'yolox_s')
exp.num_classes = 1
ckpt = torch.load(yolox_model_path, map_location=device)

model = exp.get_model()
model.load_state_dict(ckpt["model"])
model = fuse_model(model).to(device).eval()

# Initialize tracker
tracker = BotSort(reid_weights=Path('osnet_x0_25_msmt17.pt'), device=device, half=False)

# Video capture setup
vid = cv2.VideoCapture(0)

while True:
    ret, frame = vid.read()
    if not ret:
        break

    # Preprocess frame
    frame_letterbox, ratio, (dw, dh) = letterbox(frame, new_shape=[640, 640], auto=False, scaleFill=True)
    frame_tensor = preprocess(frame_letterbox).unsqueeze(0).to(device)

    # Detection with YOLOX
    with torch.no_grad():
        dets = model(frame_tensor)
    dets = postprocess(dets, 1, 0.5, 0.2, class_agnostic=True)[0]

    if dets is not None:
        # Rescale coordinates from letterbox back to the original frame size
        dets[:, 0] = (dets[:, 0] - dw) / ratio[0]
        dets[:, 1] = (dets[:, 1] - dh) / ratio[1]
        dets[:, 2] = (dets[:, 2] - dw) / ratio[0]
        dets[:, 3] = (dets[:, 3] - dh) / ratio[1]
        dets[:, 4] *= dets[:, 5]
        dets = dets[:, [0, 1, 2, 3, 4, 6]].cpu().numpy()
    else:
        dets = np.empty((0, 6))

    # Update tracker
    res = tracker.update(dets, frame)

    # Plot results and display
    tracker.plot_results(frame, show_trajectories=True)
    cv2.imshow('BoXMOT + YOLOX', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
vid.release()
cv2.destroyAllWindows()