import cv2
import os
import re
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from ML.model import TransformerNet

class Webcam():
    # Get paths and set vars
    weights_fname = "candy.pth"
    script_path = os.path.dirname(os.path.abspath(__file__))
    path_to_weights = os.path.join(script_path, "models", weights_fname)
    resolution = (640, 480)
    # Change to GPU if desired
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerNet()
    with torch.no_grad():
        state_dict = torch.load(path_to_weights)
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        model.load_state_dict(state_dict)
        model.to(device)


    # Load PyTorch Model


    def __init__(self):
        self.video = cv2.VideoCapture(-1)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()

        ret, jpeg = cv2.imencode('.jpg', frame)

        return jpeg.tobytes()

    def style_image(self):
        ret, frame = self.video.read()
        cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        img = pil_im.resize(self.resolution)

        # Transforms to feed to network
        small_frame_tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        small_frame_tensor = small_frame_tensor_transform(img)
        small_frame_tensor = small_frame_tensor.unsqueeze(0).to(self.device)

        # Run inference and resize
        output = self.model(small_frame_tensor).cpu()
        styled = output[0]
        styled = styled.clone().clamp(0, 255).detach().numpy()
        styled = styled.transpose(1, 2, 0).astype("uint8")
        # styled_resized_frame = cv2.resize(styled, (frame.shape[0], frame.shape[1]))
        ret, jpeg = cv2.imencode('.jpg', styled)
        return jpeg.tobytes()


