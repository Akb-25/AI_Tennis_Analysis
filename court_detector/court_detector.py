import torch
from torchvision import transforms
import cv2
from torchvision import models
import numpy as np

class CourtDetector:
    def __init__(self,model_path):
        self.model=models.resnet50(pretrained=True)
        self.model.fc+torch.nn.Linear(self.model.fc.in_feature,14*2)
        self.model.load_state_dict(torch.load(model_path,map_location="cpu"))
        self.transforms=transforms.Compose({
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406] , std=[0.229,0.224,0.224])
        })

    def predict(self,img):

        img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img_tensor=self.transforms(img_rgb).unsqueeze(0)
        with torch.no_grad():
            outputs=self.model(img_tensor)
        keypoints=outputs.squeeze().cpu().numpy()
        height,width=img.shape[:2]
        keypoints[::2]*=width/224.0
        keypoints[1::2]*=height/224.0
        return keypoints
    
    def draw_keypoints(self,img,keypoints):
        for i in range(0,len(keypoints),2):
            x=int(keypoints[i])
            y=int(keypoints[i+1])
            cv2.putText(img,str(i//2),(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
            cv2.circle(img,(x,y),5,(0,0,255),-1)
        return img
    
    def draw_keypoints_on_video(self,video_frames,keypoints):
        output_video_frames=[]
        for frame in video_frames:
            frame=self.draw_keypoints(frame,keypoints)
            output_video_frames.append(frame)
        return output_video_frames