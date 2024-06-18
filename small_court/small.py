import cv2
import sys
sys.path.append("../")
import constants
from utils import convert_pixel_dist_to_meter,convert_meter_to_pixel
class SmallCourt():
    def __init__(self,frame):
        self.drawing_rectangle_width=250
        self.drawing_rectangle_height=450
        self.buffer=50
        self.padding_court=20
        self.set_background_pos(frame)
    def set_background_pos(self,frame):
        frame=frame.copy()
        self.endx=frame.shape[1]-self.buffer
        self.endy=self.buffer+self.drawing_rectangle_height
        self.startx=self.endx-self.drawing_rectangle_width
        self.starty=self.endy-self.drawing_rectangle_height
    def set_small_court_position(self,frame):
        self.court_startx=self.startx+self.padding_court
        self.court_starty=self.starty+self.padding_court
        self.court_endx=self.endx-self.padding_court
        self.court_endy=self.endy-self.padding_court
        self.court_drawing_width=self.court_endx-self.court_startx
    def convert_meters_to_pixels(self,meters):
        return convert_meter_to_pixel(meters,constants.DOUBLE_LINE_WIDTH,self.court_drawing_width)
    def set_court_keypoints(self):
        drawing_keypoints=[0]*28
        #point 0
        drawing_keypoints[0],drawing_keypoints[1]=int(self.court_startx),int(self.court_starty)
        #point 1
        drawing_keypoints[2],drawing_keypoints[3]=int(self.court_endx),int(self.court_endy)
        #point 2
        drawing_keypoints[4]=int(self.court_startx)
        drawing_keypoints[5]=self.court_starty+self.convert_meters_to_pixels(constants.DOUBLE_LINE_HEIGHT*2)
        #point 3
        drawing_keypoints[6]=drawing_keypoints[0]+self.court_drawing_width
        drawing_keypoints[7]=drawing_keypoints[5]