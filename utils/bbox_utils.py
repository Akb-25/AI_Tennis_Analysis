def get_center_of_bbox(bbox):
    x1,x2,y1,y2=bbox
    centerx=(x1+x2)/2
    centery=(y1+y2)/2
    return (int(centerx),int(centery))

def measure_distance_bw(p1,p2):
    return ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5