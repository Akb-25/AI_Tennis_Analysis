def convert_pixel_dist_to_meter(pixel_dist,reference_height_meter,reference_height_pixel):
    return (pixel_dist*reference_height_meter)/reference_height_pixel
def convert_meter_to_pixel(meter_dist,reference_height_meter,reference_height_pixel):
    return (meter_dist*reference_height_pixel)/reference_height_meter
