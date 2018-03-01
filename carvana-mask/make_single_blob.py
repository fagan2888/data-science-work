import cv2

def make_single_blob(img):
    # get contours
    _, contours, hier = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    hier = hier[0]
    
    # find contour with largest area
    max_area = 0
    max_area_idx = 0
    for idx,hier_elem in enumerate(hier):
        if hier_elem[3] == -1:
            area = cv2.contourArea(contours[idx])
            if area > max_area:
                max_area = area
                max_area_idx = idx

    # fill in that contour
    img = cv2.drawContours(img, contours, max_area_idx, 1, -1)

    # make sure all other top-level contours are not filled in
    for idx,hier_elem in enumerate(hier):
        if hier_elem[3] == -1 and idx != max_area_idx:
            img = cv2.drawContours(img, contours, idx, 0, -1)

    return img
