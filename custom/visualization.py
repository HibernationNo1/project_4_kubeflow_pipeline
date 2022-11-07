import cv2
import os
import numpy as np
import torch
import random

from eval import parse_inferece_result 

def show_result(img,
                result,
                class_names,
                score_thr=0.3,
                out_file=None): 
    """Draw `result` over `img`.

    Args:
        img (numpy): The image to be displayed.
        result (Tensor or tuple): The results to draw over `img`
            bbox_result or (bbox_result, segm_result).
        score_thr (float, optional): Minimum score of bboxes to be shown.
            Default: 0.3.
        out_file (str or None): The filename to write the image.
            Default: None.

    Returns:
        img (numpy): Only if not `show` or `out_file`
    """
    
    bboxes, labels, segms = parse_inferece_result(result)

    # draw bounding boxes
    img = draw_to_img(img,
                      bboxes,
                      labels,
                      segms,
                      class_names=class_names,
                      score_thr=score_thr)
    
    if out_file is not None:
        cv2.imwrite(out_file, img)
        
        
def mask_to_polygon(masks):
    polygons = []
    for mask in masks:       
        polygon, _ = bitmap_to_polygon(mask)
        polygons.append(polygon[0])
    return polygons



def draw_to_img(img,
                bboxes=None,
                labels=None,
                segms=None,
                class_names=None,
                score_thr=0
                ):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str | ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        segms (ndarray | None): Masks, shaped (n,h,w) or None.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown. Default: 0.
        bbox_color (list[tuple] | tuple | str | None): Colors of bbox lines.
           If a single color is given, it will be applied to all classes.
           The tuple of color should be in RGB order. Default: 'green'.
        text_color (list[tuple] | tuple | str | None): Colors of texts.
           If a single color is given, it will be applied to all classes.
           The tuple of color should be in RGB order. Default: 'green'.
        mask_color (list[tuple] | tuple | str | None, optional): Colors of
           masks. If a single color is given, it will be applied to all
           classes. The tuple of color should be in RGB order.
           Default: None.
        thickness (int): Thickness of lines. Default: 2.
        font_size (int): Font size of texts. Default: 13.
        show (bool): Whether to show the image. Default: True.
        win_name (str): The window name. Default: ''.
        wait_time (float): Value of waitKey param. Default: 0.
        out_file (str, optional): The filename to write the image.
            Default: None.

    Returns:
        ndarray: The image with bboxes drawn on it.
    """    
    assert bboxes is None or bboxes.ndim == 2, \
        f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert labels.ndim == 1, \
        f' labels ndim should be 1, but its ndim is {labels.ndim}.'
    assert bboxes is None or bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
        f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
    assert bboxes is None or bboxes.shape[0] <= labels.shape[0], \
        'labels.shape[0] should not be less than bboxes.shape[0].'
    assert segms is None or segms.shape[0] == labels.shape[0], \
        'segms.shape[0] and labels.shape[0] should have the same length.'
    assert segms is not None or bboxes is not None, \
        'segms and bboxes should not be None at the same time.'

    # width, height
    img = img.astype(np.uint8)
    
    if score_thr > 0:
        assert bboxes is not None and bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]
            
        
    # img = bgr2rgb(img)
    
    scores = bboxes[:, -1]      # [num_instance]
    bboxes = bboxes[:, :4]      # [num_instance, [x_min, y_min, x_max, y_max]]
    
    object_labels = []
    for label_num, score in zip(labels, scores):
        confidence = int(round(score, 2)*100)
        object_name = class_names[label_num]
        object_labels.append(f"{object_name} {confidence}%")
    # object_labels len: num_instance,  each type: str,     e.g. object 34%
   
    polygons = mask_to_polygon(segms)
    # len(polygons) : num_instance,     len(polygons[n]): num_points of polygon
    
    
    colors = get_colors(len(object_labels))
    img = draw_box(img, bboxes, colors)
    img = draw_polygon(img, polygons, colors)
    img = put_text(img, object_labels, bboxes)
    # TODO draw_mask


    return img
    
   
  
def bgr2rgb(input_img):
    out_img = cv2.cvtColor(input_img, 4)
    return out_img
    
    
   
def bitmap_to_polygon(bitmap):
    """Convert masks from the form of bitmaps to polygons.

    Args:
        bitmap (ndarray): masks in bitmap representation.

    Return:
        list[ndarray]: the converted mask in polygon representation.
        bool: whether the mask has holes.
    """
    bitmap = np.ascontiguousarray(bitmap).astype(np.uint8)
    # cv2.RETR_CCOMP: retrieves all of the contours and organizes them
    #   into a two-level hierarchy. At the top level, there are external
    #   boundaries of the components. At the second level, there are
    #   boundaries of the holes. If there is another contour inside a hole
    #   of a connected component, it is still put at the top level.
    # cv2.CHAIN_APPROX_NONE: stores absolutely all the contour points.
    outs = cv2.findContours(bitmap, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = outs[-2]
    hierarchy = outs[-1]
    if hierarchy is None:
        return [], False
    # hierarchy[i]: 4 elements, for the indexes of next, previous,
    # parent, or nested contours. If there is no corresponding contour,
    # it will be -1.
    with_hole = (hierarchy.reshape(-1, 4)[:, 3] >= 0).any()
    contours = [c.reshape(-1, 2) for c in contours]
    return contours, with_hole

def get_colors(num_instance):
    
    
    colors = []
    for _ in range(num_instance):
        rgb = []
        for _ in range(3):
            rgb.append(random.randrange(100,255))
        colors.append(tuple(rgb))
    
    return colors
    
    

def draw_box(img, bboxes, colors, thickness= 1, lineType= cv2.LINE_4):
    for bbox, edge_color in zip(bboxes, colors):
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(img, 
                    (int(x_min), int(y_min)), (int(x_max), int(y_max)),
                    edge_color, 
                    lineType = lineType, 
                    thickness = thickness)
    
    return img


def draw_polygon(img, polygons, colors, radius = 1):
    for polygon, edge_color in zip(polygons, colors):
        for point in polygon:
            cv2.circle(img, point, color = edge_color, thickness= -1, radius = radius)
    
    return img
    
def draw_mask(img, mask):  # TODO
    assert img.shape[:2] == mask.shape
    sub_mask = np.one()
    print(f"mask.shape : {mask.shape}")
    
    
def put_text(img, object_labels, bboxes, fontface = cv2.FONT_HERSHEY_SIMPLEX):
    for label_text, bboxe in zip(object_labels, bboxes):
        x_min, y_min, x_max, y_max = bboxe
        width, height = x_max - x_min, y_max - y_min
        text_lb = (int(x_min), int(y_min))
        
        scale = width*height/70000
        fontscale = scale if scale>0.4 else 0.4 
        thickness = int(fontscale) if fontscale>=1 else 1
        text_size, _ = cv2.getTextSize(label_text, fontFace = fontface, fontScale = fontscale, thickness = thickness)
        text_w, text_h = text_size
        text_rt = (int(x_min+text_w), int(y_min-text_h))
        
        cv2.rectangle(img, text_lb, text_rt, (0, 0, 0), thickness=-1)
        cv2.putText(img, label_text, 
                    org = text_lb, 
                    fontFace = fontface, 
                    fontScale = fontscale, 
                    color = (255, 255, 255), 
                    lineType = cv2.LINE_AA,
                    thickness = thickness)
    return img