import numpy as np
from inference import inference_detector, parse_inferece_result

def compute_iou(infer_box, gt_box):
    """
    infer_box : x_min, y_min, x_max, y_max
    gt_box : x_min, y_min, x_max, y_max
    """
    box1_area = (infer_box[2] - infer_box[0]) * (infer_box[3] - infer_box[1])
    box2_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    
    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(infer_box[0], gt_box[0])       # x_min 중 큰 것
    y1 = max(infer_box[1], gt_box[1])       # y_min 중 큰 것   # (x1, y1) : 두 left_top points 중 큰 값, intersction의 lest_top
    x2 = min(infer_box[2], gt_box[2])       # x_max 중 작은 것
    y2 = min(infer_box[3], gt_box[3])       # y_max 중 작은 것  # (x2, y2) : 두 right_bottom points 중 작은 값  intersction의 right_bottom

    # compute the width and height of the intersection
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou


def get_divided_polygon(polygon, window_num):
    """
        divide polygon by the number of `window_num` piece by sort in x and y direction
    Args:
        polygon (list): 
        window_num (int): 
    
    Return 
        [x_lt_rb_list, y_lt_rb_list]
            len(x_lt_rb_list) == `window_num`
            x_lt_rb_list[0]: [x_min, y_min, x_max, y_max]
    """
    if isinstance(polygon, np.ndarray): polygon = polygon.tolist()
  
    piece_point = int(len(polygon)/window_num)
    polygon_xsort = polygon.copy()
    polygon_xsort.sort(key=lambda x: x[0])
    polygon_ysort = polygon.copy()
    polygon_ysort.sort(key=lambda x: x[1])
    
    
    xsort_div_pol = divide_polygon(polygon_xsort, window_num, piece_point)
    ysort_div_pol = divide_polygon(polygon_ysort, window_num, piece_point)

    x_lt_rb_list, y_lt_rb_list = [], []
    for x_pol, y_pol in zip(xsort_div_pol, ysort_div_pol):
        x_lt_rb_list.append(get_box_from_pol(x_pol))
        y_lt_rb_list.append(get_box_from_pol(y_pol))
    
    
    return [x_lt_rb_list, y_lt_rb_list]
    
    
 
def divide_polygon(polygon_sorted, window_num, piece_point):
    """divide polygon by the number of `window_num` piece

    Args:
        polygon_sorted (list): 
        window_num (int): 
        piece_point (int): 

    Returns:
        sort_list: list, len== [`window_num`],      window_num[n]: list
    """
    sort_list = []
    last_num = 0
    for i in range(window_num):
        if i == window_num-1:
            sort_list.append(polygon_sorted[last_num:])
            break
        
        sort_list.append(polygon_sorted[last_num:piece_point*(i+1)])
        last_num = piece_point*(i+1)
    
    return sort_list    
    

def get_box_from_pol(polygon):
    x_min, y_min, x_max, y_max = 100000, 100000, -1, -1
    
    for point in polygon:
        x, y = point[0], point[1]
        if x < x_min and x > x_max:
            x_min = x_max = x
        elif x < x_min:
            x_min = x
        elif x > x_max:
            x_max = x
        
        if y < y_min and y > y_max:
            y_min = y_max = y
        elif y < y_min:
            y_min = y
        elif y > y_max:
            y_max = y
    
    return [x_min, y_min, x_max, y_max]
        


def get_precision_recall_value(model, cfg, val_dataloader, func_mask_to_polygon):
    classes = model.CLASSES
    num_thrshd_divi = cfg.num_thrshd_divi
    thrshd_value = (cfg.iou_threshold[-1] - cfg.iou_threshold[0]) / num_thrshd_divi
    iou_threshold = [round(cfg.iou_threshold[0] + (thrshd_value*i), 2) for i in range(num_thrshd_divi+1)]
    
    confusion_dict = dict()
    for class_name in classes:
        confusion_dict[class_name] = []
        for i in range(len(iou_threshold)):
            confusion_dict[class_name].append(dict(iou_threshold = iou_threshold[i],
                                            num_gt = 0, 
                                            num_pred = 0, 
                                            num_true = 0)
                                        )
    
    for i, val_data_batch in enumerate(val_dataloader):
        gt_bboxes_list = val_data_batch['gt_bboxes'].data
        gt_labels_list = val_data_batch['gt_labels'].data
        img_list = val_data_batch['img'].data
        gt_masks_list = val_data_batch['gt_masks'].data
        assert len(gt_bboxes_list) == 1 and (len(gt_bboxes_list) ==
                                                len(gt_labels_list) ==
                                                len(img_list) == 
                                                len(gt_masks_list))
        # len: batch_size
        batch_gt_bboxes = gt_bboxes_list[0]           
        batch_gt_labels = gt_labels_list[0]  
        batch_gt_masks = gt_masks_list[0]    
        
        img_metas = val_data_batch['img_metas'].data[0]
        batch_images_path = []    
        for img_meta in img_metas:
            batch_images_path.append(img_meta['filename'])
        batch_results = inference_detector(model, batch_images_path, cfg.data.val.batch_size)
        
        assert (len(batch_gt_bboxes) == 
                    len(batch_gt_labels) ==
                    len(batch_images_path) ==
                    len(batch_gt_masks) ==
                    len(batch_results))
        batch_conf_list = [batch_gt_masks, batch_gt_bboxes, batch_gt_labels, batch_results]
        
        confusion_dict = get_confusion_value(batch_conf_list, confusion_dict, 
                                                    iou_threshold, cfg.confidence_threshold, 
                                                    classes, 
                                                    func_mask_to_polygon)
    
    confusion_value_dict= confusion_dict   
    precision_recall_dict = compute_precision_recall(confusion_value_dict)
                            
    return precision_recall_dict
            
            
def compute_precision_recall(confusion_value_dict):
    for class_name, threshold_list in confusion_value_dict.items():
        for idx, threshold in enumerate(threshold_list):
            if threshold['num_pred'] == 0: precision = 0
            else: precision = threshold['num_true']/threshold['num_pred']
                
            recall = threshold['num_true']/threshold['num_gt']
            
            confusion_value_dict[class_name][idx]['recall'] = recall
            confusion_value_dict[class_name][idx]['precision'] = precision
        
            if recall == 0 and precision == 0: confusion_value_dict[class_name][idx]['F1_score'] =0
            else: confusion_value_dict[class_name][idx]['F1_score'] = 2*(precision*recall)/(precision+recall)
    
    return confusion_value_dict
                    
                    
def get_confusion_value(batch_conf_list, confusion_dict, 
                        iou_threshold, confidence_threshold, 
                        classes, func_mask_to_polygon):
    batch_gt_masks, batch_gt_bboxes, batch_gt_labels, batch_results = batch_conf_list
    
    for gt_mask, gt_bboxes, gt_labels, result in zip(
        batch_gt_masks, batch_gt_bboxes, batch_gt_labels, batch_results
        ):
        i_bboxes, i_labels, i_mask = parse_inferece_result(result)
        if iou_threshold[0] > 0:
            assert i_bboxes is not None and i_bboxes.shape[1] == 5
            scores = i_bboxes[:, -1]
            inds = scores > iou_threshold[0]
            i_bboxes = i_bboxes[inds, :]
            i_labels = i_labels[inds]
            if i_mask is not None:
                i_mask = i_mask[inds, ...]
        
        i_cores = i_bboxes[:, -1]      # [num_instance]
        i_bboxes = i_bboxes[:, :4]      # [num_instance, [x_min, y_min, x_max, y_max]]

        
        i_polygons = func_mask_to_polygon(i_mask)
        gt_polygons = func_mask_to_polygon(gt_mask.masks)
        
        infer_dict = dict(bboxes = i_bboxes,
                        polygons = i_polygons,
                        labels = i_labels,
                        score = i_cores)
        gt_dict = dict(bboxes = gt_bboxes,
                    polygons = gt_polygons,
                    labels = gt_labels)
        
        
        confusion_value_dict = get_num_posi_nega(iou_threshold, classes, 
                                            gt_dict, infer_dict, 
                                            confusion_dict, 
                                            confidence_threshold)  
    return confusion_value_dict 
                
                
def get_num_posi_nega(iou_threshold, classes, gt_dict, infer_dict, class_dict, confidence_threshold):
    num_pred = len(infer_dict['bboxes'])                    
    num_gt = len(gt_dict['bboxes'])
    for idx, threshold in enumerate(iou_threshold): 
        done_gt = []
        for i in range(num_pred):
            pred_class_name = classes[infer_dict['labels'][i]]
            for j in range(num_gt):
                gt_class_name = classes[gt_dict['labels'][j]]
                if i == 0:  class_dict[gt_class_name][idx]['num_gt'] +=1
                if j in done_gt: continue
                
                i_bboxes, gt_bboxes = infer_dict['bboxes'][i], gt_dict['bboxes'][j]
                iou = compute_iou(i_bboxes, gt_bboxes)
                
                if (iou > threshold and          
                    infer_dict['score'][i] > confidence_threshold ):
                    
                    # compute iou by sliced polygon 
                    i_polygons, gt_polygons = infer_dict['polygons'][i], gt_dict['polygons'][j]
                    i_xsort_bbox_list, i_ysort_bbox_list = get_divided_polygon(i_polygons, 3)
                    gt_xsort_bbox_list, gt_ysort_bbox_list = get_divided_polygon(gt_polygons, 3)
                
                    for i_xsort_bbox, gt_xsort_bbox in zip(i_xsort_bbox_list, gt_xsort_bbox_list):
                        if compute_iou(i_xsort_bbox, gt_xsort_bbox) < threshold:  continue
                    for i_ysort_bbox, gt_ysort_bbox in zip(i_ysort_bbox_list, gt_ysort_bbox_list):
                        if compute_iou(i_ysort_bbox, gt_ysort_bbox) < threshold:  continue
                    
                    class_dict[pred_class_name][idx]['num_pred'] +=1
                    
                    if pred_class_name == gt_class_name: 
                        class_dict[pred_class_name][idx]['num_true'] +=1
                    
                    done_gt.append(j)
    return class_dict