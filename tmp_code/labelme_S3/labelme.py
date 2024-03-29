from kfp.components import InputPath, OutputPath, create_component_from_func
from config import LABELME_IMAGE


def lebelme(args : dict, 
            cfg_path: InputPath("dict"),
            train_dataset_path: OutputPath("dict"),
            val_dataset_path: OutputPath("dict")):   
    """
    cfg_path : {wiorkspace}/inputs/cfg/data
    """    
    import os
    import glob
    import time
    import json
    import numpy as np

    import PIL
    from PIL.Image import fromarray as fromarray
    from PIL.ImageDraw import Draw as Draw

    
    
    class labelme_custom():
        """

        """
        def __init__(self, cfg, dataset_list):
            self.cfg = cfg
            self.dataset_list = dataset_list
            
            self.train_dataset = dict(info = {}, 
                                licenses = [],
                                images = [], annotations = [], categories = [],
                                classes = None)
            self.val_dataset = dict(info = {}, 
                                licenses = [],
                                images = [], annotations = [], categories = [],
                                classes = None)
            
            self.object_names = []
            
            self.dataset_dir_path = None
            self.annnotation_dir_path = None
            
            # json_file = self.train_dataset
            # print(f"json_file['info'].keys() : {json_file['info'].keys()} \n")
            # print(f"json_file['images'][0].keys() : {json_file['images'][0].keys()} \n")
            # print(f"json_file['categories'][0].keys() : {json_file['categories'][0].keys()} \n")
            # print(f"json_file['annotations'][0].keys() : {json_file['annotations'][0].keys()} \n")
            # print(f"json_file['annotations'][0]['image_id'] : {json_file['annotations'][0]['image_id']}, {type(json_file['annotations'][0]['image_id'])}")
            
            self.data_transfer()
            

   
        def get_dataset(self):
            return self.train_dataset, self.val_dataset
  
        
        def data_transfer(self):
            labelme_json_list = self.dataset_list
            
            if self.cfg.options.proportion_val == 0:
                val_split_num = len(labelme_json_list) + 100000
            else:
                val_image_num = len(labelme_json_list) * self.cfg.options.proportion_val     
                if val_image_num == 0 :
                    val_split_num = 1
                else : val_split_num = int(len(labelme_json_list)/val_image_num)
        
            
            print(f" part_1 : info")
            self.get_info("train")
            self.get_info("val")
            
            print(f"\n part_2 : licenses")
            self.get_licenses("train")
            self.get_licenses('val')
            
            print(f"\n part_3 : images")
            self.get_images(labelme_json_list, val_split_num)
            
            print(f"\n part_4 : annotations")
            self.get_annotations(labelme_json_list, val_split_num)
            
            print(f"\n part_5 : categories")
            self.get_categories("train")   
            self.get_categories("val") 

        def get_info(self, mode) : 
            if mode == "train":
                self.train_dataset['info']['description'] = self.cfg.dataset.info.description
                self.train_dataset['info']['url']         =  self.cfg.dataset.info.url
                self.train_dataset['info']['version']     =  self.cfg.dataset.info.version
                self.train_dataset['info']['year']        =  self.cfg.dataset.info.year
                self.train_dataset['info']['contributor'] =  self.cfg.dataset.info.contributor
                self.train_dataset['info']['data_created']=  self.cfg.dataset.info.data_created
                self.train_dataset['info']['for_what']= "train"
            elif mode == "val":
                self.val_dataset['info']['description'] =  self.cfg.dataset.info.description
                self.val_dataset['info']['url']         =  self.cfg.dataset.info.url
                self.val_dataset['info']['version']     =  self.cfg.dataset.info.version
                self.val_dataset['info']['year']        =  self.cfg.dataset.info.year
                self.val_dataset['info']['contributor'] =  self.cfg.dataset.info.contributor
                self.val_dataset['info']['data_created']=  self.cfg.dataset.info.data_created
                self.val_dataset['info']['for_what']= "va;"
                

        def get_licenses(self, mode):            
            if self.cfg.dataset.info.licenses is not None:
                tmp_dict = dict(url = self.cfg.dataset.info.licenses.url,
                                id = self.cfg.dataset.info.licenses.id,
                                name = self.cfg.dataset.info.licenses.name)   
                if mode == "train":
                    self.train_dataset['licenses'].append(tmp_dict)  # original coco dataset have several license
                elif mode == "val":
                    self.val_dataset['licenses'].append(tmp_dict) 
            else: 
                pass  
        
        def get_categories(self, mode):
            for i, object_name in enumerate(self.object_names):
                tmp_categories_dict = {}
                tmp_categories_dict['supercategory'] = object_name                          # str
                tmp_categories_dict['id'] = self.object_names.index(object_name)            # int
                tmp_categories_dict['name'] = object_name                                    # str
                if mode == "train" :
                    self.train_dataset['categories'].append(tmp_categories_dict)
                elif mode == "val" :
                    self.val_dataset['categories'].append(tmp_categories_dict)

            if mode == "train" :
                self.train_dataset['classes'] = self.object_names
            elif mode == "val" :
                self.val_dataset['classes'] = self.object_names
            
            
        def get_annotations(self, labelme_json_list, val_split_num):
            id_count = 1
            for i, json_file in enumerate(labelme_json_list):
                with open(json_file, "r") as fp:
                    data = json.load(fp) 

                    image_height, image_width = data["imageHeight"], data["imageWidth"]
    
                    for shape in data['shapes']:    # shape == 1 thing object.    
                        if shape['label'] not in self.object_names:
                            if shape['label'] in self.cfg.json.valid_object:
                                self.object_names.append(shape['label'])
                            else: 
                                if self.cfg.options.only_val_obj: raise KeyError(f"{shape['label']} is not valid object.")   
                                else: continue


                        tmp_annotations_dict = {}
                        if shape['shape_type'] == "polygon":  
                                                    
                            contour = np.array(shape['points'])
                            tmp_segmentation = []
                            points = list(np.asarray(contour).flatten())
                            for point in points:
                                tmp_segmentation.append(round(point, 2))
                            tmp_annotations_dict['segmentation'] = [tmp_segmentation]
                            mask = self.polygons_to_mask([image_height, image_width], contour)
                            x = contour[:, 0]
                            y = contour[:, 1]
                            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                            tmp_annotations_dict["area"] = float(area)
                            
                            tmp_annotations_dict['iscrowd'] = 0     # choise in [0, 1]  TODO : add when using included [iscrowd == 1] dataset
                            tmp_annotations_dict['image_id'] = i+1
                            tmp_annotations_dict['bbox'] = list(map(float, self.mask2box(mask)))
                            tmp_annotations_dict['category_id'] = self.object_names.index(shape['label'])       # int, same to category_id
                            tmp_annotations_dict['id'] = id_count
                            id_count +=1
                            
                            
                        else : continue     # TODO : add when using dataset not type of segmentation

                        if i % val_split_num == 0:
                            self.val_dataset['annotations'].append(tmp_annotations_dict)
                        else:
                            self.train_dataset['annotations'].append(tmp_annotations_dict)                    


        def polygons_to_mask(self, img_shape, polygons):
            mask = np.zeros(img_shape, dtype=np.uint8)
            
            mask = fromarray(mask)
            xy = list(map(tuple, polygons))  
            Draw(mask).polygon(xy=xy, outline=1, fill=1)
            mask = np.array(mask, dtype=bool)
            return mask
        
        
        def mask2box(self, mask):
            index = np.argwhere(mask == 1)   
            
            rows = index[:, 0]
            clos = index[:, 1]       

            left_top_r = np.min(rows)  # y
            left_top_c = np.min(clos)  # x

            right_bottom_r = np.max(rows)
            right_bottom_c = np.max(clos)

            
            return [
                left_top_c,
                left_top_r,
                right_bottom_c - left_top_c,
                right_bottom_r - left_top_r,
            ]

        def get_images(self, labelme_json_list, val_split_num):
            yyyy_mm_dd_hh_mm = time.strftime('%Y-%m-%d', time.localtime(time.time())) \
                        + " "+ str(time.localtime(time.time()).tm_hour) \
                        + ":" + str(time.localtime(time.time()).tm_min)
            
            for i, json_file in enumerate(labelme_json_list):                
                tmp_images_dict = {}
                with open(json_file, "r") as fp:
                    data = json.load(fp) 
                    
                    tmp_images_dict['license'] = f"{len(self.train_dataset['licenses'])}"  # just '1' because license is only one
                    tmp_images_dict['file_name'] = data['imagePath']
                    tmp_images_dict['coco_url'] = " "                       # str
                    tmp_images_dict['height'] = data["imageHeight"]         # int
                    tmp_images_dict['width'] = data["imageWidth"]           # int
                    tmp_images_dict['date_captured'] = f"{yyyy_mm_dd_hh_mm}"     # str   
                    tmp_images_dict['flickr_url'] = " "                     # str
                    tmp_images_dict['id'] = i+1                                 # non-duplicate int value
                
                if i % val_split_num == 0:
                    self.val_dataset["images"].append(tmp_images_dict)
                else:
                    self.train_dataset["images"].append(tmp_images_dict)

    import json
    import subprocess
    import glob
    from pipeline_taeuk4958.utils.utils import NpEncoder
    from pipeline_taeuk4958.configs.config import Config
    
    ## load config
    cfg_pyformat_path = cfg_path + ".py"        # cfg_pyformat_path : {wiorkspace}/inputs/cfg/data.py
                                                # can't command 'mv' 
    # change format to .py
    with open(cfg_path, "r") as f:
        data = f.read()
    with open(cfg_pyformat_path, "w") as f:
        f.write(data)       # 
    f.close()

    cfg = Config.fromfile(cfg_pyformat_path)    # cfg_pyformat_path : must be .py format   
    
    ## download dataset from s3 bucket by dvc
    access_key_id = f"dvc remote modify --local storage access_key_id {args['access_key_id']}"
    secret_access_key = f"dvc remote modify --local storage secret_access_key {args['secret_access_key']}"
    subprocess.call([access_key_id], shell=True)
    subprocess.call([secret_access_key], shell=True)
    subprocess.call(["dvc pull"], shell=True)                           # dvc pull
    data_dir = os.path.join(os.getcwd(), cfg.dir_info.dataset_dir)
    dataset_list = glob.glob(f"{data_dir}/*.json")
    print(f"\n data_dir : {data_dir}\n ")
    print(f"\n number of annotations : {len(dataset_list)} \n")
    
    # get dataset
    labelme_instance = labelme_custom(cfg, dataset_list)
    train_dataset, val_dataset = labelme_instance.get_dataset()
    
    json.dump(train_dataset, open(train_dataset_path, "w"), indent=4, cls = NpEncoder)
    json.dump(val_dataset, open(val_dataset_path, "w"), indent=4, cls = NpEncoder)
    
    

lebelme_op = create_component_from_func(func = lebelme,
                                        base_image = LABELME_IMAGE,
                                        output_component_file="lebelme.component.yaml")