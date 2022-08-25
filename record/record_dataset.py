from re import L
from kfp.components import InputPath, OutputPath, create_component_from_func
from pipeline_config import Pipeline_Config
pl_cfg = Pipeline_Config


def record(cfg_path: InputPath("dict"),
           train_dataset_path: OutputPath("dict"),
           val_dataset_path: OutputPath("dict")):   
    """
    cfg_path : {wiorkspace}/inputs/cfg/data
    """    
    
    import json
    import subprocess
    import glob
    import os
    import time
    import numpy as np
    import PIL
    from PIL.Image import fromarray as fromarray
    from PIL.ImageDraw import Draw as Draw
    
    from pipeline_taeuk4958.utils.utils import NpEncoder
    from pipeline_taeuk4958.configs.config import load_config_in_pipeline
    from pipeline_taeuk4958.cloud.gs import get_client_secrets
    
    
    class Record_Dataset():
        """

        """
        def __init__(self, cfg, dataset_list, data_anns_config):
            self.cfg = cfg
            self.dataset_list = dataset_list
            self.data_anns_config = data_anns_config
            
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
                self.train_dataset['info']['ann_version']= self.data_anns_config['version']             # ann_version
                self.train_dataset['info']['ann_description']= self.data_anns_config['descriptoin']     # ann_description
                
                
            elif mode == "val":
                self.val_dataset['info']['description'] =  self.cfg.dataset.info.description
                self.val_dataset['info']['url']         =  self.cfg.dataset.info.url
                self.val_dataset['info']['version']     =  self.cfg.dataset.info.version
                self.val_dataset['info']['year']        =  self.cfg.dataset.info.year
                self.val_dataset['info']['contributor'] =  self.cfg.dataset.info.contributor
                self.val_dataset['info']['data_created']=  self.cfg.dataset.info.data_created
                self.val_dataset['info']['for_what']= "val"
                self.val_dataset['info']['ann_version']= self.data_anns_config['version']           # ann_version
                self.val_dataset['info']['ann_description']= self.data_anns_config['descriptoin']   # ann_description 
                

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
                            if shape['label'] in self.cfg.dataset.valid_object:
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
    

    def get_anns(cfg):
        # set client secret for dvc pull
        client_secrets_path = os.path.join(os.getcwd(), cfg.gs.client_secrets)
        gs_secret = get_client_secrets()
        
        json.dump(gs_secret, open(client_secrets_path, "w"), indent=4)   
        remote_bucket_command = f"dvc remote add -d -f bikes gs://{cfg.gs.ann_bucket_name}"
        credentials_command = f"dvc remote modify --local bikes credentialpath '{client_secrets_path}'" 
      
        subprocess.call([remote_bucket_command], shell=True)
        subprocess.call([credentials_command], shell=True)
        subprocess.call(["dvc pull"], shell=True)           # download dataset from GS by dvc
        
        # get annotations
        anns_dir = os.path.join(os.getcwd(), cfg.dataset.anns_dir)
        anns_list = glob.glob(f"{anns_dir}/*.json")    
        if len(anns_list)==0 : raise OSError("Failed download dataset!!")
        print(f"\n number of annotations : {len(anns_list)} \n")
        
        anns_config_path = os.path.join(os.getcwd(), cfg.dataset.anns_config_path)
        with open(anns_config_path, "r", encoding='utf-8') as f:
            anns_config = json.load(f)    
            
        return anns_list, anns_config
     
    
    if __name__=="__main__":                
        cfg = load_config_in_pipeline(cfg_path) 
        
        ## download dataset from google cloud stroage bucket by dvc
        dvc_path = os.path.join(os.getcwd(), 'dataset.dvc')             # check file exist (downloaded from git repo with git clone )
        if not os.path.isfile(dvc_path): raise OSError(f"{dvc_path}")    
        
        anns_list, anns_config = get_anns(cfg)      
        
        
        print(f"anns_config : {anns_config}")
        ##  get dataset
        labelme_instance = Record_Dataset(cfg, anns_list, anns_config)
        train_dataset, val_dataset = labelme_instance.get_dataset()

        json.dump(train_dataset, open(train_dataset_path, "w"), indent=4, cls = NpEncoder)
        json.dump(val_dataset, open(val_dataset_path, "w"), indent=4, cls = NpEncoder)
        
        
print(f"record base_image : {pl_cfg.RECORD_IMAGE}")  
record_op = create_component_from_func(func = record,
                                        base_image = pl_cfg.RECORD_IMAGE,
                                        output_component_file= pl_cfg.RECORD_COM_FILE)