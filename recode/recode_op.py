from kfp.components import create_component_from_func
from pipeline_base_image_cfg import Base_Image_Cfg
base_image = Base_Image_Cfg()

def check_status(cfg : dict, cfg_flag:dict) :
    
    import os, os.path as osp
    import numpy as np
    import json
    import subprocess
    import pymysql
    import datetime
    import time    
    import pandas as pd
    import shutil
    
    from PIL.Image import fromarray as fromarray
    from PIL.ImageDraw import Draw as Draw    
    from git import Repo
    
    from hibernation_no1.configs.utils import change_to_tuple, NpEncoder
    from hibernation_no1.configs.config import Config
    from hibernation_no1.utils.utils import get_environ
    from hibernation_no1.cloud.google.storage import get_client_secrets
    from hibernation_no1.cloud.google.dvc import dvc_pull, dvc_push
    from hibernation_no1.database.mysql import check_table_exist
    
    
    TODAY = str(datetime.date.today())
    
    class Record_Dataset():
        """

        """
        def __init__(self, cfg, image_list, json_list, dataset_dir_path, pre_processing = False):
            """ Parsing the json list and recode the train dataset and validation dataset
                to one file json format each.
                
                save recoded images for training

            Args:
                cfg (Config): config 
                image_list (list): List of file jpg format.
                json_list (list): List of file json format.
                                Contains annotations information of label.
                dataset_dir_path: path of dataset.     `category/ann/version`
                pre_processing (optional): whether apply pre-processing to dataset.
                                           It can be data augmentation.
                                             
            """
            self.cfg = cfg
            self.image_list = image_list   
            self.json_list = json_list   
            self.dataset_dir_path = dataset_dir_path
            self.pre_processing = pre_processing
            
            self.train_dataset = dict(info = {}, 
                                licenses = [],
                                images = [], annotations = [], categories = [],
                                classes = None)
            self.val_dataset = dict(info = {}, 
                                licenses = [],
                                images = [], annotations = [], categories = [],
                                classes = None)
            
            self.object_names = []
            
            self.recode_dataset_path = osp.join(os.getcwd(), 
                                          cfg.dvc.category,
                                          cfg.dvc.recode.name,
                                          cfg.dvc.recode.version)
            os.makedirs(self.recode_dataset_path, exist_ok=True)
            
            self.data_transfer()
            
            
        def data_transfer(self):            
            if self.cfg.recode.options.proportion_val == 0:        # TODO cfg.recode.options
                val_split_num = len(self.json_list) + 100000
            else:
                val_image_num = len(self.json_list) * self.cfg.recode.options.proportion_val     
                if val_image_num == 0 :
                    val_split_num = 1
                else : val_split_num = int(len(self.json_list)/val_image_num)
        
            
            print(f" Part_1: info")
            self.get_info("train")
            self.get_info("val")
            
            print(f"\n Part_2: licenses")
            self.get_licenses("train")
            self.get_licenses('val')
            
            print(f"\n Part_3: images")
            self.get_images(val_split_num)
            
            print(f"\n Part_4: annotations")
            self.get_annotations(val_split_num)
            
            print(f"\n Part_5: categories")
            self.get_categories("train")   
            self.get_categories("val") 
            
            if self.pre_processing:
                print(f"\n Part_optional: apply pre-processing to dataset.")
                self.run_pre_processing()
            
            print(f"\n Part_6: save recoded dataset.")
            self.save_json()
            
            # if self.pre_processing, save pre-processed image in run_pre_processing()
            if not self.pre_processing:     
                print(f"\n Part_7: save images for taining")
                self.save_image()


            
        def get_info(self, mode) : 
            if mode == "train":
                self.train_dataset['info']['description'] = self.cfg.recode.info.description
                self.train_dataset['info']['url']         =  self.cfg.recode.info.url
                self.train_dataset['info']['version']     =  self.cfg.recode.info.version
                self.train_dataset['info']['year']        =  f"{TODAY.split('-')[0]}"
                self.train_dataset['info']['contributor'] =  self.cfg.recode.info.contributor
                self.train_dataset['info']['data_created']=  f"{TODAY.split('-')[0]}/{TODAY.split('-')[1]}/{TODAY.split('-')[2]}"
                self.train_dataset['info']['for_what']= "train"
                
                
            elif mode == "val":
                self.val_dataset['info']['description'] =  self.cfg.recode.info.description
                self.val_dataset['info']['url']         =  self.cfg.recode.info.url
                self.val_dataset['info']['version']     =  self.cfg.recode.info.version
                self.val_dataset['info']['year']        =  f"{TODAY.split('-')[0]}"
                self.val_dataset['info']['contributor'] =  self.cfg.recode.info.contributor
                self.val_dataset['info']['data_created']=  f"{TODAY.split('-')[0]}/{TODAY.split('-')[1]}/{TODAY.split('-')[2]}"
                self.val_dataset['info']['for_what']= "val"
            
            
        def get_licenses(self, mode):                  
            if self.cfg.recode.info.licenses is not None:
                tmp_dict = dict(url = self.cfg.recode.info.licenses.url,
                                id = self.cfg.recode.info.licenses.id,
                                name = self.cfg.recode.info.licenses.name)   
                if mode == "train":
                    # why list?: original coco dataset have several license 
                    self.train_dataset['licenses'].append(tmp_dict)  
                elif mode == "val":
                    self.val_dataset['licenses'].append(tmp_dict) 
            else: 
                pass
            
            
        def get_images(self, val_split_num):
            yyyy_mm_dd_hh_mm = time.strftime('%Y-%m-%d', time.localtime(time.time())) \
                        + " "+ str(time.localtime(time.time()).tm_hour) \
                        + ":" + str(time.localtime(time.time()).tm_min)
            
            for i, json_file in enumerate(self.json_list):                
                tmp_images_dict = {}
                with open(json_file, "r") as fp:
                    data = json.load(fp) 
                    
                    tmp_images_dict['license'] = f"{len(self.train_dataset['licenses'])}"  # just '1' because license is only one
                    tmp_images_dict['file_name'] = data['imagePath']
                    tmp_images_dict['coco_url'] = " "                           # str
                    tmp_images_dict['height'] = data["imageHeight"]             # int
                    tmp_images_dict['width'] = data["imageWidth"]               # int
                    tmp_images_dict['date_captured'] = f"{yyyy_mm_dd_hh_mm}"    # str   
                    tmp_images_dict['flickr_url'] = " "                         # str
                    tmp_images_dict['id'] = i+1                                 # non-duplicate int value
                
                if i % val_split_num == 0:
                    self.val_dataset["images"].append(tmp_images_dict)
                else:
                    self.train_dataset["images"].append(tmp_images_dict)                
        
        
        def get_annotations(self, val_split_num):
            id_count = 1
            for i, json_file in enumerate(self.json_list):
                with open(json_file, "r") as fp:
                    data = json.load(fp) 

                    image_height, image_width = data["imageHeight"], data["imageWidth"]
    
                    for shape in data['shapes']:    # shape == 1 thing object.    
                        if shape['label'] not in self.object_names:
                            if shape['label'] in self.cfg.recode.valid_object:
                                self.object_names.append(shape['label'])
                            else: 
                                if self.cfg.recode.options.only_val_obj: raise KeyError(f"{shape['label']} is not valid object.")   
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
            
        def get_dataset(self):
            return self.train_dataset, self.val_dataset
        
        # TODO
        def run_pre_processing(self):       
            """
                apply pre-processing to images and save to self.recode_dataset_path
            """
            
            print(f"\n Part_optional: save pre-processed images for taining")

            # save the image with a modified name to distinguish if the image has been preprocessed.
            # self.saved_image_list = []
            pass
                
        
        def save_json(self):             
            self.train_dataset_path = osp.join(self.recode_dataset_path, cfg.recode.train_dataset)
            self.val_dataset_path = osp.join(self.recode_dataset_path, cfg.recode.val_dataset)
            json.dump(self.train_dataset, open(self.train_dataset_path, "w"), indent=4, cls = NpEncoder)
            json.dump(self.val_dataset, open(self.val_dataset_path, "w"), indent=4, cls = NpEncoder)


        def save_image(self):
            # save image with original name
            # no need to distinguish if the image preprocessed.
            self.saved_image_list_train = []
            self.saved_image_list_val = []
            
            after_image_list = []
            before_image_list = []
                 
            def _save_image(image_info_list, purpose):
                for file_info in image_info_list:
                    image_name = file_info["file_name"]
                    
                    after_image_list.append(osp.join(self.recode_dataset_path, image_name))
                    before_image_list.append(osp.join(self.dataset_dir_path, image_name))
                    
                    if purpose == "train":
                        self.saved_image_list_train.append(image_name)
                    elif purpose == "val":
                        self.saved_image_list_val.append(image_name)
            
                for befor_image_path, after_image_path in zip(before_image_list, after_image_list):
                    shutil.copyfile(befor_image_path, after_image_path)
            
            _save_image(self.train_dataset['images'], "train")
            _save_image(self.val_dataset['images'], "val")
               
               
    
    def select_ann_data(cursor, cfg):
        # select ann data
        select_sql = f"SELECT * FROM {cfg.db.table.anns} WHERE ann_version = '{cfg.dvc.ann.version}';"
        num_results = cursor.execute(select_sql)
        assert num_results != 0, f" `{cfg.dvc.ann.version}` version of annotations dataset is not being inserted into database"\
           f"\n     DB: {cfg.db.name},      table: {cfg.db.table.anns}"
    
        ann_version_df = pd.read_sql(select_sql, database)

        json_list = []
        image_list = []
        for category, ann_version, json_name, image_name in zip(ann_version_df.category,
                                                                ann_version_df.ann_version,
                                                                ann_version_df.json_name,
                                                                ann_version_df.image_name):
            json_list.append(osp.join(os.getcwd(), 
                                      category, 
                                      cfg.dvc.ann.name, 
                                      ann_version,
                                      json_name))
            image_list.append(osp.join(os.getcwd(), 
                                      category, 
                                      cfg.dvc.ann.name, 
                                      ann_version,
                                      image_name))
        
        return image_list, json_list
    
    
    # insert dataset to database
    def insert_recode_data(cfg, cursor, saved_image_list, purpose):
        if purpose == "train":
            recode_file = cfg.recode.train_dataset
        elif purpose == "val":
            recode_file = cfg.recode.val_dataset            
            
        for image_name in saved_image_list:
            insert_sql = f"INSERT INTO {cfg.db.table.image_data}"\
                        f"(dataset_purpose, image_name, recode_file, category, recode_version)"\
                        f"VALUES('{purpose}', '{image_name}', '{recode_file}',"\
                        f"'{cfg.dvc.category}', '{cfg.dvc.recode.version}');" 
            
            cursor.execute(insert_sql)
            
        
        select_sql = f"SELECT * FROM {cfg.db.table.image_data} "\
                     f"WHERE recode_version = '{cfg.dvc.recode.version}' AND dataset_purpose = '{purpose}';"
        num_results = cursor.execute(select_sql)
        assert num_results == len(saved_image_list),\
            f" `{cfg.dvc.recode.version}` version of recode dataset is not being inserted into database"\
            f"\n     DB: {cfg.db.name},      table: {cfg.db.table.image_data},    purpose: {purpose}"\
            f"      num_results: {num_results}      len(saved_image_list): {len(saved_image_list)}"
            
        # insert dataset (.json fomat)
        insert_sql = f"INSERT INTO {cfg.db.table.dataset}"\
                         f"(dataset_purpose, category, recode_file, recode_version)"\
                         f"VALUES('{purpose}', '{cfg.dvc.category}', '{recode_file}',"\
                         f"'{cfg.dvc.recode.version}');" 
        cursor.execute(insert_sql)
    
    
    
    
    def git_push(cfg):
        repo = Repo(os.getcwd())
        # Path for git add must not include $(pwd) path.
        repo.git.add(f"{cfg.dvc.category}/{cfg.dvc.recode.name}/.gitignore")        
        repo.git.add(f"{cfg.dvc.category}/{cfg.dvc.recode.name}/{cfg.dvc.recode.version}.dvc")
        repo.index.commit(f"{cfg.dvc.category}:: {cfg.dvc.recode.name}:: {cfg.dvc.recode.version}")
        subprocess.call([f"git push {cfg.git.remote} {cfg.git.branch}"], shell=True)
        ##  repo.remote is not work in container
        # origin = repo.remote(name= 'origin')
        # origin.push()
            
            
     
    if __name__=="__main__":  
        cfg = change_to_tuple(cfg, cfg_flag)
        cfg = Config(cfg)

        target_dataset = osp.join(os.getcwd(), cfg.dvc.category,
                                               cfg.dvc.ann.name,
                                               cfg.dvc.ann.version)
        
        dvc_cfg = dict(remote = cfg.dvc.remote,
                       bucket_name = cfg.gs.bucket.ann,
                       client_secrets = get_client_secrets(),
                       dataset_name = target_dataset)
        dataset_dir_path = dvc_pull(**dvc_cfg)
        
        database = pymysql.connect(host=get_environ(cfg.db, 'host'), 
                        port=int(get_environ(cfg.db, 'port')), 
                        user=cfg.db.user, 
                        passwd=os.environ['password'], 
                        database=cfg.db.name, 
                        charset=cfg.db.charset)
    
        cursor = database.cursor() 

        check_table_exist(cursor, cfg.db.table)
        
        image_list, json_list = select_ann_data(cursor, cfg)
        
        recode_dataset = Record_Dataset(cfg, image_list, json_list, dataset_dir_path)
        
        result_dir = recode_dataset.recode_dataset_path
        
        assert osp.isdir(result_dir), f"Path: {result_dir} is not exist!!"
        assert len(os.listdir(result_dir)) != 0, f"Images not saved!  \nPath: {result_dir}"
        assert osp.isfile(recode_dataset.train_dataset_path),\
            f"Paht: {recode_dataset.train_dataset_path}, is not exist!!"

        insert_recode_data(cfg, cursor, recode_dataset.saved_image_list_train, "train")
        insert_recode_data(cfg, cursor, recode_dataset.saved_image_list_val, "val")
       
        
        dvc_push(cfg.dvc.remote, cfg.gs.bucket.recoded, get_client_secrets(), 
                 result_dir, cfg.dvc.recode.version)
        
        git_push(cfg)
        
        database.commit()
        database.close()
        print(f"completion")
    

print(f"recode_op base_image : {base_image.recode}")
check_status_op = create_component_from_func(func = check_status,
                                        base_image = base_image.recode,
                                        output_component_file= base_image.recode_cp)

