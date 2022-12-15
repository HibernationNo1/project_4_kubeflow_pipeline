from genericpath import exists
from kfp.components import OutputPath, create_component_from_func

def check_status(cfg : dict, cfg_flag:dict) :
    
    import os, os.path as osp
    import numpy as np
    import json
    import subprocess
    import pymysql
    from PIL.Image import fromarray as fromarray
    from PIL.ImageDraw import Draw as Draw    
    import datetime
    import time    
    import glob
    import warnings
    
    from hibernation_no1.configs.utils import change_to_tuple
    from hibernation_no1.configs.config import Config
    from hibernation_no1.utils.utils import get_environ
    from hibernation_no1.cloud.gs import get_client_secrets
    
    
    TODAY = str(datetime.date.today())
    
    class Record_Dataset():
        """

        """
        def __init__(self, cfg, dataset):
            self.cfg = cfg
            self.dataset = dataset     # list of json
            
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
            
            self.data_transfer()
            
        def data_transfer(self):            
            if self.cfg.recode.options.proportion_val == 0:        # TODO cfg.recode.options
                val_split_num = len(self.dataset) + 100000
            else:
                val_image_num = len(self.dataset) * self.cfg.recode.options.proportion_val     
                if val_image_num == 0 :
                    val_split_num = 1
                else : val_split_num = int(len(self.dataset)/val_image_num)
        
            
            print(f" part_1 : info")
            self.get_info("train")
            self.get_info("val")
            
            print(f"\n part_2 : licenses")
            self.get_licenses("train")
            self.get_licenses('val')
            
            print(f"\n part_3 : images")
            self.get_images(val_split_num)
            
            print(f"\n part_4 : annotations")
            self.get_annotations(val_split_num)
            
            print(f"\n part_5 : categories")
            self.get_categories("train")   
            self.get_categories("val") 
            
            
            
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
            
            for i, json_file in enumerate(self.dataset):                
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
            for i, json_file in enumerate(self.dataset):
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

    
    def dvc_gs_credentials(remote: str, bucket_name: str, client_secrets: dict):
        """ access google cloud with credentials

        Args:
            remote (str): name of remote of dvc  
            bucket_name (str): bucket name of google storage
            client_secrets (dict): credentials info for access google storage
        """
        
        client_secrets_path = osp.join(os.getcwd(), "client_secrets.json")
        json.dump(client_secrets, open(client_secrets_path, "w"), indent=4)
        
        remote_bucket_command = f"dvc remote add -d -f {remote} gs://{bucket_name}"
        credentials_command = f"dvc remote modify --local {remote} credentialpath {client_secrets_path}"   
        
        subprocess.call([remote_bucket_command], shell=True)
        subprocess.call([credentials_command], shell=True)
        

    def dvg_pull(remote: str, bucket_name: str, client_secrets: dict, dataset_name: str):
        """ run dvc pull from google cloud storage

        Args:
            remote (str): name of remote of dvc
            bucket_name (str): bucket name of google storage
            client_secrets (dict): credentials info to access google storage
            dataset_name (str): name of folder where located dataset(images)

        Returns:
            _type_: _description_
        """
        
        dvc_path = osp.join(os.getcwd(), f'{dataset_name}.dvc')          # check file exist (downloaded from git repo by git clone)
        assert os.path.isfile(dvc_path), f"Path: {dvc_path} is not exist!" 

        dvc_gs_credentials(remote, bucket_name, client_secrets)
        
        subprocess.call(["dvc pull"], shell=True)           # download dataset from GS by dvc 
        
        dataset_dir_path = osp.join(os.getcwd(), dataset_name)
        assert osp.isdir(dataset_dir_path), f"Directory: {dataset_dir_path} is not exist!"
        
        return dataset_dir_path
    
    
    
    def create_table(cursor, table_name, schema):
        cursor.execute(f"SHOW TABLES")
        fetchs = cursor.fetchall()
        if len(fetchs) !=0:
            tables = fetchs[0]
            if table_name not in tables:
                print(f"create table: {table_name}")
                cursor.execute(schema)
            else:
                print(f"table: {table_name} is already exist!")
        else:
            print(f"create table: {table_name}")
            cursor.execute(schema)
                
                
            
    if __name__=="__main__":  
        cfg = change_to_tuple(cfg, cfg_flag)
        cfg = Config(cfg)
        
        dvc_cfg = dict(remote = cfg.dvc.remote,
                       bucket_name = cfg.gs.ann_bucket_name,
                       client_secrets = get_client_secrets(),
                       dataset_name = cfg.dvc.dataset_cate)
        dataset_dir_path = dvg_pull(**dvc_cfg)
            
        json_list = glob.glob(f"{dataset_dir_path}/*.json")
        image_list = glob.glob(f"{dataset_dir_path}/*.jpg")
        if len(json_list) != len(image_list): 
            warnings.warn(f"number of images and json files are not same!!  \n"\
                          f"number of images {len(image_list)},  "\
                          f"number of json files : {len(json_list)}")
        
        
        
        
        
        
        labelme_instance = Record_Dataset(cfg, json_list)
        train_dataset, val_dataset = labelme_instance.get_dataset()
        
  
        
        database = pymysql.connect(host=get_environ(cfg.db, 'host'), 
                        port=int(get_environ(cfg.db, 'port')), 
                        user=cfg.db.user, 
                        passwd=os.environ['password'], 
                        database=cfg.db.name, 
                        charset=cfg.db.charset)
    
        cursor = database.cursor() 

        create_table(cursor, cfg.db.table.ann_dataset_name, cfg.db.table.ann_dataset_schema)
        
        num_results = cursor.execute(f"SELECT * FROM {cfg.db.table.ann_dataset_name} WHERE ann_version = '{cfg.dvc.version}'")
        assert num_results == 0, f"ann version: {cfg.dvc.version} has been stored in DB!!  "\
           f"DB: [{cfg.db.name}],         table: [{cfg.db.table.ann_dataset_name}]"
    
        
        image_list = []
        for json_path in json_list:
            image_path = osp.basename(json_path).split(".")[0] + ".jpg"
            image_list.append(image_path)
            
            
        # insert annn dataset to database
        for img_json_path in zip(image_list, json_list):
            image_path, json_path = img_json_path
            image_name, json_name = os.path.basename(image_path), os.path.basename(json_path)
            insert_sql = f"INSERT INTO {cfg.db.table.ann_dataset_name} "\
                         f"(json_name, image_name, catecory, ann_version) "\
                         f"VALUES('{json_name}', '{image_name}', '{cfg.db.table.ann_dataset_name}', '{cfg.dvc.version}');"
            cursor.execute(insert_sql)     
    
   
    
    
    
# print("set_config base_image : hibernation4958/check:0.2")
check_status_op = create_component_from_func(func = check_status,
                                        base_image = "hibernation4958/check_re:0.2",
                                        output_component_file= "check")

