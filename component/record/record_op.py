from kfp.components import create_component_from_func, OutputPath
from pipeline_base_config import Base_Image_cfg
base_image = Base_Image_cfg()

def record(cfg : dict, input_run_flag: dict, 
           run_flag_path: OutputPath("dict")):   # Error when using variable name 'run_flag_path'
 
    import git
    import os, os.path as osp
    import numpy as np
    import json
    import subprocess
    import pymysql
    import datetime
    import time    
    import pandas as pd
    import shutil
    import warnings
    import glob
    import sys
    from PIL.Image import fromarray as fromarray
    from PIL.ImageDraw import Draw as Draw    
    from git import Repo

 
    WORKSPACE = dict(component_volume = cfg['path']['component_volume'],       # pvc volume path on component container
                     local_volume = cfg['path'].get('local_volume', None),     # pvc volume path on local
                     docker_volume = cfg['path']['docker_volume'],     # volume path on katib container
                     work = cfg['path']['work_space']
                     )    

    # set package path to 'import {custom_package}'
    if __name__=="component.record.record_op":
        docker_volume = f"/{WORKSPACE['docker_volume']}"
        
        if WORKSPACE['local_volume'] is not None:
            local_module_path = osp.join('/opt/local-path-provisioner', WORKSPACE['local_volume']) 
        else:
            local_module_path = osp.join(os.getcwd(), cfg['git']['package_repo'])       
        
        if osp.isdir(local_module_path):
            PACKAGE_PATH = os.getcwd()
            print(f"    Run `record` locally")
            
        elif osp.isdir(docker_volume):
            PACKAGE_PATH = docker_volume
            print(f"    Run `record` in docker container")
            
        else:
            raise OSError(f"Paths '{docker_volume}' and '{local_module_path}' do not exist!")

    if __name__=="__main__":    
        assert osp.isdir(WORKSPACE['work']), f"The path '{WORKSPACE['work']}' is not exist!"
        assert osp.isdir(WORKSPACE['component_volume']), f"The path '{WORKSPACE['component_volume']}' is not exist!"
        print(f"    Run `record` in component for pipeline")
        PACKAGE_PATH = WORKSPACE['component_volume']
        # for import sub_module
        package_repo_path = osp.join(WORKSPACE['component_volume'], cfg['git']['package_repo'])        
        if not osp.isdir(package_repo_path):
            print(f" git clone 'sub_module' to {package_repo_path}")
            
            Repo.clone_from(f"git@github.com:HibernationNo1/{cfg['git']['package_repo']}.git", package_repo_path)
     
    sys.path.append(PACKAGE_PATH) 
    
    
    from sub_module.configs.pipeline import dict2Config 
    from sub_module.configs.utils import NpEncoder
    from sub_module.configs.config import Config
    from sub_module.utils.utils import get_environ
    from sub_module.cloud.google.storage import get_client_secrets
    from sub_module.cloud.google.dvc import dvc_pull, dvc_add, dvc_push
    from sub_module.database.mysql import check_table_exist
    
    TODAY = str(datetime.date.today())
    
    def main(cfg, in_pipeline = False):           
        if in_pipeline:
            git_repo = git_clone_dataset(cfg)
                        
            target_dataset = osp.join(os.getcwd(), 
                                      cfg.dvc.ann.dir,
                                      cfg.dvc.category)
            
            dvc_cfg = dict(remote = cfg.dvc.ann.remote,
                           bucket_name = cfg.dvc.ann.gs_bucket,
                           client_secrets = get_client_secrets(),
                           data_root = target_dataset)
            data_root = dvc_pull(**dvc_cfg)
            
            database = pymysql.connect(host=get_environ(cfg.db, 'host'), 
                            port=int(get_environ(cfg.db, 'port')), 
                            user=cfg.db.user, 
                            passwd=os.environ['password'], 
                            database=cfg.db.name, 
                            charset=cfg.db.charset)
        
            cursor = database.cursor() 
            check_table_exist(cursor, cfg.db.table)
        
            image_list, json_list = select_ann_data(cfg, cursor, database)
        else:
            data_root = osp.join(os.getcwd(), cfg.ann_data_root)
            if not osp.isdir(data_root): raise OSError(f"The path dose not exist!  \n path: {data_root}")
            image_list = glob.glob(data_root +'/*.jpg')
            json_list = glob.glob(data_root +'/*.json')
            if len(image_list) == 0 : raise OSError(f"Images dose not exist!  \n dir path: {data_root}")
       
        record_dataset_cfg = dict(
            cfg = cfg,
            image_list = image_list,
            json_list = json_list,
            data_root = data_root,
            in_pipeline = in_pipeline
        )
        record_dataset = Record_Dataset(**record_dataset_cfg)
        
        result_dir = record_dataset.record_dataset_path
        
        assert osp.isdir(result_dir), f"Path: {result_dir} is not exist!!"
        assert len(os.listdir(result_dir)) != 0, f"Images not saved!  \nPath: {result_dir}"
        assert osp.isfile(record_dataset.train_dataset_path),\
            f"Paht: {record_dataset.train_dataset_path}, is not exist!!"
       
        if in_pipeline:
            insert_record_data(cfg, cursor, record_dataset.saved_image_list_train, "train")
            insert_record_data(cfg, cursor, record_dataset.saved_image_list_val, "val")
            
            dvc_add(target_dir = result_dir)
            git_push(cfg, git_repo)
            dvc_push(remote = cfg.dvc.record.remote,
                    bucket_name = cfg.dvc.record.gs_bucket,
                    client_secrets = get_client_secrets())
            
            print(f"Run DataBase commit")
            database.commit()
            database.close()
        print(f"completed.")
        
        
    
    class Record_Dataset():
        """

        """
        def __init__(self, 
                     cfg, 
                     image_list, 
                     json_list, 
                     data_root, 
                     pre_processing = False, in_pipeline = False):
            """ Parsing the json list and record the train dataset and validation dataset
                to one file json format each.
                
                save record images for training

            Args:
                cfg (Config): config 
                image_list (list): List of file jpg format.
                json_list (list): List of file json format.
                                Contains annotations information of label.
                data_root: path of dataset.     
                pre_processing (optional): whether apply pre-processing to dataset.
                                           It can be data augmentation.
                                             
            """
            self.cfg = cfg
            self.image_list = image_list   
            self.json_list = json_list   
            if len(self.image_list)!=len(self.json_list):
                raise ValueError(f"Number of images and json files is not the same!"
                                 f"\n images:{len(self.image_list)}, json files: {len(self.json_list)}")
            self.data_root = data_root
            self.pre_processing = pre_processing
            self.in_pipeline = in_pipeline
            
            self.train_dataset = dict(info = {}, 
                                licenses = [],
                                images = [], annotations = [], categories = [],
                                classes = None)
            self.val_dataset = dict(info = {}, 
                                licenses = [],
                                images = [], annotations = [], categories = [],
                                classes = None)
            
            self.object_names = []
            
            if self.in_pipeline:
                self.record_dataset_path = osp.join(os.getcwd(), 
                                                    cfg.dvc.record.dir,
                                                    cfg.dvc.category)
            else:
                self.record_dataset_path = osp.join(os.getcwd(),
                                                    cfg.record_result)
        
            os.makedirs(self.record_dataset_path, exist_ok=True)
            print(f" category: {cfg.dvc.category}")
            print(f" Number of image: {len(image_list)}")
            self.data_transfer()
            
            
        def data_transfer(self):            
            if self.cfg.record.options.proportion_val == 0:        # TODO cfg.record.options
                val_split_num = len(self.json_list) + 100000
            else:
                val_image_num = len(self.json_list) * self.cfg.record.options.proportion_val     
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
            
            print(f"\n Part_6: save record dataset.")
            self.save_json()
            
            # if self.pre_processing, save pre-processed image in run_pre_processing()
            if not self.pre_processing:     
                print(f"\n Part_7: save images for training")
                self.save_image()


            
        def get_info(self, mode) : 
            if mode == "train":
                self.train_dataset['info']['description'] = self.cfg.record.info.description
                self.train_dataset['info']['url']         =  self.cfg.record.info.url
                self.train_dataset['info']['version']     =  self.cfg.record.info.version
                self.train_dataset['info']['year']        =  f"{TODAY.split('-')[0]}"
                self.train_dataset['info']['contributor'] =  self.cfg.record.info.contributor
                self.train_dataset['info']['data_created']=  f"{TODAY.split('-')[0]}/{TODAY.split('-')[1]}/{TODAY.split('-')[2]}"
                self.train_dataset['info']['for_what']= "train"
                
                
            elif mode == "val":
                self.val_dataset['info']['description'] =  self.cfg.record.info.description
                self.val_dataset['info']['url']         =  self.cfg.record.info.url
                self.val_dataset['info']['version']     =  self.cfg.record.info.version
                self.val_dataset['info']['year']        =  f"{TODAY.split('-')[0]}"
                self.val_dataset['info']['contributor'] =  self.cfg.record.info.contributor
                self.val_dataset['info']['data_created']=  f"{TODAY.split('-')[0]}/{TODAY.split('-')[1]}/{TODAY.split('-')[2]}"
                self.val_dataset['info']['for_what']= "val"
            
            
        def get_licenses(self, mode):                  
            if self.cfg.record.info.licenses is not None:
                tmp_dict = dict(url = self.cfg.record.info.licenses.url,
                                id = self.cfg.record.info.licenses.id,
                                name = self.cfg.record.info.licenses.name)   
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
            unvalid_object = list()
            for i, json_file in enumerate(self.json_list):
                with open(json_file, "r") as fp:
                    data = json.load(fp) 

                    image_height, image_width = data["imageHeight"], data["imageWidth"]
    
                    for shape in data['shapes']:    # shape == 1 thing object.   
                        object_name = shape['label'] 
                        if object_name not in self.object_names:
                            if object_name in self.cfg.record.valid_object:
                                self.object_names.append(object_name)
                            else: 
                                if object_name not in unvalid_object:
                                    warnings.warn(f"{object_name} is not valid object.", UserWarning)   
                                    unvalid_object.append(object_name)
                                continue
                                


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
                apply pre-processing to images and save to self.record_dataset_path
            """
            
            print(f"\n Part_optional: save pre-processed images for taining")

            # save the image with a modified name to distinguish if the image has been preprocessed.
            # self.saved_image_list = []
            pass
                
        
        def save_json(self):             
            self.train_dataset_path = osp.join(self.record_dataset_path, self.cfg.record.train_dataset)
            self.val_dataset_path = osp.join(self.record_dataset_path, self.cfg.record.val_dataset)
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
                    
            
                    after_image_list.append(osp.join(self.record_dataset_path, image_name))
                    before_image_list.append(osp.join(self.data_root, image_name))
                    
                    if purpose == "train":
                        self.saved_image_list_train.append(image_name)
                    elif purpose == "val":
                        self.saved_image_list_val.append(image_name)

                
                # copy images from ann dataset to training dataset
                for befor_image_path, after_image_path in zip(before_image_list, after_image_list):
                    shutil.copyfile(befor_image_path, after_image_path)
            
            _save_image(self.train_dataset['images'], "train")
            _save_image(self.val_dataset['images'], "val")
               
               
    
    def select_ann_data(cfg, cursor, database):
        assert cfg.dvc.ann.version == cfg.git.dataset.db_ann_version, \
            f"The config `cfg.dvc.ann.version` and `cfg.git.dataset.db_ann_version` must be same."\
            f"\b cfg.dvc.ann.version: {cfg.dvc.ann.version}"\
            f"cfg.git.dataset.db_ann_version: {cfg.git.dataset.db_ann_version}"
        # select ann data
        select_sql = f"SELECT * FROM {cfg.db.table.anns} WHERE ann_version = '{cfg.dvc.ann.version}';"
        num_results = cursor.execute(select_sql)
        assert num_results != 0, f" `{cfg.dvc.ann.version}` version of annotations dataset is not being inserted into database"\
           f"\n     DB: {cfg.db.name},      table: {cfg.db.table.anns}"
    
        ann_version_df = pd.read_sql(select_sql, database)

        json_list = []
        image_list = []
        for category, json_name, image_name in zip(ann_version_df.category,
                                                   ann_version_df.json_name,
                                                   ann_version_df.image_name):
            json_list.append(osp.join(os.getcwd(), 
                                      cfg.dvc.ann.dir,
                                      category, 
                                      json_name))
            image_list.append(osp.join(os.getcwd(), 
                                      cfg.dvc.ann.dir,
                                      category,
                                      image_name))
        
        return image_list, json_list
    
    
    # insert dataset to database
    def insert_record_data(cfg, cursor, saved_image_list, purpose):
        
        if purpose == "train":
            record_file = cfg.record.train_dataset
        elif purpose == "val":
            record_file = cfg.record.val_dataset            
        
        print(f"\nInserte data to DB.  Name of table: {cfg.db.table.image_data}, for {purpose}")
        print(f"    Columns: `dataset_purpose`, `image_name`, `record_file`, `category`, `train_version`")
        for image_name in saved_image_list:
            insert_sql = f"INSERT INTO {cfg.db.table.image_data}"\
                        f"(dataset_purpose, image_name, record_file, category, train_version)"\
                        f"VALUES('{purpose}', '{image_name}', '{record_file}',"\
                        f"'{cfg.dvc.category}', '{cfg.dvc.record.version}');" 
            
            cursor.execute(insert_sql)
                    
        select_sql = f"SELECT * FROM {cfg.db.table.image_data} "\
                     f"WHERE train_version = '{cfg.dvc.record.version}' AND dataset_purpose = '{purpose}';"
        num_results = cursor.execute(select_sql)
        assert num_results == len(saved_image_list),\
            f" `{cfg.dvc.record.version}` version of record dataset is not being inserted into database"\
            f"\n     DB: {cfg.db.name},      table: {cfg.db.table.image_data},    purpose: {purpose}"\
            f"      num_results: {num_results}      len(saved_image_list): {len(saved_image_list)}"

        print(f"Inserte data to DB.  Name of table: {cfg.db.table.dataset}")
        print(f"    Columns: `dataset_purpose`, `category`, `record_file`, `train_version`")
        # insert dataset (.json fomat)
        insert_sql = f"INSERT INTO {cfg.db.table.dataset}"\
                         f"(dataset_purpose, category, record_file, train_version)"\
                         f"VALUES('{purpose}', '{cfg.dvc.category}', '{record_file}',"\
                         f"'{cfg.dvc.record.version}');" 
        cursor.execute(insert_sql)
    
    
    
    
    def git_push(cfg, repo):
        repo.git.checkout(f"{cfg.git.branch.dataset_repo}")
        
        # Check where HEAD is located
        print(f"Run `$ git branch`")
        for branch in repo.branches:
            if str(branch)==cfg.git.branch.dataset_repo:
                print(f"  * {branch}(activate)")
            else:
                print(f"    {branch}")

        assert cfg.git.branch.dataset_repo == str(repo.active_branch), \
            f"The branch of the set HEAD and active branch is different.\n"\
            f"Set branch HEAD: {cfg.git.branch.dataset_repo}, active branch: {repo.active_branch} "

        # Check working directory status
        untracked_files = repo.untracked_files
        for file_ in untracked_files:
            file_path = osp.join(os.getcwd(), str(file_))
            if not osp.isfile(file_path):
                raise OSError(f"file_path: {file_path} is not exist!")
            print(f"Run `$ git add {file_path}`")      
            repo.git.add(f"{file_path}") 
        
        # Check staging status
        staged_files = [item.a_path for item in repo.index.diff("HEAD")]
        assert len(staged_files)!=0, f"There are no staged file.  check command: `git add`"
        
        # Run git commit with massage
        commit_msg = f"docs: tag: {cfg.dvc.category}_{cfg.dvc.record.dir}_{cfg.dvc.record.version}"
        print(f"Run `$ git commit -m '{commit_msg}'`")
        repo.index.commit(f"{commit_msg}")
        
        # Run git push
        git_remote_name = cfg.git.get("remote", "origin")
        print(f"Run `$ git push {git_remote_name} {cfg.git.branch.dataset_repo}`")
        origin = repo.remote(git_remote_name)
        origin.push(cfg.git.branch.dataset_repo)        
    
    def git_clone_dataset(cfg):
        repo_path = osp.join(WORKSPACE['work'], cfg.git.dataset.repo)
        if osp.isdir(repo_path):
            if len(os.listdir(repo_path)) != 0:
                # ----
                # repo = Repo(osp.join(WORKSPACE['work'], cfg.git.dataset_repo))
                # origin = repo.remotes.origin  
                # repo.config_writer().set_value("user", "email", "taeuk4958@gmail.com").release()
                # repo.config_writer().set_value("user", "name", "HibernationNo1").release()
                
                # import subprocess       # this command working only shell, not gitpython.
                # safe_directory_str = f"git config --global --add safe.directory {repo_path}"
                # subprocess.call([safe_directory_str], shell=True)      

                # # raise: stderr: 'fatal: could not read Username for 'https://github.com': No such device or address'  
                # origin.pull()   
                # ----
                
                # ssh key not working when trying git pull with gitpython
                # delete all file cloned before and git clone again  
                import shutil
                shutil.rmtree(repo_path, ignore_errors=True)
                os.makedirs(repo_path, exist_ok=True)

        try:
            print(f"Run `$ git clone git@github.com:HibernationNo1/{cfg.git.dataset.repo}.git`")
            repo = Repo.clone_from(f'git@github.com:HibernationNo1/{cfg.git.dataset.repo}.git', os.getcwd())  
        except:
            print(f"Can't git clone with ssh!")
            print(f"Run `$ git clone https://github.com/HibernationNo1/{cfg.git.dataset.repo}.git`")
            repo = Repo.clone_from(f"https://github.com/HibernationNo1/{cfg.git.dataset.repo}.git", os.getcwd())

        remote_tags = repo.git.ls_remote("--tags").split("\n")
        tag_names = [tag.split('/')[-1] for tag in remote_tags if tag]
        if cfg.git.dataset.tag not in tag_names:
            raise KeyError(f"The `{cfg.git.dataset.tag}` is not exist in tags of repository `Hibernation/{cfg.git.dataset.repo}`")

        # checkout HEAD to tag
        repo.git.checkout(cfg.git.dataset.tag)
        
        return repo


    if __name__=="component.record.record_op":    
        print(f"    Run record")
        cfg = Config(cfg)
        main(cfg)
        
        
    if __name__=="__main__":
        if 'record' in input_run_flag['pipeline_run_flag']:
            cfg = dict2Config(cfg, key_name ='flag_list2tuple')    
            
            main(cfg, in_pipeline = True)
        else:
            print(f"Pass component: record")
        
        return json.dump(input_run_flag, open(run_flag_path, "w"), indent=4)
        
            

record_op = create_component_from_func(func = record,
                                        base_image = base_image.record,
                                        output_component_file= base_image.record_cp)

