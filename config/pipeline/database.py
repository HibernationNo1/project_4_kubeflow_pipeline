db = dict(
    host=None, 
    port=None, 
    user='project-pipeline', 
    name='ann_dataset',         # TODO rename
    charset='utf8',   
    
    

    table = dict(
        ann_dataset_name = "ann_dataset",           # name of annotations dataset table
        ann_dataset_schema = f"""
                              CREATE TABLE ann_dataset (
                              id INT UNSIGNED NOT NULL AUTO_INCREMENT,
                              json_name VARCHAR(200) NOT NULL,
                              image_name VARCHAR(200) NOT NULL,
                              category VARCHAR(20) NOT NULL,
                              ann_version VARCHAR(10) NOT NULL,
                              PRIMARY KEY(id)
                              );
                              """,
            # json_name: path of json file
            # image_name: path of image file
            # category: category of dataset
            # ann_version: version of annotation dataset
    
        image_dataset_name = "image_dataset",      # name of recoded dataset table
        image_dataset_schema = f"""
                                      CREATE TABLE image_dataset (
                                      id INT UNSIGNED NOT NULL AUTO_INCREMENT,
                                      dataset_purpose VARCHAR(10) NOT NULL,
                                      image_name VARCHAR(200) NOT NULL,
                                      category VARCHAR(20) NOT NULL,
                                      recode_version VARCHAR(10) NOT NULL,
                                      PRIMARY KEY(id)
                                      );
                                      """,
            # dataset_purpose: one of `train` and `val`
            # image_name: path of image data file for train or validation
            # category: category of dataset
            # recode_version: version of recoded dataset 
        
        train_dataset_name = "train_dataset",
        train_dataset_schema = f"""
                                      CREATE TABLE train_dataset (
                                      id INT UNSIGNED NOT NULL AUTO_INCREMENT,
                                      dataset_purpose VARCHAR(10) NOT NULL,
                                      category VARCHAR(20) NOT NULL,
                                      dataset_name VARCHAR(200) NOT NULL,
                                      recode_version VARCHAR(10) NOT NULL,
                                      PRIMARY KEY(id)
                                      );
                                      """
            # dataset_purpose: one of `train` and `val`
            # dataset_name: path of train_dataset or val_dataset file for train or validation
            # category: category of dataset
            # recode_version: version of recoded dataset 
    )


    
    
    
    
    
    
    
    
)