db = dict(
    host=None, 
    port=None, 
    user='hibernation', 
    name='pipeline_database',        
    charset='utf8',   

    table = dict(
        # name of `annotations data`` table     
        # annotations data: dataset made with labelme.exe
        anns = "ann_data",              
        
        # name of `image dataset`` table.     
        # image dataset: images for training or validation
        image_data = "image_data",      
        
        # name of `recordd dataset` table
        # recordd dataset: dataset that combines annotations data into a single file 
        dataset = "train_data",        
    )

    
)