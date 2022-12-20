db = dict(
    host=None, 
    port=None, 
    user='project-pipeline', 
    name='dataset',        
    charset='utf8',   

    table = dict(
        # name of `annotations data`` table     
        # annotations data: dataset made with labelme.exe
        anns = "ann_data",              
        
        # name of `image dataset`` table.     
        # image dataset: images for training or validation
        image_data = "image_data",      
        
        # name of `recoded dataset` table
        # recoded dataset: dataset that combines annotations data into a single file 
        dataset = "dataset",        
    )

    
)