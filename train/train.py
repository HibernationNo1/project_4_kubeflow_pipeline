import argparse
import os
import glob

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data-path", type=str,
                       help = "path of dir where dataset.json, original_image dir is located.")
    
    args = parser.parse_args()
    
    
    train_dataset = os.path.join(os.getcwd(), args.data_path)
    dataset = os.path.join(train_dataset, 'train_dataset.json')
    train_image_dir_path = os.path.join(train_dataset, 'train_images')
    
    print(f"train_dataset : {train_dataset}")
    file_list = glob.glob(train_dataset)
    print(f"file_list : {file_list}")
    
   	
    if os.path.isfile(dataset):
        print("not")
    else:
        print("success")
        