import argparse
from os import listdir, path
import random
import numpy as np
import tqdm


def convert(main_folder, output):
    ret = []
    class_count = 0    
    class_folders = listdir(main_folder)
    for  class_folder in tqdm.tqdm(class_folders,desc="traverse_images"):
        class_folder_path = path.join(main_folder, class_folder)
        file_list = listdir(class_folder_path)
        if len(file_list) <=0:
            continue        
        for img_name in file_list:
            image_path = path.join(main_folder,class_folder, img_name)            
            ret.append([image_path,class_count])
        class_count += 1
    
    dataset_info = "ids:%d images:%d" % (class_count,len(ret))  
    random.shuffle(ret)  
    open(output.split(".")[0]+"_dataset_info.txt", "w").write(dataset_info)
    np.savetxt(output, ret, delimiter=" ", fmt="%s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Folder with classes subfolders to a file to train."
    )
    parser.add_argument("--folder", "-f", help="Folder to convert.",type=str,default="/datasets/faceid/WebFace260M")
    parser.add_argument("--output", "-o", help="Output file.",type=str,default="/datasets/faceid/web42m.txt")

    args = parser.parse_args()

    convert(args.folder, args.output)