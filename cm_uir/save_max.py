import os
import blobfile as bf
import torch
def recursive_delete(path):
    for root, dirs, files in os.walk(path, topdown=False):  # 自底向上遍历
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)  # 删除文件
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            os.rmdir(dir_path)  # 删除空子文件夹
    os.rmdir(path)  # 删除最外层文件夹
def get_folders_with_prefix(path, prefix):
    return [f for f in os.listdir(path) 
            if os.path.isdir(os.path.join(path, f)) and f.startswith(prefix)]
def save_max(file_path):

    folders = get_folders_with_prefix(file_path, "step")
    filename ='test.txt'
    max = min = float(folders[0][18:][:8])
    max_file = folders[0]
    min_file = folders[0]
    if len(folders)<=10:
        return
    for filename in folders:
        if float(filename[18:][:8])>max:
            max_file = filename
            max = float(filename[18:][:8])
        if float(filename[18:][:8])<min:
            min_file =filename
            min = float(filename[18:][:8])
    if min_file ==folders[0] and max_file!=folders[0]:
        recursive_delete(bf.join(file_path,folders[0]))
    elif min_file != folders[0] and min!=max_file:
        recursive_delete(bf.join(file_path,min_file))
    else:
        return
    print(folders)
#save_max("/home/customer4/Documents/DEPTH_V5/models")