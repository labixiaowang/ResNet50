import os
import glob
## 获取图片的路径和类别标签，并写入文本文件中
from PIL import Image

def is_image_valid(img_path):
    """检查图片是否可以成功打开，作为图片完整性的判断"""
    try:
        img = Image.open(img_path)
        img.verify()  # verify()方法可以帮助检测图像的完整性
    except (IOError, OSError, Image.DecompressionBombError):
        return False
    return True

def get_image_filenames_in_folder(folder_path, output_file_path,invalid_output_path,class_name):
    image_extensions = ['*.jpg']  # 可根据需要添加更多图片格式
    image_filenames = []
    unimage_filenames = []

    for ext in image_extensions:
        pattern = os.path.join(folder_path, ext)
        for img_file in glob.glob(pattern, recursive=True):
            if is_image_valid(img_file):
                image_filenames.append(img_file)
            else:
                unimage_filenames.append(img_file)
    # first_write = True
    # 将图片文件名写入到输出文件中
    with open(output_file_path, 'a') as output_file,open(invalid_output_path, 'a') as invalid_file:
        for filename in image_filenames:
            output_file.write(filename+","+str(class_name)+"\n")
        for filename in unimage_filenames:
            invalid_file.write(filename+","+str(class_name)+"\n")
current_dir = os.path.dirname(os.path.abspath(__file__))
# print(current_dir)
output_file_path = os.path.join(current_dir, 'valid.txt')
invalid_output_path = os.path.join(current_dir, 'invalid.txt')
# print(output_file_path)
# output_file_path = "/home/Resnet50/data.txt"  # 替换为你想要保存图片文件名的文本文件路径
for k,v in {"Harmful":0,"Kitchen":1,"Other":2,"Recyxclable":3}.items():
    input_folder_path = os.path.join("/home/Resnet50/DATA",k)
    get_image_filenames_in_folder(input_folder_path, output_file_path,invalid_output_path,v)
