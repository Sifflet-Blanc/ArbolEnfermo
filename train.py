from ultralytics import YOLO
import xml.etree.ElementTree as ET
import pandas as pd
import os
import re
import random as rd
import shutil

def purge_data(folder):
    for file in os.listdir(folder):
        if file != ".gitingore":
            os.remove(folder+'/'+file)


def parse_bndbox(bndbox_str):
    xmin = re.search(r'<xmin>(.*?)</xmin>', bndbox_str).group(1)
    ymin = re.search(r'<ymin>(.*?)</ymin>', bndbox_str).group(1)
    xmax = re.search(r'<xmax>(.*?)</xmax>', bndbox_str).group(1)
    ymax = re.search(r'<ymax>(.*?)</ymax>', bndbox_str).group(1)
    return pd.Series({'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})

def damage_to_class(damage):
    if damage == "H":
        return 0
    elif damage == "LD":
        return 1
    elif damage == "HD":
        return 2
    else:
        return 3

print("Clean des dossiers...")
if not os.path.exists("data/images"):
    os.makedirs("data/images")
if not os.path.exists("data/labels"):
    os.makedirs("data/labels")
if not os.path.exists("data/images/train"):
    os.makedirs("data/images/train")
if not os.path.exists("data/images/val"):
    os.makedirs("data/images/val")
if not os.path.exists("data/labels/train"):
    os.makedirs("data/labels/train")
if not os.path.exists("data/labels/val"):
    os.makedirs("data/labels/val")

purge_data("data/images/train")
purge_data("data/images/val")
purge_data("data/labels/train")
purge_data("data/labels/val")


print("Compilation du dataset...")
folder = "data/Data_Set_Larch_Casebearer/Bebehojd_20190527/Annotations/"

base_path = "data/Data_Set_Larch_Casebearer/"
for drone_survey in os.listdir("data/Data_Set_Larch_Casebearer/"):
    if drone_survey.endswith("20190527"):
        folder = base_path + drone_survey + "/Annotations/"
        for anotation in os.listdir(folder):
            img_path = base_path + drone_survey + '/Images/'+anotation.replace(".xml", "")+'.JPG'
            if os.path.exists(img_path):
                tree = ET.parse(folder + anotation)
                root = tree.getroot()
                
                # Prepare a list to store the data
                data = []
            
                width = int(root.find('size/width').text)
                height = int(root.find('size/height').text)
                
                # Iterate over each 'object' in the XML
                for obj in root.findall('object'):
                    row = {
                        'tree': obj.find('tree').text if obj.find('tree') is not None else None,
                        'damage': obj.find('damage').text if obj.find('damage') is not None else None,
                        'pose': obj.find('pose').text if obj.find('pose') is not None else None,
                        'truncated': obj.find('truncated').text if obj.find('truncated') is not None else None,
                        'difficult': obj.find('difficult').text if obj.find('difficult') is not None else None,
                        'xmin': obj.find('bndbox/xmin').text if obj.find('bndbox/xmin') is not None else None,
                        'ymin': obj.find('bndbox/ymin').text if obj.find('bndbox/ymin') is not None else None,
                        'xmax': obj.find('bndbox/xmax').text if obj.find('bndbox/xmax') is not None else None,
                        'ymax': obj.find('bndbox/ymax').text if obj.find('bndbox/ymax') is not None else None,
                    }
                    data.append(row)
                
                df = pd.DataFrame(data)
    
                data_set = "train"
                if rd.random() < 0.2:
                    data_set = "val"
    
                shutil.copy2(img_path, 'data/images/'+data_set+"/"+anotation.replace(".xml", "")+'.jpg')  
    
                if 'damage' in list(df.columns.values):
                    df["class"] = df['damage'].apply(damage_to_class)
                    df["xmin"] = pd.to_numeric(df["xmin"])
                    df["xmax"] = pd.to_numeric(df["xmax"])
                    df["ymin"] = pd.to_numeric(df["ymin"])
                    df["ymax"] = pd.to_numeric(df["ymax"])
                    
                    df["width"] = (df["xmax"] - df["xmin"]) / width
                    df["height"] = (df["ymax"] - df["ymin"]) / height
                    df["x_center"] = df["xmin"] / width + df["width"]
                    df["y_center"] = df["ymin"] / height + df["height"]
                
                    df = df[["class", "x_center", "y_center", "width", "height"]]
                
                    df.to_csv(r'data/labels/'+data_set+"/"+anotation.replace(".xml", "")+'.txt', header=None, index=None, sep=' ', mode='w')


print("Training...")
model = YOLO("yolo12n.pt")
results = model.train(data="data/data.yaml", 
    epochs=100, 
    imgsz=1500,
    project="/home/arbremalade/ArbolEnfermo/runs/",
    name="train")
model.export(imgsz=(1500, 1500), save=True, name='train')