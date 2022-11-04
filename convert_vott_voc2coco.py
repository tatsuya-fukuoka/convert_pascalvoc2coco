import os
import json
import glob
import argparse

from tqdm import tqdm
import xml.etree.ElementTree as element_tree

import datetime
import shutil
import logging
from natsort import natsorted

def get_args():
    parser = argparse.ArgumentParser(
        description="Convert Pascal VOC annotation to COCO format.")

    parser.add_argument(
        "pascalvoc_dir",
        help="***PascalVOC-export",
        type=str,
    )
    
    parser.add_argument(
        "--bbox_offset",
        help="Bounding Box offset.",
        type=int,
        default=-1,
    )
    args = parser.parse_args()

    return args

def output_folder_make():
    today = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = f'{today}_COCO_format'
    os.mkdir(save_folder)
    output_each_folder = [f'{save_folder}/train2017', f'{save_folder}/val2017', f'{save_folder}/annotations']
    os.mkdir(output_each_folder[0])
    os.mkdir(output_each_folder[1])
    os.mkdir(output_each_folder[2])
    
    return output_each_folder


def image_name_list_get(path):
    with open(path) as f:
        each_filenames = [s.strip() for s in f.readlines()]
        
        return [filename.split(" ")[0] for filename in each_filenames]


def train_val_img_xml_path(txt_files):
    train_image_path = image_name_list_get(txt_files[0])
    val_image_path = image_name_list_get(txt_files[1])
    
    train_xml_files = natsorted([f"{filename.split('.')[0]}.xml" for filename in train_image_path])
    val_xml_files = natsorted([f"{filename.split('.')[0]}.xml" for filename in val_image_path])
    
    return [train_image_path, val_image_path], [train_xml_files, val_xml_files]


def image_copy(output_each_folder, images_dir, img_files, DATASETS_NAME):
    for i in range(len(img_files)):
        for image_name in tqdm(img_files[i], f"Copy Image: {DATASETS_NAME[i]}"):
            origin_file_posi = os.path.join(images_dir, image_name)
            copy_file_posi = os.path.join(output_each_folder[i], image_name)
            shutil.copyfile(origin_file_posi, copy_file_posi)
        logging.info(f'Image Copy {DATASETS_NAME[i]}: OK')


def get_categories(xml_files):
    classes_names = []

    # 全XMLのobjectからnameを取得
    for xml_file in xml_files:
        xml_file = os.path.join(xml_file)
        tree = element_tree.parse(xml_file)
        root = tree.getroot()

        for member in root.findall("object"):
            classes_names.append(member[0].text)

    # 重複を削除してソート
    classes_names = list(set(classes_names))
    classes_names.sort()
    # Dict形式に変換
    categories = {name: i + 1 for i, name in enumerate(classes_names)}

    return categories


def get_element(root, name, length=None):
    # 指定タグの値を取得
    vars = root.findall(name)

    # 長さチェック
    if length is not None:
        if len(vars) == 0:
            raise ValueError("Can not find %s in %s." % (name, root.tag))
        if length > 0 and len(vars) != length:
            raise ValueError(
                "The size of %s is supposed to be %d, but is %d." %
                (name, length, len(vars)))

        if length == 1:
            vars = vars[0]

    return vars


def convert_xml_to_json(
    xml_files, 
    categories, 
    xml_dir,
    bbox_offset=-1
    ):
    
    #COCOデータの生成
    json_dict = {
        "info":{},
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }
    
    #info data
    today = datetime.datetime.now()
    json_dict["info"]['year'] = int(today.strftime("%Y"))
    json_dict["info"]['version'] = "1.0"
    json_dict["info"]['description'] = "For object detection"
    json_dict["info"]['date_created'] = today.strftime("%Y-%m-%d")
    
    count_dict = {}
    
    bbox_id = 1
    image_id = 1
    for xml_file in tqdm(xml_files, "Convert XML to JSON"):
        xml_file = os.path.join(xml_dir,xml_file)
        # ルート要素取得
        tree = element_tree.parse(xml_file)
        root = tree.getroot()

        # 画像ファイル名取得
        path = get_element(root, "path")
        if len(path) == 1:
            filename = os.path.basename(path[0].text)
        elif len(path) == 0:
            filename = get_element(root, "filename", 1).text
        else:
            raise ValueError("%d paths found in %s" % (len(path), xml_file))

        # 画像情報取得
        size = get_element(root, "size", 1)
        width = int(get_element(size, "width", 1).text)
        height = int(get_element(size, "height", 1).text)

        # JSON Dict追加
        image_info = {
            "file_name": filename,
            "height": height,
            "width": width,
            "id": image_id,
        }
        json_dict["images"].append(image_info)

        # object情報取得
        for obj in get_element(root, "object"):
            # カテゴリー名取得
            category = get_element(obj, "name", 1).text
            # 初出のカテゴリー名の場合、リストに追加
            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
            # カテゴリー数カウント
            if category not in count_dict:
                count_dict[category] = 0
            else:
                count_dict[category] += 1

            # カテゴリーID取得
            category_id = categories[category]

            # バウンディングボックス情報取得
            bbox = get_element(obj, "bndbox", 1)
            xmin = int(float(get_element(bbox, "xmin", 1).text)) + bbox_offset
            ymin = int(float(get_element(bbox, "ymin", 1).text)) + bbox_offset
            xmax = int(float(get_element(bbox, "xmax", 1).text))
            ymax = int(float(get_element(bbox, "ymax", 1).text))
            assert xmax > xmin
            assert ymax > ymin
            bbox_width = abs(xmax - xmin)
            bbox_height = abs(ymax - ymin)

            # JSON Dict追加
            annotation_info = {
                "area": bbox_width * bbox_height,
                "iscrowd": 0,
                "image_id": image_id,
                "bbox": [xmin, ymin, bbox_width, bbox_height],
                "category_id": category_id,
                "id": bbox_id,
                "ignore": 0,
                "segmentation": [],
            }
            json_dict["annotations"].append(annotation_info)

            bbox_id = bbox_id + 1
            
        image_id += 1
        
    # カテゴリー情報
    for category_name, category_id in categories.items():
        category_info = {
            "supercategory": "none",
            "id": category_id,
            "name": category_name
        }
        json_dict["categories"].append(category_info)

    logging.info(f'Categories Count: {count_dict}')

    return json_dict


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s")
    
    pascalvoc_dir = args.pascalvoc_dir
    bbox_offset = args.bbox_offset
    
    images_dir = os.path.join(pascalvoc_dir,'JPEGImages')
    txt_dir = os.path.join(pascalvoc_dir,'ImageSets/Main')
    xml_dir = os.path.join(pascalvoc_dir,'Annotations')
    
    txt_files = glob.glob(os.path.join(txt_dir, "*.txt"))
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
    
    logging.info(f'Image Check Path: {images_dir}')
    logging.info(f'Txt Check Path: {txt_dir}')
    logging.info(f'Xml Check PAth: {xml_dir}')
    
    #指定したクラス名の定義
    categories = get_categories(xml_files)
    logging.info(f'Categories: {categories}')
    
    #Outputフォルダの作成
    output_each_folder = output_folder_make()
    logging.info('Output Folder Make: OK')
    
    OUTPUT_NAME = ['instances_train2017.json', 'instances_val2017.json']
    
    #train、valのimageファイル名、xmlファイル名を取得
    img_files, xml_files = train_val_img_xml_path(txt_files)
    
    DATASETS_NAME = ['train data', 'val data']
    
    #train2017,val2017へ画像のコピー
    image_copy(output_each_folder, images_dir, img_files, DATASETS_NAME)    
    
    for i in range(len(xml_files)):
        logging.info(f'Processing: {DATASETS_NAME[i]}')
        json_dict = convert_xml_to_json(
            xml_files[i],
            categories, 
            xml_dir
            )

        # json保存
        json_save_path = output_each_folder[2]
        save_path = os.path.join(json_save_path,OUTPUT_NAME[i])
        with open(save_path, "w") as fp:
            json_text = json.dumps(json_dict)
            fp.write(json_text)
            
        logging.info(f'Success: {DATASETS_NAME[i]}')


if __name__ == "__main__":
    main()
