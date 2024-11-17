import pandas as pd
import cv2
import os
from ImageTools import *

def parse_annotations(label_string):
    bboxes = []
    labels = label_string.split()
    for i in range(0, len(labels), 5):
        unicode_char = labels[i]
        x = int(labels[i + 1])
        y = int(labels[i + 2])
        w = int(labels[i + 3])
        h = int(labels[i + 4])
        unicode_pt = labels[i]
        bboxes.append(((x, y, w, h), unicode_pt))
    return bboxes


def get_kuzushiji_labels(annotations_file):
    df = pd.read_csv(annotations_file)
    labels = set()
    for _, row in df.iterrows():
      label_string = row["labels"]
      tokens = label_string.split()
      for i in range(0, len(tokens), 5):
          unicode_char = tokens[i]
          labels.add(unicode_char)
    return labels
	
def get_kuzushiji_stats(annotations_file):
    df = pd.read_csv(annotations_file)
    stats = {}
    for _, row in df.iterrows():
      label_string = row["labels"]
      tokens = label_string.split()
      for i in range(0, len(tokens), 5):
          unicode_char = tokens[i]
          if unicode_char in stats:
            stats[unicode_char] +=1
          else :
            stats[unicode_char] = 1
    return stats
    
    
def crop_images(annotations_file, img_src_dir, img_dst_dir, data_filter = None):
    df = pd.read_csv(annotations_file)
    for _, row in df.iterrows():
        image_id = row['image_id']
        p = os.path.join(img_src_dir, image_id + '.jpg')
        if not os.path.exist(p):
            return
        img = cv2.imread(p)
        bboxes = parse_annotations(row['labels'])
        for bbox, lab in bboxes:
            x1 = min(max(0 , bbox[0]), img.shape[1])
            y1 = min(max(0, bbox[1]), img.shape[0])
            x2 = min(x1 +  bbox[2], img.shape[1])
            y2 = min(y1 + bbox[3], img.shape[0])
            cropped_img = img[y1 : y2, x1 : x2]
            if len(cropped_img.shape) == 2 or cropped_img.shape[2] == 1: 
                cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2RGB)
            else :
                cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            image_name = f'{image_id}_{lab}_{x1}_{y1}_{x2 - x1}_{y2-y1}.png'
            image_dst_path = os.path.join(img_dst_dir, image_name)
            cv2.imwrite(image_dst_path, cropped_img)
            
def crop_and_save(image_id, bbox, label, img_src_dir, img_dst_dir):
    p = os.path.join(img_src_dir, image_id + '.jpg')
    if not os.path.exists(p):
        return
    img = cv2.imread(p)
    x1 = min(max(0 , bbox[0]), img.shape[1])
    y1 = min(max(0, bbox[1]), img.shape[0])
    x2 = min(x1 +  bbox[2], img.shape[1])
    y2 = min(y1 + bbox[3], img.shape[0])
    cropped_img = img[y1 : y2, x1 : x2]
    if len(cropped_img.shape) == 2 or cropped_img.shape[2] == 1: 
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2RGB)
    else :
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    image_name = f'{label}_{image_id}_{x1}_{y1}_{x2 - x1}_{y2-y1}.png'
    image_dst_path = os.path.join(img_dst_dir, image_name)
    #print(f'saving image {image_dst_path}')
    result = cv2.imwrite(image_dst_path, cropped_img)
    if not result:
      print(f'could not save image {image_dst_path}')
    return image_dst_path
        
        
def crop_images_with_ctx(annotations_file, img_src_dir, img_dst_dir, radius = 2, data_filter = None):
    df = pd.read_csv(annotations_file)
    for _, row in df.iterrows():
        image_id = row['image_id']
        p = os.path.join(img_src_dir, image_id + '.jpg')
        if not os.path.exist(p):
            return
        img = cv2.imread(p)
        bboxes = parse_annotations(row['labels'])
        for bbox, label in bboxes:
            x1 = max(0, bbox[0] - radius * bbox[2])
            x2 = min(img.shape[1], bbox[0] + (radius+1) * bbox[2])
            y1 = max(0, bbox[1] - radius * bbox[3])
            y2 = min(img.shape[0], bbox[1] + (radius+1) * bbox[3])
            cropped_img = img[y1 : y2, x1 : x2]
            if len(cropped_img.shape) == 2 or cropped_img.shape[2] == 1: 
                cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2RGB)
            else :
                cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            image_name = f'{image_id}_label_{x1}_{y1}_{x2 - x1}_{y2-y1}_ctx.png'
            image_dst_path = os.path.join(img_dst_dir, image_name)
            cv2.imwrite(image_dst_path, cropped_img)
            
def crop_with_ctx_and_save(image_id, bbox, label, img_src_dir, img_dst_dir, radius=2):
        p = os.path.join(img_src_dir, image_id + '.jpg')
        if not os.path.exists(p):
            return
        img = cv2.imread(p)
        x1 = max(0, bbox[0] - radius * bbox[2])
        x2 = min(img.shape[1], bbox[0] + (radius+1) * bbox[2])
        y1 = max(0, bbox[1] - radius * bbox[3])
        y2 = min(img.shape[0], bbox[1] + (radius+1) * bbox[3])
        cropped_img = img[y1 : y2, x1 : x2]
        if len(cropped_img.shape) == 2 or cropped_img.shape[2] == 1: 
            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2RGB)
        else :
            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        image_name = f'{label}_{image_id}_{bbox[0]}_{bbox[1]}_{bbox[2] - bbox[0]}_{bbox[3]-bbox[1]}_ctx_{radius}.png'
        image_dst_path = os.path.join(img_dst_dir, image_name)
        cv2.imwrite(image_dst_path, cropped_img)
        return image_dst_path