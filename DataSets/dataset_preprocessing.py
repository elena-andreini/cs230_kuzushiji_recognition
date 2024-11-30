import cv2
import os
import shutil
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
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
           


def crop_boxes_and_save(image_id, bboxes_and_labels, img_src_dir, img_dst_dir):
    p = os.path.join(img_src_dir, image_id + '.jpg')
    if not os.path.exists(p):
        return
    img = cv2.imread(p)    
    image_dst_paths = []
    for item  in bboxes_and_labels:
        bbox = item[:-1][0]
        label = item[-1]
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
        else :
          image_dst_paths.append([label, image_dst_path])
    return image_dst_paths
    
def crop_boxes_ctx_and_save(image_id, bboxes_and_labels, img_src_dir, img_dst_dir, radius=2):
    p = os.path.join(img_src_dir, image_id + '.jpg')
    if not os.path.exists(p):
        return
    img = cv2.imread(p)    
    image_dst_paths = []
    for item  in bboxes_and_labels:
        bbox = item[:-1][0]
        label = item[-1]
        x1 = max(0, bbox[0] - radius * bbox[2])
        x2 = min(img.shape[1], bbox[0] + (radius+1) * bbox[2])
        y1 = max(0, bbox[1] - radius * bbox[3])
        y2 = min(img.shape[0], bbox[1] + (radius+1) * bbox[3])
        cropped_img = img[y1 : y2, x1 : x2]
        if len(cropped_img.shape) == 2 or cropped_img.shape[2] == 1: 
            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2RGB)
        else :
            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        image_name = f'{label}_{image_id}_{x1}_{y1}_{x2 - x1}_{y2-y1}_ctx.png'
        image_dst_path = os.path.join(img_dst_dir, image_name)
        result = cv2.imwrite(image_dst_path, cropped_img)
        if not result:
          print(f'could not save image {image_dst_path}')
        else :
          image_dst_paths.append([label, image_dst_path])
    return image_dst_paths
           
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
        image_name = f'{label}_{image_id}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}_ctx_{radius}.png'
        image_dst_path = os.path.join(img_dst_dir, image_name)
        cv2.imwrite(image_dst_path, cropped_img)
        return image_dst_path
        
        
        
def copy_dataset(annotations_file, src_dir, dst_dir, fraction= 1.0, index_range = None):
    df = pd.read_csv(annotations_file)
    if fraction < 1.0:
        df =df.sample(frac=fraction).reset_index(drop=True)
    data_range = None
    if index_range is not None:
        if isinstance(index_range, int) :
            data_range = (0, min(max_index, df.shape[0]))
        elif isinstance(index_range, tuple):
            data_range = index_range
    if data_range is not None:
        df = df[index_range[0]:index_range[1]]
    # Create the destination directory if it doesn't exist
    os.makedirs(dst_dir, exist_ok=True)
    print(f'copying {df.shape[0]} images')
    # Iterate over the dataframe and copy files
    for file_name in df['image_id']:
         src_path = os.path.join(src_dir, file_name+'.jpg')
         dest_path = os.path.join(dst_dir, file_name+'.jpg')
         shutil.copy(src_path, dest_path)
        
        
def generate_classification_dataset(df, classes, full_images_path, char_images_dst_path, context_images_dst_path, dst_annotations_path):
    """
    Generates the training data for the classification model
    cropping patches from full page images. 
    Only training examples belonging the classes are generated
    """
    data = []
    for _, row in df.iterrows():
        image_id = row['image_id']
        aa = parse_annotations(row['labels'])
        for a in aa:
          if a[1] not in classes:
            continue
          im1 = crop_and_save(image_id, a[0], a[1],
                                              full_images_path,
                                              char_images_dst_path)
          im2 = crop_with_ctx_and_save(image_id, a[0], a[1],
                                              full_images_path,
                                              context_images_dst_path)

          data.append([a[1], im1, im2])
          

        proc_df = pd.DataFrame(data, columns=['label', 'char_path', 'ctx_path'])
        proc_df.to_csv(dst_annotations_path)


def generate_char_dataset(df,  full_images_path, char_images_dst_path, context_images_dst_path, dst_annotations_file):
    """
    Generates the training data for the classification model
    cropping patches from full page images. 
    Only training examples belonging the classes are generated
    """
    data = [[], []]
    for _, row in df.iterrows():
        image_id = row['image_id']
        aa = parse_annotations(row['labels'])

        im1 = crop_boxes_and_save(image_id, aa,
                                              full_images_path,
                                              char_images_dst_path)
        data[0].append(np.array(im1)[:, 0])
        data[1].append(np.array(im1)[:, 1])


    proc_df = pd.DataFrame(zip(*data), columns=['label', 'char_path'])
    proc_df.to_csv(dst_annotations_path)


def generate_char_and_ctx_dataset(df, classes, full_images_path, char_images_dst_path, context_images_dst_path, dst_annotations_path):
    """
    Generates the training data for the classification model
    cropping patches from full page images. 
    Only training examples belonging the classes are generated
    """
    total = len(df)
    data = [[], [], []]
    counter = 0
    for _, row in df.iterrows():
        image_id = row['image_id']
        aa = parse_annotations(row['labels'])

        im1 = crop_boxes_and_save(image_id, aa,
                                              full_images_path,
                                              char_images_dst_path)
        im2 = crop_boxes_ctx_and_save(image_id, aa,
                                              full_images_path,
                                              char_images_dst_path)                                     
          
        if len(im1) != len(im2):
            print(f'problems cropping image {image_id}')
            continue        
        data[0].append(np.array(im1)[:, 0])
        data[1].append(np.array(im1)[:, 1])
        data[2].append(np.array(im2)[:, 1])
        counter += 1
        if counter%20 == 0:
            print(f'processed {counter}/{total} images')
    proc_df = pd.DataFrame(zip(*data), columns=['label', 'char_path', 'ctx_path'])
    proc_df.to_csv(dst_annotations_path)


def split_classification_dataset(annotations_file, train_annotation_dst_path, valid_annotation_dst_path):
    pd.read_csv(annotations_file)
    # Shuffling the DataFrame
    shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # Splitting the DataFrame into training and validation sets
    train_df, val_df = train_test_split(shuffled_df, test_size=0.2, random_state=42)
    train_df.to_csv(train_annotation_dst_path, index=False)
    val_df.to_csv(valid_annotation_dst_path, index=False)
