import pandas as ps


class KuzushijiDualDataset(Dataset):
    def __init__(self, images_dir, annotations_file, char_transform=None, ctx_transform=None, img_size=256, down_ratio=4, fraction = 1.0):
        self.images_dir = images_dir
        self.annotations = pd.read_csv(annotations_file)
        self.annotations = self.annotations[:100]

        # Reshape the DataFrame so that each row corresponds to a single cropped image and its bounding box
        reshaped_data = []
        for _, row in self.annotations.iterrows():
          image_id = row.iloc[0]
          boxes = self.parse_annotations(row['labels'])
          for b in boxes:
            reshaped_data.append([image_id, b])
        self.annotations = pd.DataFrame(reshaped_data, columns=['image_id', 'box'])
        self.char_transform = char_transform
        self.ctx_transform = ctx_transform
        self.img_size = img_size
        self.down_ratio = down_ratio
        # Reduce the dataset to a fraction
        if fraction < 1.0:
            self.annotations = self.annotations.sample(frac=fraction).reset_index(drop=True)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image_path = Path(self.images_dir) / (self.annotations.iloc[idx]['image_id']+'.jpg')
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        box = self.annotations.iloc[idx]['box']
        #print(f'image shape {image.shape}')
        #print(f'box {box}')
        char_image = image[box[1] : box[1] + box[3], box[0] : box[0] + box[2], :]
        char_padding = calculate_padding(char_image)
        ctx_box_x_min = max(0, box[1] - 2 * box[3])
        ctx_box_x_max = min(image.shape[1], box[1] + 3 * box[3])
        ctx_box_y_min = max(0, box[0] - 2 * box[2])
        ctx_box_y_max = min(image.shape[0], box[0] + 3 * box[2])
        ctx_image = image[ctx_box_x_min : ctx_box_x_max, ctx_box_y_min : ctx_box_y_max, :]
        ctx_padding = calculate_padding(ctx_image)
        char_image = edge_aware_pad(char_image, char_padding)
        ctx_image = edge_aware_pad(ctx_image, ctx_padding)
        if self.char_transform:
            char_image = self.char_transform(char_image)
        if self.ctx_transform:
            ctx_image = self.ctx_transform(ctx_image)
        unicode_pt = box[4]
        return char_image, ctx_image, unicode_pt

    def parse_annotations(self, label_string):
        bboxes = []
        labels = label_string.split()
        for i in range(0, len(labels), 5):
            unicode_char = labels[i]
            x = int(labels[i + 1])
            y = int(labels[i + 2])
            w = int(labels[i + 3])
            h = int(labels[i + 4])
            unicode_pt = labels[i]
            bboxes.append((x, y, w, h, unicode_pt))
        return bboxes