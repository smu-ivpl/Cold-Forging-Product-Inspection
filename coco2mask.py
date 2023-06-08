import glob
import os
import cv2
import json
import numpy as np
import pandas as pd

source_folder = '/home/yjkim/HANSUNG/datasets/hs_data/GT_100/'
total_folder = '/home/yjkim/HANSUNG/datasets/stabbed_data_1013/*/*Normal.png'
json_path = os.path.join('/home/yjkim/HANSUNG/datasets/hs_data/labels100_230202.json')  # datasets-100
# json_path = '/home/yjkim/HANSUNG/datasets/stabbed_data_0119/datasets_0119_annotations.coco.json'  # datasets-90
count = 0  # Count of total images saved
file_bbs = {}  # Dictionary containing polygon coordinates for mask
MASK_WIDTH = 2048  # Dimensions should  match those of ground truth image
MASK_HEIGHT = 4580
file_neg = []  # crack_X
file_pos = []  # crack_O
file_pos_i = []  # crack_O_index
file_neg_i = []  # crack_X_index
index_dict = {}
sub_index = 0
mask = np.zeros((MASK_HEIGHT, MASK_WIDTH, 3))
img_path = [file for file in os.listdir(source_folder) if file.endswith("Normal.png")]
file_path = [p for p in glob.glob(total_folder)]

# Read JSON file
with open(json_path) as f:
    data = json.load(f)

# Extract X and Y coordinates if available and update dictionary
def add_to_dict(data, itr, key, count, mask):
    all_points = []
    for index in range(0, len(data["annotations"][itr]['segmentation'][0]), 2):
        x_points = data["annotations"][itr]['segmentation'][0][index]
        y_points = data["annotations"][itr]['segmentation'][0][index + 1]
        all_points.append([int(x_points), int(y_points)])

    file_bbs[key] = all_points
    fill_pts = np.array(all_points, np.int32)
    mask = cv2.fillPoly(mask, [fill_pts], color=(255, 255, 255))
    img = cv2.imread(os.path.join(source_folder, key + '.png'))
    rf = cv2.imread(os.path.join(source_folder, key[:-6] + 'SpecularRF.png'))

    file_bbs[key] = all_points

    find_i = index_dict.get(key + '.png')
    # print('find index is ', find_i)

    file_pos_i.append(find_i)

    pos_dir = '/home/yjkim/HANSUNG/datasets/hs_data/check_GT_1013/Positive/' + find_i

    if not os.path.exists(pos_dir):
        os.makedirs(pos_dir)

    cv2.imwrite(os.path.join(pos_dir, key + '_mask.png'), mask)
    cv2.imwrite(os.path.join(pos_dir, key + '.png'), img)
    cv2.imwrite(os.path.join(pos_dir, key[:-6] + 'SpecularRF.png'), rf)

    ## cv2.imwrite(os.path.join('/home/yjkim/HANSUNG/datasets/hs_data/check_GT_100/' + find_i, key + "_mask.png"), mask)
    ## cv2.imwrite(os.path.join(source_folder, key + "_mask.png"), mask)


if __name__ == "__main__":
    # create to index directory & save to normal/SpecularRF.png in this directory

    for path in file_path:
        normal_name = path.split('/')[-1]
        rf_path = path[:-10] + 'SpecularRF.png'

        normal_img = cv2.imread(path)
        rf_img = cv2.imread(rf_path)

        index = path.split('/')[-2]

        index_dict[normal_name] = index

        directory = '/home/yjkim/HANSUNG/datasets/hs_data/check_GT_100/' + index

        if not os.path.exists(directory):
            os.makedirs(directory)

        # cp -r *normal.png
        ## cv2.imwrite(os.path.join(directory, normal_name), normal_img)
        # cp -r *SpecularRF.png
        ## cv2.imwrite(os.path.join(directory, rf_path.split('/')[-1]), rf_img)



    for itr in range(len(data["annotations"])):
        img_num = data["annotations"][itr]['image_id'] - 1
        file_name_json = data["images"][img_num]["file_name"].split('.')[0] + '.png'  # .split('.')[0][:-4] + '.png'
        file_pos.append(file_name_json)
        sub_num = itr - img_num

        if sub_index < sub_num:  # Image with more than 2 defect.
            # key = file_name_json[:-4] + "*" + str(sub_num - sub_index)
            add_to_dict(data, itr, file_name_json[:-4], sub_num - sub_index, mask)
            try:
                next_itr = itr + 1  # for exception.
                if data['annotations'][next_itr]['image_id'] != data['annotations'][itr]['image_id']:  # The final defect case.
                    sub_index += 1
            except:
                print('The end!')
        else:
            mask = np.zeros((MASK_HEIGHT, MASK_WIDTH))
            add_to_dict(data, itr, file_name_json[:-4], 0, mask)

    print("\nDict size: ", len(file_bbs))

    file_pos = list(set(file_pos))  # Deduplication.

    for file_name in img_path:
        if file_name not in file_pos:
            file_neg.append(file_name)

    for neg in file_neg:
        neg_mask = np.zeros((MASK_HEIGHT, MASK_WIDTH))
        find_ni = index_dict.get(neg)
        # print('find neg index is ', find_ni)
        file_neg_i.append(find_ni)

        neg_dir = '/home/yjkim/HANSUNG/datasets/hs_data/check_GT_1013/Negative/' + find_ni

        if not os.path.exists(neg_dir):
            os.makedirs(neg_dir)

        neg_img = cv2.imread(os.path.join(source_folder, neg[:-4] + '.png'))
        neg_rf = cv2.imread(os.path.join(source_folder, neg[:-10] + 'SpecularRF.png'))

        cv2.imwrite(os.path.join(neg_dir, neg[:-4] + '_mask.png'), neg_mask)
        cv2.imwrite(os.path.join(neg_dir, neg[:-4] + '.png'), neg_img)
        cv2.imwrite(os.path.join(neg_dir, neg[:-10] + 'SpecularRF.png'), neg_rf)

        ## cv2.imwrite(os.path.join('/home/yjkim/HANSUNG/datasets/hs_data/check_GT_100/' + find_ni, neg[:-4] + "_mask.png"), neg_mask)

        ## cv2.imwrite(os.path.join(source_folder, neg[:-4] + '_mask.png'), neg_mask)

    file_pos = list(set(file_pos))  # Deduplication.
    file_pos_i = list(set(file_pos_i))  # Deduplication.

    print('number of the negative file: ', len(file_neg_i))
    print('number of the positive file: ', len(file_pos_i))

    # file_pos = file_pos.sort()
    # file_neg = file_neg.sort()
    df1 = pd.DataFrame({'positive': file_pos_i})
    df2 = pd.DataFrame({'negative': file_neg_i})
    df = pd.concat([df1, df2], axis=1)
    df.to_csv('/home/yjkim/HANSUNG/datasets/hs_data/check_GT_1013/DATA_GT_1013.csv')
    # df.to_csv(os.path.join(source_folder, 'DATA_GT_100.csv'))

