import os
import cv2
import json
import numpy as np
import pandas as pd

# source_folder = os.path.join(os.getcwd(), "images")
source_folder = '/home/yjkim/HANSUNG/datasets/hs_data/total/'  # /home/yjkim/HANSUNG/datasets/hs_data/train_100_1130/'
json_path = "/home/yjkim/HANSUNG/datasets/hs_data/labels_08171013.json"  # labels_train_100.json"  # Relative to root directory
save_path = '/home/yjkim/HANSUNG/datasets/hs_data/SpecularRF_new/crop/'  # 0102_zoom2.5/crop/'

count = 0  # Count of total images saved
zoom_ratio = 2.5  # 1.8
file_bbs = {}  # Dictionary containing polygon coordinates for mask
ORIG_WIDTH = 2048  # Dimensions should match those of ground truth image
ORIG_HEIGHT = 4580
INPUT_WIDTH = 2048
INPUT_HEIGHT = 4096
ZOOM_WIDTH = 256  # ORIG_WIDTH/8
ZOOM_HEIGHT = 512  # ORIG_HEIGHT/8
CROP_WIDTH = int(ZOOM_WIDTH/zoom_ratio)
CROP_HEIGHT = int(ZOOM_HEIGHT/zoom_ratio)
arr2 = np.array([], np.int32)
file_neg = []
file_pos = []


center_x = 0
center_y = 0

# empty mask
# empty = cv2.imread('/home/yjkim/HANSUNG/datasets/hs_data/train_0817/IMG_BE710001_2022-08-16_13-08-49_Normal/masks/IMG_BE710001_2022-08-16_13-08-49_Normal_mask.png', 0)
# mask_folder = '/home/yjkim/HANSUNG/datasets/hs_data/train_70/'

# Read JSON file
with open(json_path) as f:
    data = json.load(f)


# Extract X and Y coordinates if available and update dictionary
def add_to_dict(data, itr, key, count):
    try:
        str_cnt = str(count)
        x_points = data[itr]["regions"][str_cnt]["shape_attributes"]["all_points_x"]
        y_points = data[itr]["regions"][str_cnt]["shape_attributes"]["all_points_y"]
    except:
        print("No BB. Skipping", key)
        return

    all_points = []
    for i, x in enumerate(x_points):
        all_points.append([x, y_points[i]])

    file_bbs[key] = all_points

def find_crop(crack_x, crack_y):  # , type):
    top = int(crack_y - CROP_HEIGHT / 2)
    bottom = int(crack_y + CROP_HEIGHT / 2)
    left = int(crack_x - CROP_WIDTH / 2)
    right = int(crack_x + CROP_WIDTH / 2)

    if left < 0:
        right += abs(left)
        left = 0
    elif right > INPUT_WIDTH:
        left -= abs(int(right - CROP_WIDTH))
        right = INPUT_WIDTH

    if top < 0:
        bottom += abs(top)
        top = 0
    elif bottom > INPUT_HEIGHT:
        top -= abs(int(bottom - CROP_HEIGHT))
        bottom = INPUT_HEIGHT

    print(f'top:{top}|bottom:{bottom}|left:{left}|right:{right}')

    return top, bottom, left, right


if __name__ == "__main__":
    for itr in data:
        file_name_json = data[itr]["filename"]
        file_pos.append(file_name_json[:-4])
        sub_count = 0  # Contains count of masks for a single ground truth image

        if len(data[itr]["regions"]) > 1:
            for _ in range(len(data[itr]["regions"])):
                key = file_name_json[:-4] + "*" + str(sub_count + 1)
                add_to_dict(data, itr, key, sub_count)
                sub_count += 1
        else:
            add_to_dict(data, itr, file_name_json[:-4], 0)

    print("\nDict size: ", len(file_bbs))

    for itr in file_bbs:
        x_list = []
        y_list = []
        for ptr in range(len(file_bbs[itr])):
            x_list.append(file_bbs[itr][ptr][0])
            y_list.append(file_bbs[itr][ptr][1])

        center_x = int((np.max(x_list)+np.min(x_list))/2)
        center_y = int((np.max(y_list)+np.min(y_list))/2)

        print(f'center point is ({center_x}, {center_y})')

        num_masks = itr.split("*")
        ## img_folder = os.path.join(source_folder, num_masks[0] + '.png')
        img_folder = os.path.join('/home/yjkim/HANSUNG/datasets/hs_data/SpecularRF_new/', num_masks[0][:-6] + 'SpecularRF.png')  # Using SepcularRF
        mask_folder = os.path.join(source_folder, num_masks[0] + '_mask.png')
        img = cv2.imread(os.path.join(source_folder, img_folder))
        mask = cv2.imread(os.path.join(source_folder, mask_folder))

        center_t, center_b, center_l, center_r = find_crop(center_x, center_y)


        # cv2.imshow('boxes', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # cv2.imwrite(f'{save_path}{itr}.png', img)

        # image crop
        # crop_img = img[top:bottom, left:right]
        center_img = img[center_t:center_b, center_l:center_r]
        center_img = cv2.resize(center_img, (ZOOM_WIDTH, ZOOM_HEIGHT))

        flip_img0 = cv2.flip(center_img, 0)  # 상하 반전
        flip_img1 = cv2.flip(center_img, 1)  # 좌우 반전
        flip_img2 = cv2.flip(flip_img0, 1)  # 상하 좌우 반전

        # mask crop
        center_mask = mask[center_t:center_b, center_l:center_r]
        center_mask = cv2.resize(center_mask, (ZOOM_WIDTH, ZOOM_HEIGHT))

        flip_mask0 = cv2.flip(center_mask, 0)  # 상하 반전
        flip_mask1 = cv2.flip(center_mask, 1)  # 좌우 반전
        flip_mask2 = cv2.flip(flip_mask0, 1)  # 상하 좌우 반전

        # cv2.rectangle(center_img, (center_l, center_t), (center_r, center_b), (255, 0, 0), 5)
        # cv2.rectangle(left_img, (left_l, left_t), (left_r, left_b), (255, 0, 0), 5)
        # cv2.rectangle(right_img, (right_l, right_t), (right_r, right_b), (255, 0, 0), 5)
        # cv2.rectangle(top_img, (top_l, top_t), (top_r, top_b), (255, 0, 0), 5)
        # cv2.rectangle(bottom_img, (bottom_l, bottom_t), (bottom_r, bottom_b), (255, 0, 0), 5)

        print('shape of center image: ', center_img.shape[:2])

        if len(num_masks) > 1:
            cv2.imwrite(f'{save_path}{num_masks[0][:-7]}_{num_masks[1]}_crack0_Normal.png', center_img)
            cv2.imwrite(f'{save_path}{num_masks[0][:-7]}_{num_masks[1]}_crack0_Normal_mask.png', center_mask)
            cv2.imwrite(f'{save_path}{num_masks[0][:-7]}_{num_masks[1]}_crack1_Normal.png', flip_img0)
            cv2.imwrite(f'{save_path}{num_masks[0][:-7]}_{num_masks[1]}_crack1_Normal_mask.png', flip_mask0)
            cv2.imwrite(f'{save_path}{num_masks[0][:-7]}_{num_masks[1]}_crack2_Normal.png', flip_img1)
            cv2.imwrite(f'{save_path}{num_masks[0][:-7]}_{num_masks[1]}_crack2_Normal_mask.png', flip_mask1)
            cv2.imwrite(f'{save_path}{num_masks[0][:-7]}_{num_masks[1]}_crack3_Normal.png', flip_img2)
            cv2.imwrite(f'{save_path}{num_masks[0][:-7]}_{num_masks[1]}_crack3_Normal_mask.png', flip_mask2)
            '''
            ## Using SpecularRF
            cv2.imwrite(f'{save_path}{num_masks[0][:-7]}_{num_masks[1]}_crack0_RF.png', center_img)
            cv2.imwrite(f'{save_path}{num_masks[0][:-7]}_{num_masks[1]}_crack1_RF.png', flip_img0)
            cv2.imwrite(f'{save_path}{num_masks[0][:-7]}_{num_masks[1]}_crack2_RF.png', flip_img1)
            cv2.imwrite(f'{save_path}{num_masks[0][:-7]}_{num_masks[1]}_crack3_RF.png', flip_img2)
            '''

        else:
            cv2.imwrite(f'{save_path}{itr[:-7]}_crack0_Normal.png', center_img)
            cv2.imwrite(f'{save_path}{itr[:-7]}_crack0_Normal_mask.png', center_mask)
            cv2.imwrite(f'{save_path}{itr[:-7]}_crack1_Normal.png', flip_img0)
            cv2.imwrite(f'{save_path}{itr[:-7]}_crack1_Normal_mask.png', flip_mask0)
            cv2.imwrite(f'{save_path}{itr[:-7]}_crack2_Normal.png', flip_img1)
            cv2.imwrite(f'{save_path}{itr[:-7]}_crack2_Normal_mask.png', flip_mask1)
            cv2.imwrite(f'{save_path}{itr[:-7]}_crack3_Normal.png', flip_img2)
            cv2.imwrite(f'{save_path}{itr[:-7]}_crack3_Normal_mask.png', flip_mask2)
            '''
            ## Using SpecularRF
            cv2.imwrite(f'{save_path}{itr[:-7]}_crack0_RF.png', center_img)
            cv2.imwrite(f'{save_path}{itr[:-7]}_crack1_RF.png', flip_img0)
            cv2.imwrite(f'{save_path}{itr[:-7]}_crack2_RF.png', flip_img1)
            cv2.imwrite(f'{save_path}{itr[:-7]}_crack3_RF.png', flip_img2)
            '''

    df = pd.DataFrame({'filename': file_pos})
    df.to_csv(os.path.join(save_path, 'positive_filename.csv'))

