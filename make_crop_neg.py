import csv
import os.path
import cv2
import pandas as pd

zoom_ratio = 2.5  # 1.8

ORIG_WIDTH = 2048  # Dimensions should match those of ground truth image
ORIG_HEIGHT = 4580
ZOOM_WIDTH = 256  # ORIG_WIDTH/8
ZOOM_HEIGHT = 512  # ORIG_HEIGHT/8
CROP_WIDTH = int(ZOOM_WIDTH/zoom_ratio)
CROP_HEIGHT = int(ZOOM_HEIGHT/zoom_ratio)

source_path = '/home/yjkim/HANSUNG/datasets/hs_data/total/'
csv_path = '/home/yjkim/HANSUNG/datasets/hs_data/train_total/DATA_GT.csv'
# save_path = '/home/yjkim/HANSUNG/datasets/hs_data/0102_zoom2.5/crop_neg/'
save_path = '/home/yjkim/HANSUNG/datasets/hs_data/SpecularRF_new/crop_neg/'

crop_img_name = []
crop_img_size = []
crop_mask_size = []

if __name__ == "__main__":

    f = open(csv_path)
    rdr = csv.reader(f)

    for line in rdr:
        if line[2] == 'negative':
            continue

        print('This image name is', line[2])
        ## img_path = os.path.join(source_path, line[2]) + '.png'
        img_path = os.path.join('/home/yjkim/HANSUNG/datasets/hs_data/SpecularRF_new/', line[2][:-6]) + 'SpecularRF.png'  # Using SepcualrRF
        mask_path = os.path.join(source_path, line[2]) + '_mask.png'

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path)

        # ratio zoom in!
        # img = cv2.resize(img, (int(ORIG_WIDTH*zoom_ratio), int(ORIG_HEIGHT*zoom_ratio)))
        # mask = cv2.resize(mask, (int(ORIG_WIDTH*zoom_ratio), int(ORIG_HEIGHT*zoom_ratio)))

        zh, zw = img.shape[:2]

        print('image size is ', img.shape[:2])

        # number = 0
        for n, start in enumerate([int(ORIG_WIDTH/2), int(ORIG_WIDTH/2+CROP_WIDTH), int(ORIG_WIDTH/2+CROP_WIDTH*2)]):
            # line0
            img0 = img[:CROP_HEIGHT, start:start + CROP_WIDTH]
            mask0 = mask[:CROP_HEIGHT, start:start + CROP_WIDTH]
            img0 = cv2.resize(img0, (ZOOM_WIDTH, ZOOM_HEIGHT))
            mask0 = cv2.resize(mask0, (ZOOM_WIDTH, ZOOM_HEIGHT))

            cv2.imwrite(os.path.join(save_path, line[2][:-6] + f'line0{n}_RF.png'), img0)
            ## cv2.imwrite(os.path.join(save_path, line[2][:-6] + f'line0{n}_Normal.png'), img0)
            ## cv2.imwrite(os.path.join(save_path, line[2][:-6] + f'line0{n}_Normal_mask.png'), mask0)
            crop_img_name.append(line[2][:-6] + f'line0{n}_Normal')
            crop_img_size.append(img0.shape[:2])
            crop_mask_size.append(mask0.shape[:2])

            # line1
            img1 = img[int(ORIG_HEIGHT / 2 - CROP_HEIGHT):int(ORIG_HEIGHT / 2), start:start + CROP_WIDTH]
            mask1 = mask[int(ORIG_HEIGHT / 2 - CROP_HEIGHT):int(ORIG_HEIGHT / 2), start:start + CROP_WIDTH]
            img1 = cv2.resize(img1, (ZOOM_WIDTH, ZOOM_HEIGHT))
            mask1 = cv2.resize(mask1, (ZOOM_WIDTH, ZOOM_HEIGHT))

            cv2.imwrite(os.path.join(save_path, line[2][:-6] + f'line1{n}_RF.png'), img1)
            ## cv2.imwrite(os.path.join(save_path, line[2][:-6] + f'line1{n}_Normal.png'), img1)
            ## cv2.imwrite(os.path.join(save_path, line[2][:-6] + f'line1{n}_Normal_mask.png'), mask1)
            crop_img_name.append(line[2][:-6] + f'line1{n}_Normal')
            crop_img_size.append(img1.shape[:2])
            crop_mask_size.append(mask1.shape[:2])

            # line2
            img2 = img[ORIG_HEIGHT - CROP_HEIGHT:ORIG_HEIGHT, start:start + CROP_WIDTH]
            mask2 = mask[ORIG_HEIGHT - CROP_HEIGHT:ORIG_HEIGHT, start:start + CROP_WIDTH]
            img2 = cv2.resize(img2, (ZOOM_WIDTH, ZOOM_HEIGHT))
            mask2 = cv2.resize(mask2, (ZOOM_WIDTH, ZOOM_HEIGHT))

            cv2.imwrite(os.path.join(save_path, line[2][:-6] + f'line2{n}_RF.png'), img2)
            ## cv2.imwrite(os.path.join(save_path, line[2][:-6] + f'line2{n}_Normal.png'), img2)
            ## cv2.imwrite(os.path.join(save_path, line[2][:-6] + f'line2{n}_Normal_mask.png'), mask2)
            crop_img_name.append(line[2][:-6] + f'line2{n}_Normal')
            crop_img_size.append(img2.shape[:2])
            crop_mask_size.append(mask2.shape[:2])

            # number += 1
    '''
    df1 = pd.DataFrame({'filename': crop_img_name})
    df2 = pd.DataFrame({'crop image size': crop_img_size})
    df3 = pd.DataFrame({'crop mask size': crop_mask_size})
    df = pd.concat([df1, df2, df3], axis=1)
    df.to_csv(os.path.join(save_path, 'imgsize_check.csv'))
    '''
