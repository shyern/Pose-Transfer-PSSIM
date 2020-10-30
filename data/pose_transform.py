import numpy as np
import pandas as pd
import json
import os

from skimage.draw import circle, line_aa, polygon
from skimage.io import imread
from skimage.transform import warp_coords
import skimage.measure
import skimage.transform
import pylab as plt

LABELS = ['nose', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri',
               'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Leye', 'Reye', 'Lear', 'Rear']

MISSING_VALUE = -1

def load_pose_cords_from_strings(y_str, x_str):
    y_cords = json.loads(y_str)
    x_cords = json.loads(x_str)
    return np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)

def give_name_to_keypoints(array):
    res = {}
    # if pose_dim==16:
    #     for i, name in enumerate(LABELS):
    #         if array[i][0] != MISSING_VALUE and array[i][1] != MISSING_VALUE:
    #             res[name] = array[i][::-1]
    # else:

    for i, name in enumerate(LABELS):
        if array[i][0] != MISSING_VALUE and array[i][1] != MISSING_VALUE:
            res[name] = array[i][::-1]
    return res


def check_valid(kp_array):
    kp = give_name_to_keypoints(kp_array)
    return check_keypoints_present(kp, ['Rhip', 'Lhip', 'Lsho', 'Rsho'])


def check_keypoints_present(kp, kp_names):
    result = True
    for name in kp_names:
        result = result and (name in kp)
    return result


def compute_st_distance(kp):
    st_distance1 = np.sum((kp['Rhip'] - kp['Rsho']) ** 2)
    st_distance2 = np.sum((kp['Lhip'] - kp['Lsho']) ** 2)
    return np.sqrt((st_distance1 + st_distance2)/2.0)


def mask_from_kp_array(kp_array, border_inc, img_size):
    min = np.min(kp_array, axis=0)
    max = np.max(kp_array, axis=0)
    min -= int(border_inc)
    max += int(border_inc)

    min = np.maximum(min, 0)
    max = np.minimum(max, img_size[::-1])

    mask = np.zeros(img_size)
    mask[min[1]:max[1], min[0]:max[0]] = 1
    return mask


def get_array_of_points(kp, names):
    return np.array([kp[name] for name in names])


def estimate_polygon(fr, to, st, inc_to, inc_from, p_to, p_from):
    fr = fr + (fr - to) * inc_from
    to = to + (to - fr) * inc_to

    norm_vec = fr - to
    norm_vec = np.array([-norm_vec[1], norm_vec[0]])
    norm = np.linalg.norm(norm_vec)
    if norm == 0:
        return np.array([
            fr + 1,
            fr - 1,
            to - 1,
            to + 1,
        ])
    norm_vec = norm_vec / norm
    vetexes = np.array([
        fr + st * p_from * norm_vec,
        fr - st * p_from * norm_vec,
        to - st * p_to * norm_vec,
        to + st * p_to * norm_vec
    ])

    return vetexes


# note that the transforms are the inverse transforms, from output to input
# this is how tf and pytorch apis expect the affine warp matrices
def affine_transforms(array1, array2):
    kp1 = give_name_to_keypoints(array1)
    kp2 = give_name_to_keypoints(array2)

    st1 = compute_st_distance(kp1)
    st2 = compute_st_distance(kp2)


    no_point_tr = np.array([[1, 0, 1000], [0, 1, 1000], [0, 0, 1]])

    transforms = []
    def to_transforms(tr):
        from numpy.linalg import LinAlgError
        try:
            np.linalg.inv(tr)
            transforms.append(tr)
        except LinAlgError:
            transforms.append(no_point_tr)

    body_poly_1 = get_array_of_points(kp1, ['Rhip', 'Lhip', 'Lsho', 'Rsho'])
    body_poly_2 = get_array_of_points(kp2, ['Rhip', 'Lhip', 'Lsho', 'Rsho'])
    tr = skimage.transform.estimate_transform('affine', src=body_poly_2, dst=body_poly_1)


    to_transforms(tr.params)

    head_candidate_names = {'Leye', 'Reye', 'Lear', 'Rear', 'nose'}
    head_kp_names = set()
    for cn in head_candidate_names:
        if cn in kp1 and cn in kp2:
            head_kp_names.add(cn)
    if len(head_kp_names) != 0:
        #if len(head_kp_names) < 3:
        head_kp_names.add('Lsho')
        head_kp_names.add('Rsho')
        head_poly_1 = get_array_of_points(kp1, list(head_kp_names))
        head_poly_2 = get_array_of_points(kp2, list(head_kp_names))
        tr = skimage.transform.estimate_transform('affine', src=head_poly_2, dst=head_poly_1)
        to_transforms(tr.params)
    else:
        to_transforms(no_point_tr)

    def estimate_join(fr, to, inc_to):
        if not check_keypoints_present(kp2, [fr, to]):
            return no_point_tr
        poly_2 = estimate_polygon(kp2[fr], kp2[to], st2, inc_to, 0.1, 0.2, 0.2)
        if check_keypoints_present(kp1, [fr, to]):
            poly_1 = estimate_polygon(kp1[fr], kp1[to], st1, inc_to, 0.1, 0.2, 0.2)
        else:
            if fr[0]=='R':
                fr = fr.replace('R', 'L')
                to = to.replace('R', 'L')
            else:
                fr = fr.replace('L', 'R')
                to = to.replace('L', 'R')
            if check_keypoints_present(kp1, [fr, to]):
                poly_1 = estimate_polygon(kp1[fr], kp1[to], st1, inc_to, 0.1, 0.2, 0.2)
            else:
                return no_point_tr
        return skimage.transform.estimate_transform('affine', dst=poly_1, src=poly_2).params

    to_transforms(estimate_join('Rhip', 'Rkne', 0.1))
    to_transforms(estimate_join('Lhip', 'Lkne', 0.1))

    to_transforms(estimate_join('Rkne', 'Rank', 0.3))
    to_transforms(estimate_join('Lkne', 'Lank', 0.3))

    to_transforms(estimate_join('Rsho', 'Relb', 0.1))
    to_transforms(estimate_join('Lsho', 'Lelb', 0.1))

    to_transforms(estimate_join('Relb', 'Rwri', 0.3))
    to_transforms(estimate_join('Lelb', 'Lwri', 0.3))

    return np.array(transforms).reshape((-1, 9))[..., :-1]
    # return np.array(transforms).reshape((-1, 9))



def pose_masks(array2, img_size):
    kp2 = give_name_to_keypoints(array2)
    masks = []
    st2 = compute_st_distance(kp2)
    empty_mask = np.zeros(img_size)

    # body_mask = np.ones(img_size)
    body_mask = mask_from_kp_array(get_array_of_points(kp2, ['Rhip', 'Lhip', 'Lsho', 'Rsho']), 0.1 * st2, img_size)
    masks.append(body_mask)

    head_candidate_names = {'Leye', 'Reye', 'Lear', 'Rear', 'nose'}
    head_kp_names = set()
    for cn in head_candidate_names:
        if cn in kp2:
            head_kp_names.add(cn)


    if len(head_kp_names)!=0:
        center_of_mass = np.mean(get_array_of_points(kp2, list(head_kp_names)), axis=0, keepdims=True)
        center_of_mass = center_of_mass.astype(int)
        head_mask = mask_from_kp_array(center_of_mass, 0.40 * st2, img_size)
        masks.append(head_mask)
    else:
        masks.append(empty_mask)

    def mask_joint(fr, to, inc_to):
        if not check_keypoints_present(kp2, [fr, to]):
            return empty_mask
        return skimage.measure.grid_points_in_poly(img_size, estimate_polygon(kp2[fr], kp2[to], st2, inc_to, 0.1, 0.2, 0.2)[:, ::-1])

    masks.append(mask_joint('Rhip', 'Rkne', 0.1))
    masks.append(mask_joint('Lhip', 'Lkne', 0.1))

    masks.append(mask_joint('Rkne', 'Rank', 0.5))
    masks.append(mask_joint('Lkne', 'Lank', 0.5))

    masks.append(mask_joint('Rsho', 'Relb', 0.1))
    masks.append(mask_joint('Lsho', 'Lelb', 0.1))

    masks.append(mask_joint('Relb', 'Rwri', 0.5))
    masks.append(mask_joint('Lelb', 'Lwri', 0.5))

    masks = np.array(masks)
    # return np.array(masks)

    bg_mask = np.expand_dims(1.0 - np.amax(masks, axis=0), 0)
    full_masks = np.concatenate((bg_mask, masks), axis=0)
    return full_masks

def estimate_uniform_transform(array1, array2):
    kp1 = give_name_to_keypoints(array1)
    kp2 = give_name_to_keypoints(array2)

    no_point_tr = np.array([[1, 0, 1000], [0, 1, 1000], [0, 0, 1]])

    def check_invertible(tr):
        from numpy.linalg import LinAlgError
        try:
            np.linalg.inv(tr)
            return True
        except LinAlgError:
            return False

    keypoint_names = {'Rhip', 'Lhip', 'Lsho', 'Rsho'}
    candidate_names = {'Rkne', 'Lkne'}

    for cn in candidate_names:
        if cn in kp1 and cn in kp2:
            keypoint_names.add(cn)

    poly_1 = get_array_of_points(kp1, list(keypoint_names))
    poly_2 = get_array_of_points(kp2, list(keypoint_names))

    tr = skimage.transform.estimate_transform('affine', src=poly_2, dst=poly_1)
    # tr = skimage.transform.estimate_transform('affine', src=poly_1, dst=poly_2)

    tr = tr.params

    if check_invertible(tr):
        # return tr.reshape((-1, 9))[..., :-1]
        return tr.reshape((-1, 9))
    else:
        return no_point_tr.reshape((-1, 9))[..., :-1]
        # return no_point_tr.reshape((-1, 9))


def draw_pose_from_cords(pose_joints, img_size, radius=2, draw_joints=True):
    LIMB_SEQ = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9],
                [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16],
                [0, 15], [15, 17], [2, 16], [5, 17]]

    COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    colors = np.zeros(shape=img_size + (3, ), dtype=np.uint8)
    mask = np.zeros(shape=img_size, dtype=bool)

    if draw_joints:
        for f, t in LIMB_SEQ:
            from_missing = pose_joints[f][0] == MISSING_VALUE or pose_joints[f][1] == MISSING_VALUE
            to_missing = pose_joints[t][0] == MISSING_VALUE or pose_joints[t][1] == MISSING_VALUE
            if from_missing or to_missing:
                continue
            yy, xx, val = line_aa(pose_joints[f][0], pose_joints[f][1], pose_joints[t][0], pose_joints[t][1])
            colors[yy, xx] = np.expand_dims(val, 1) * 255
            mask[yy, xx] = True

    for i, joint in enumerate(pose_joints):
        if pose_joints[i][0] == MISSING_VALUE or pose_joints[i][1] == MISSING_VALUE:
            continue
        yy, xx = circle(joint[0], joint[1], radius=radius, shape=img_size)
        colors[yy, xx] = COLORS[i]
        mask[yy, xx] = True

    return colors, mask


def compute_AFtrans_param(pairs_dir, anno_dir, img_size, save_path, warp_type='mask'):
    pairs_file = pd.read_csv(pairs_dir)
    pairs_len = len(pairs_file)

    annotations_file = pd.read_csv(anno_dir, sep=':')
    annotations_file = annotations_file.set_index('name')

    if warp_type == 'mask':
        AFtrans_param = [np.empty([pairs_len, 10, 8]),   # Affine transformation parameters. 10: the len of body limbs, 8: the param for each limbs
                         np.empty([pairs_len, 10] + img_size)]  # The mask of 10 body limbs.

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for i in range(pairs_len):
        print('processing %d / %d ...' % (i, pairs_len))
        pair_name = [pairs_file.iloc[i]['from'], pairs_file.iloc[i]['to']]
        save_name = pair_name[0] + '___' + pair_name[1]
        file_name = os.path.join(save_path, save_name+'.npy')
        fr = annotations_file.loc[pair_name[0]]
        to = annotations_file.loc[pair_name[1]]
        kp_array1 = load_pose_cords_from_strings(fr['keypoints_y'],
                                                 fr['keypoints_x'])
        kp_array2 = load_pose_cords_from_strings(to['keypoints_y'],
                                                 to['keypoints_x'])
        if warp_type == 'mask':
            AFtrans_param[0][i] = affine_transforms(kp_array1, kp_array2)
            AFtrans_param[1][i] = pose_masks(kp_array2, img_size)

        np.save(file_name, AFtrans_param)

def draw_line(fr, to, thickness, shape):
    norm_vec = fr - to
    norm_vec = np.array([-norm_vec[1], norm_vec[0]])
    norm_vec = thickness * norm_vec / np.linalg.norm(norm_vec)

    vetexes = np.array([
        fr + norm_vec,
        fr - norm_vec,
        to - norm_vec,
        to + norm_vec
    ])

    return skimage.draw.polygon(vetexes[:, 1], vetexes[:, 0], shape=shape)

def make_stickman(kp_array, img_shape):
    kp = give_name_to_keypoints(kp_array)
    #Adapted from https://github.com/CompVis/vunet/
    # three channels: left, right, center
    scale_factor = img_shape[1] / 128.0
    thickness = int(3 * scale_factor)
    imgs = list()
    for i in range(3):
        imgs.append(np.zeros(img_shape[:2], dtype="float32"))

    body = ["Lhip", "Lsho", "Rsho", "Rhip"]
    body_pts = get_array_of_points(kp, body)
    if np.min(body_pts) >= 0:
        body_pts = np.int_(body_pts)
        rr,cc = skimage.draw.polygon(body_pts[:,1], body_pts[:, 0], shape=img_shape)
        imgs[2][rr, cc] = 1

    right_lines = [
            ("Rank", "Rkne"),
            ("Rkne", "Rhip"),
            ("Rhip", "Rsho"),
            ("Rsho", "Relb"),
            ("Relb", "Rwri")]
    for line in right_lines:
        if check_keypoints_present(kp, line):
            line_pts = get_array_of_points(kp, line)
            rr,cc = draw_line(line_pts[0], line_pts[1], thickness=thickness, shape=img_shape)
            imgs[0][rr,cc] = 1

    left_lines = [
            ("Lank", "Lkne"),
            ("Lkne", "Lhip"),
            ("Lhip", "Lsho"),
            ("Lsho", "Lelb"),
            ("Lelb", "Lwri")]
    for line in left_lines:
        if check_keypoints_present(kp, line):
            line_pts = get_array_of_points(kp, line)
            rr,cc = draw_line(line_pts[0], line_pts[1], thickness=thickness, shape=img_shape)
            imgs[1][rr, cc] = 1

    if check_keypoints_present(kp, ['Rsho', 'Lsho', 'nose']):
        rs = kp["Rsho"]
        ls = kp["Lsho"]
        cn = kp["nose"]

        neck = 0.5*(rs+ls)
        a = neck
        b = cn
        if np.min(a) >= 0 and np.min(b) >= 0:
            rr,cc = draw_line(a, b, thickness=thickness, shape=img_shape)
            imgs[0][rr, cc] = 0.5
            imgs[1][rr, cc] = 0.5

    if check_keypoints_present(kp, ['Reye', 'Leye', 'nose']):
        reye = kp["Reye"]
        leye = kp["Leye"]
        cn = kp["nose"]

        neck = 0.5*(rs+ls)
        a = tuple(np.int_(neck))
        b = tuple(np.int_(cn))
        if np.min(a) >= 0 and np.min(b) >= 0:
            rr,cc = draw_line(cn, reye, thickness=thickness, shape=img_shape)
            imgs[0][rr, cc] = 0.5
            rr,cc = draw_line(cn, leye, thickness=thickness, shape=img_shape)
            imgs[1][rr, cc] = 0.5
    img = np.stack(imgs, axis = -1)
    return img


def make_masked_image(img, mask, idx=None):
    # idx = [0,1,2,3,4,5,6,7,8,9]
    # idx = [2]
    if idx is None:
        idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    idx_mask = np.expand_dims(np.max(mask[idx,:,:], axis=0), -1)
    # idx_mask = np.expand_dims(mask, -1)
    img_masked = img * idx_mask.astype(np.uint8)

    return img_masked


def make_masked_image2(img, mask):
    # idx = [0,1,2,3,4,5,6,7,8,9]
    # idx = [0]
    # idx_mask = np.expand_dims(np.max(mask[idx,:,:], axis=0), -1)
    idx_mask = np.expand_dims(mask, -1)
    img_masked = img * idx_mask.astype(np.uint8)

    return img_masked


if __name__=='__main__':
    # pairs_dir = '/data0/haoyue/codes/datasets/market_data/market-pairs-train.csv'
    # anno_dir = '/data0/haoyue/codes/datasets/market_data/market-annotation-train.csv'
    # save_path = '/data0/haoyue/codes/datasets/market_data/trainAF'
    # img_size = [128, 64]
    # compute_AFtrans_param(pairs_dir, anno_dir, img_size, save_path)

    import pandas as pd
    import os
    from skimage.transform import resize
    from models.normalization import AffineTransformLayer
    import torch
    import matplotlib
    matplotlib.use('TKAgg')

    pairs_df = pd.read_csv('/data0/haoyue/codes/datasets/market_data_s/market-pairs-train.csv')
    kp_df = pd.read_csv('/data0/haoyue/codes/datasets/market_data_s/market-annotation-train.csv', sep=':')
    img_folder = '/data0/haoyue/codes/datasets/market_data_s/train'
    f = open('lolkek.txt', 'w')
    # plt.axis('off')
    for _, row in pairs_df.iterrows():
        print(1)
        # fr = '0002_c1s1_069056_02.jpg'  # row['from']
        # to = '0002_c2s1_000301_01.jpg'  # row['to']
        fr = '0002_c1s2_064446_01.jpg'
        to = '0002_c1s1_000801_01.jpg'
        fr_img = imread(os.path.join(img_folder, fr))
        to_img = imread(os.path.join(img_folder, to))

        kp_fr = kp_df[kp_df['name'] == fr].iloc[0]
        kp_to = kp_df[kp_df['name'] == to].iloc[0]

        kp_fr = load_pose_cords_from_strings(kp_fr['keypoints_y'], kp_fr['keypoints_x'])
        kp_to = load_pose_cords_from_strings(kp_to['keypoints_y'], kp_to['keypoints_x'])

        to_masks = pose_masks(kp_to, fr_img.shape[:2])
        fr_masks = pose_masks(kp_fr, fr_img.shape[:2])

        kp_fr_img = make_stickman(kp_fr, fr_img.shape)
        kp_to_img = make_stickman(kp_to, to_img.shape)

        mask_idx = 4

        plt.subplot(2,4,1)
        img1 = fr_img.copy()
        p, m = draw_pose_from_cords(kp_fr, fr_img.shape[:2])
        img1[m] = p[m]
        # img1 = img1*np.expand_dims(fr_masks[mask_idx, :, :],-1).astype(np.uint8)
        img1 = make_masked_image(img1, fr_masks)
        plt.imshow(img1)

        plt.subplot(2,4,2)
        img2 = to_img.copy()
        p2, m2 = draw_pose_from_cords(kp_to, to_img.shape[:2])
        img2[m2] = p2[m2]
        img2 = make_masked_image(img2, to_masks)
        plt.imshow(img2)

        plt.subplot(2,4,3)
        img3 = make_masked_image(img1, fr_masks, [mask_idx])
        plt.imshow(img3)

        tr = affine_transforms(kp_fr, kp_to)
        model = AffineTransformLayer(init_image_size=(128, 64), modu_type=None, re_mask_trans=True)

        # tr_torch = torch.from_numpy(tr[mask_idx,:][np.newaxis][np.newaxis]).cuda()
        tr_torch = torch.from_numpy(tr[np.newaxis]).cuda()
        # to_masks_torch = torch.from_numpy((to_masks[mask_idx, :, :])[np.newaxis][np.newaxis]).cuda()
        # to_masks_torch = torch.from_numpy(to_masks[np.newaxis]).cuda()
        # fr_masks_torch = torch.from_numpy((fr_masks[mask_idx, :, :])[np.newaxis][np.newaxis]).cuda()
        fr_masks_torch = torch.from_numpy(fr_masks[np.newaxis]).cuda()

        plt.subplot(2,4,4)
        # img1_torch = torch.from_numpy(img1[np.newaxis].astype(float)).cuda().permute(0, 3, 1, 2)
        img1_torch = torch.from_numpy(fr_img[np.newaxis].astype(float)).cuda().permute(0, 3, 1, 2)
        # mask1_torch = torch.from_numpy(fr_masks[mask_idx,:,:][np.newaxis][np.newaxis]).cuda()

        img1_trans,_, mask_trans = model([img1_torch,img1_torch], tr_torch, fr_masks_torch)
        img1_trans = img1_trans.permute(0, 1, 3, 4, 2).cpu().numpy()
        img4 = (img1_trans[0, mask_idx, ..., 0:3]).copy().astype('uint8')
        plt.imshow(img4)

        plt.subplot(2,4,5)
        img5 = np.max(img1_trans[0], axis=0).copy().astype('uint8')
        plt.imshow(img5)

        plt.subplot(2, 4, 6)
        img8 = np.sum(img1_trans[0], axis=0).copy().astype('uint8')
        plt.imshow(img8)

        plt.subplot(2,4,7)
        whole_fr_mask_torch = torch.from_numpy(np.ones(fr_img.shape[:2])[np.newaxis][np.newaxis]).cuda()
        whole_tr_torch = torch.from_numpy(tr[0,:][np.newaxis][np.newaxis]).cuda()
        whole_img1_trans,_,whole_mask_trans = model([img1_torch,img1_torch], whole_tr_torch, whole_fr_mask_torch)
        whole_img1_trans = whole_img1_trans.permute(0, 1, 3, 4, 2).cpu().numpy()  # B,N,H,W,C
        whole_img_trans = np.concatenate((img1_trans[0, ..., 0:3], whole_img1_trans[0, 0, ..., 0:3][np.newaxis]), axis=0)
        img6 = np.max(whole_img_trans, axis=0).copy().astype('uint8')
        plt.imshow(img6)

        plt.subplot(2,4,8)
        img7 = np.sum(whole_img_trans, axis=0).copy().astype('uint8')
        plt.imshow(img7)


        # plt.subplot(3, 2, 6)
        # mask_trans = mask_trans.permute(0, 2, 3, 1).cpu().numpy()  # (B,N,H,W)-->(B,H,W,N)
        # img4 = np.tile(np.expand_dims(mask_trans[0, ..., mask_idx], -1), (1, 1, 3)).copy()
        # plt.imshow(img4)

        # plt.subplot(3,2,6)
        # mask1 = np.tile(np.expand_dims(fr_masks[mask_idx, :, :], -1), (1, 1, 1))
        # mask1_torch = torch.from_numpy(mask1[np.newaxis].astype(float)).cuda().permute(0,3,1,2)
        #
        # mask1_trans, _ = model(mask1_torch, tr_torch, fr_masks_torch)
        # mask1_trans = mask1_trans.permute(0,1,3,2,4).cpu().numpy()
        # img5 = np.tile(mask1_trans[0, mask_idx, ..., :], (1,1,3)).copy()
        # plt.imshow(img5)
        #
        #
        # # plt.subplot(3,2,5)
        # img2_torch = torch.from_numpy(fr_img[np.newaxis].astype(float)).cuda().permute(0, 3, 1, 2)
        #
        # img2_trans, mask_trans = model(img2_torch, tr_torch, mask1_torch)
        # img2_trans = img2_trans.permute(0, 1, 3, 4, 2).cpu().numpy()
        # img5 = (img2_trans[0, mask_idx, ..., 0:3]).copy().astype('uint8')
        # # plt.imshow(img5)
        #
        # # plt.subplot(3,2,6)
        # mask_trans = mask_trans.permute(0, 1, 3, 4, 2).cpu().numpy()
        # img6 = np.tile(mask_trans[0, mask_idx, ..., :], (1,1,3)).copy()
        # # plt.imshow(img6)

        plt.show()


        # # source image key point rgb image
        # plt.subplot(3, 2, 1)
        # kp_fr_img = make_stickman(kp_fr, fr_img.shape)
        # img = (kp_fr_img*np.expand_dims(fr_masks[mask_idx, :, :],-1)).copy()
        # # img = kp_fr_img.copy()
        # plt.imshow(img)

        # # source image rgb image
        # plt.subplot(1, 5, 1)
        # img = fr_img.copy()
        # p, m = draw_pose_from_cords(kp_fr, fr_img.shape[:2])
        # img[m] = p[m]
        # # img = img*np.expand_dims(fr_masks[mask_idx, :, :],-1).astype(np.uint8)
        # # img = img.astype(np.uint8)
        # img1 = make_masked_image(img, fr_masks).copy()
        # plt.imshow(img1)

        # plt.subplot(3, 2, 3)
        # kp_to_img = make_stickman(kp_to, to_img.shape)
        # img = (kp_to_img*np.expand_dims(to_masks[mask_idx, :, :],-1)).copy()
        # # img = kp_to_img.copy()
        # plt.imshow(img)

        # plt.subplot(1, 5, 2)
        # img = to_img.copy()
        # p2, m2 = draw_pose_from_cords(kp_to, fr_img.shape[:2])
        # img[m2] = p2[m2]
        # # img = img*np.expand_dims(to_masks[mask_idx, :, :],-1).astype(np.uint8)
        # # img = img.astype(np.uint8)
        # img = make_masked_image(img, to_masks)
        # plt.imshow(img)


        # tr = estimate_uniform_transform(kp_fr, kp_to)
        #
        # no_point_tr = np.array([[1, 0, 1000], [0, 1, 1000], [0, 0, 1]])
        # if np.all(tr == no_point_tr.reshape((-1, 9))[..., :-1]):
        #     print >>f, '_'.join([fr,to]) + '.jpg'

        # tr = affine_transforms(kp_fr, kp_to)
        #
        # model = AffineTransformLayer(init_image_size=(128,64), warp_skip='mask')
        # # tr_torch = torch.from_numpy(tr[mask_idx,:][np.newaxis][np.newaxis]).cuda()
        # tr_torch = torch.from_numpy(tr[np.newaxis]).cuda()
        # # to_masks_torch = torch.from_numpy((to_masks[mask_idx, :, :])[np.newaxis][np.newaxis]).cuda()
        # # to_masks_torch = torch.from_numpy(to_masks[np.newaxis]).cuda()
        #
        # fr_masks_torch = torch.from_numpy((fr_masks[mask_idx, :, :])[np.newaxis][np.newaxis]).cuda()
        #
        # plt.subplot(1, 5, 3)
        # # plt.subplot(3, 2, 5)
        # # kp_fr_img_torch = torch.from_numpy((kp_fr_img*np.expand_dims(fr_masks[mask_idx, :, :],-1))[np.newaxis].astype(float)).cuda().permute(0,3,1,2)
        # kp_fr_img_torch = torch.from_numpy(kp_fr_img[np.newaxis].astype(float)).cuda().permute(0,3,1,2)
        # kp_fr_img_tran, _ = model(kp_fr_img_torch, tr_torch, fr_masks_torch)  #(8,10,3,32,16)
        # kp_fr_img_tran = kp_fr_img_tran.permute(0,1,3,2,4).cpu().numpy()
        # img = (kp_fr_img_tran[0, mask_idx, ..., 0:3]).copy()
        #     # .astype('uint8') np.uint8
        # # img = kp_fr_img*np.expand_dims(fr_masks[mask_idx, :, :],-1).astype('uint8')
        # # a[m] = p[m]
        # plt.imshow(img)
        #
        # plt.subplot(1,5,3)
        # img1_torch = torch.from_numpy(img1[np.newaxis].astype(float)).cuda().permute(0,3,1,2)
        # img1_trans, _ = model(img1_torch, tr_torch, fr_masks_torch)
        # img1_trans = img1_trans.permute(0,1,3,2,4).cpu().numpy()
        # img = (img1_trans[0,mask_idx,...,0:3]).copy().astype('uint8')
        # plt.imshow(img)
        #
        # plt.subplot(1, 5, 4)
        # # plt.subplot(3, 2, 6)
        # # fr_img_torch = torch.from_numpy((fr_img*np.expand_dims(fr_masks[mask_idx, :, :],-1))[np.newaxis].astype(float)).cuda().permute(0,3,1,2)
        # fr_img_torch = torch.from_numpy((fr_img)[np.newaxis].astype(float)).cuda().permute(0,3,1,2)
        # tr2_torch = torch.from_numpy(tr[mask_idx, :][np.newaxis][np.newaxis]).cuda()
        # fr_img_tran, fr_masks_tran = model(fr_img_torch, tr2_torch, fr_masks_torch)  #(8,10,3,32,16)
        # fr_img_tran = fr_img_tran.permute(0,1,3,2,4).cpu().numpy()
        # img = fr_img_tran[0, 0, ..., 0:3].copy().astype('uint8')
        # # img = img*np.expand_dims(to_masks[mask_idx, :, :],-1).astype(np.uint8)
        # # a[m] = p[m]
        # plt.imshow(img)
        #
        # plt.subplot(1,5,5)
        # fr_masks_tran = fr_masks_tran.permute(0,1,3,2,4).cpu().numpy()
        # img_mask = fr_masks_tran[0, 0, ..., 0].copy()
        # # img = np.tile(np.expand_dims(img, axis=-1), (1, 1, 3))
        # img = make_masked_image2(img, img_mask)
        # plt.imshow(img)

        # mask_idx = 9
        # kp_fr_img_torch = torch.from_numpy(kp_fr_img[np.newaxis].astype(float)).cuda().permute(0,3,1,2)
        # fr_img_torch = torch.from_numpy(fr_img[np.newaxis].astype(float)).cuda().permute(0,3,1,2)
        # kp_trans, img_trans, mask_trans = model([kp_fr_img_torch, fr_img_torch], tr_torch, fr_masks_torch)
        #
        # # (B,N,C,H,W)
        # img_trans = img_trans.permute(0,1,3,2,4).cpu().numpy()
        # kp_trans = kp_trans.permute(0,1,3,2,4).cpu().numpy()
        # mask_trans = mask_trans.permute(0,1,3,2,4).cpu().numpy()
        #
        # img = img_trans[0,mask_idx,...,0:3].copy().astype('uint8')
        # plt.subplot(1,3,1)
        # plt.imshow(img)
        #
        # img = kp_trans[0, mask_idx, ..., 0:3].copy()
        # plt.subplot(1, 3, 2)
        # plt.imshow(img)
        #
        # img = np.tile(np.expand_dims(mask_trans[0, mask_idx, ..., mask_idx], axis=-1),(1,1,3)).copy()
        # plt.subplot(1, 3, 3)
        # plt.imshow(img)


        # plt.show()
        # f.flush()

    # def get(first, second):
    #     plt.subplot(3, 1, 1)
    #     a = first
    #     img = imread(a[0])
    #     image = img.copy()
    #     array1 = pose_utils.load_pose_cords_from_strings(a[2], a[1])
    #     print (img.shape)
    #     p, m = pose_utils.draw_pose_from_cords(array1, img.shape[:2])
    #     img[m] = p[m]
    #     plt.imshow(img)
    #
    #     plt.subplot(3, 1, 2)
    #     a = second
    #     img = imread(a[0])
    #     array2 = pose_utils.load_pose_cords_from_strings(a[2], a[1])
    #     p, m = pose_utils.draw_pose_from_cords(array2, img.shape[:2])
    #     img[m] = p[m]
    #     plt.imshow(img)
    #     pose_utils.draw_legend()
    #     trs = affine_transforms(array1, array2)
    #     masks = pose_masks(array2, (128, 64))
    #
    #     image = resize(image, (64, 32), preserve_range=True)
    #     m = resize(m, (64, 32), preserve_range=True, order=0).astype(bool)
    #     p = resize(p, (64, 32), preserve_range=True, order=0)
    #     return trs, masks, image, p,  m
    #
    #
    # trs, masks, img, p, m = get(first2, second2)
    # trs2, masks2, img2, p2, m2 = get(first, second)
    #
    # plt.subplot(3, 1, 3)
    #
    # x_v = np.concatenate([img[np.newaxis], img2[np.newaxis]])
    # i_v = np.concatenate([trs[np.newaxis], trs2[np.newaxis]])
    # m_v = np.concatenate([masks[np.newaxis], masks2[np.newaxis]])
    #
    # #trs = CordinatesWarp.affine_transforms(array1, array2)
    # x = Input((64,32,3))
    # i = Input((len(trs), 8))
    # masks = Input((len(trs), 128, 64))
    #
    # y = AffineTransformLayer(len(trs), 'max', (128, 64))([x, i])
    # model = Model(inputs=[x, i, masks], outputs=y)
    #
    # # x_v = skimage.transform.resize(image, (128, 64), preserve_range=True)[np.newaxis]
    # # i_v = trs[np.newaxis]
    #
    # b = model.predict([x_v, i_v, m_v])
    # print (b.shape)
    # b = b[..., :3]
    #
    #
    # # trs, _ = CordinatesWarp.warp_mask(array1, array2, img_size=img.shape[:2])
    # # x = Input((128,64,3))
    # # i = Input((128,62,4))
    # #
    # # y = WarpLayer(1)([x, i])
    # # model = Model(inputs=[x, i], outputs=y)
    # #
    # # x_v = skimage.transform.resize(image, (128, 64), preserve_range=True)[np.newaxis]
    # # i_v = trs[np.newaxis]
    # #
    # # b = model.predict([x_v, i_v])
    # # print (b.shape)
    #
    # warped_image = np.squeeze(b[1]).astype(np.uint8)
    # warped_image[m2] = p2[m2]
    # plt.imshow(warped_image)
    #
    # # from scipy.ndimage import map_coordinates
    # #
    # # mask = CordinatesWarp.warp_mask(array1, array2, (128, 64, 3))
    # # mask = np.moveaxis(mask, -1, 0)
    # # warped_image = map_coordinates(image, mask)
    # # warped_image[m] = p[m]
    # # plt.subplot(4, 1, 4)
    # # plt.imshow(warped_image)
    # plt.show()
