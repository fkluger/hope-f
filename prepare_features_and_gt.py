import os
import glob
import numpy as np
import cv2 as cv
import scipy.spatial.transform
import shutil
import argparse

parser = argparse.ArgumentParser(description='HOPE-F: prepare SIFT features and ground truth',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--input', default="./render", help='path to folder with rendered image pairs')
parser.add_argument('--output', default="./dataset", help='path to folder for storing the processed dataset')
opt = parser.parse_args()

rendered_images_folder = opt.input
processed_dataset_folder = opt.output

inlier_threshold = 2.0
max_num_scenes = 1000

def sampson_distance(F, points1, points2):
    Fx1 = points1 @ F.T
    Fx2 = points2 @ F

    xFx = np.diag(points2 @ F @ points1.T) ** 2

    distances = xFx / (Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Fx2[:, 0] ** 2 + Fx2[:, 1] ** 2)
    distances = np.sqrt(distances)
    return distances

for num_objects in range(4):
    source_base_folder = os.path.join(rendered_images_folder, "%d" % num_objects)
    target_base_folder = os.path.join(processed_dataset_folder, "%d" % num_objects)

    input_folders = sorted(glob.glob(os.path.join(source_base_folder, "*")))

    num_successful = 0

    sift_times = []

    for idx, input_folder in enumerate(input_folders):

        if num_successful >= max_num_scenes:
            break

        img1 = cv.imread(os.path.join(input_folder, 'render0.png'))
        img2 = cv.imread(os.path.join(input_folder, 'render1.png'))

        gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
        sift = cv.SIFT_create()

        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)

        bf = cv.BFMatcher()

        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        good_points_1 = []
        good_points_2 = []
        ratios = []
        for m,n in matches:
            if m.distance < 0.9*n.distance:
                good.append([m])
                idx1 = m.queryIdx
                idx2 = m.trainIdx
                p1 = kp1[idx1].pt
                p2 = kp2[idx2].pt
                p1 = np.array(list(p1) + [1])
                p2 = np.array(list(p2) + [1])
                good_points_1 += [p1]
                good_points_2 += [p2]
                ratios += [m.distance / n.distance]
        good_points_1 = np.stack(good_points_1, axis=1)
        good_points_2 = np.stack(good_points_2, axis=1)
        ratios = np.stack(ratios, axis=0)

        cam_data = np.load(os.path.join(input_folder, "camera_parameters.npz"), allow_pickle=True)
        obj_datas = [np.load(os.path.join(input_folder, "object_%d.npz" % i)) for i in range(num_objects)]

        K = cam_data["K"]
        K[1,1] *= -1
        R_cam = cam_data["R"]
        t_cam = cam_data["t"]

        R_swap = np.array([
            [-1, 0, 0], [0, -1, 0], [0, 0, 1]
        ])

        R_cam = R_cam @ R_swap
        R_cam = R_cam.T

        F_mats = []
        residuals = []
        points1 = good_points_1.T
        points2 = good_points_2.T

        for obj_data in obj_datas:
            R1 = obj_data["rot_euler_0"]
            R2 = obj_data["rot_euler_1"]
            R1 = scipy.spatial.transform.Rotation.from_euler("XYZ", [0, 0, R1]).as_matrix()
            R2 = scipy.spatial.transform.Rotation.from_euler("XYZ", [0, 0, R2]).as_matrix()
            t1 = obj_data["t_0"]
            t2 = obj_data["t_1"]

            R_obj = (R2 @ R1.T).T
            t_obj = t2 - R_obj @ t1

            R = R_cam @ R_obj @ R_cam.T
            t = -R_cam @ R_obj @ R_cam.T @ t_cam + t_cam + R_cam @ t_obj
            tx = np.cross(t, np.identity(t.shape[0]) * -1)

            F = np.linalg.inv(K).T @ tx @ R @ np.linalg.inv(K)

            F_mats += [F]

            distances = sampson_distance(F, points1, points2)

            residuals += [distances]

        residuals = np.stack(residuals, axis=0)
        obj_ids = np.argmin(residuals, axis=0) + 1
        inliers = (np.min(residuals, axis=0) < inlier_threshold).astype(int)
        obj_ids *= inliers

        num_objects = len(obj_datas)
        inliers_per_object = np.sum((residuals < inlier_threshold).astype(int), axis=-1)
        if np.all(inliers_per_object > 7):

            F = np.stack(F_mats, axis=0)

            target_folder = os.path.join(target_base_folder, "%04d" % num_successful)
            os.makedirs(target_folder, exist_ok=True)

            np.savez(os.path.join(target_folder, "features_and_ground_truth.npz"),
                     F=F, labels=obj_ids, points1=points1, points2=points2, ratios=ratios)

            shutil.copyfile(os.path.join(input_folder, 'render0.png'), os.path.join(target_folder, 'render0.png'))
            shutil.copyfile(os.path.join(input_folder, 'render1.png'), os.path.join(target_folder, 'render1.png'))
            shutil.copyfile(os.path.join(input_folder, "camera_parameters.npz"), os.path.join(target_folder, "camera_parameters.npz"))
            for oi in range(num_objects):
                shutil.copyfile(os.path.join(input_folder, "object_%d.npz" % oi), os.path.join(target_folder, "object_%d.npz" % oi))

            num_successful += 1
            print("%d -- %04d: success (%d)" % (num_objects, idx, num_successful))
        else:
            print("%d -- %04d: failed (%d)" % (num_objects, idx, num_successful))
