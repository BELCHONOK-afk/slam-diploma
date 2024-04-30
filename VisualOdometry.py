import os
import numpy as np
import cv2
from sympy import homogeneous_order

from tqdm import tqdm
import time

from matplotlib import pyplot as plt

from cycler import cycle
   



class CameraPoses():
    """
    Class for estimating camera poses from image frames and intrinsic camera parameters.
    
    Attributes:
        K (numpy.array): Intrinsic camera matrix.
        extrinsic (numpy.array): Extrinsic camera matrix.
        P (numpy.array): Projection matrix.
        orb (cv2.ORB): ORB feature detector.
        flann (cv2.FlannBasedMatcher): FLANN-based feature matcher.
        world_points (list): List to store triangulated 3D points.
        current_pose (numpy.array): Current camera pose.
    
    Methods:
        __init__(data_dir, skip_frames, intrinsic):
            Initializes the CameraPoses object.
        
        _load_images(filepath, skip_frames):
            Loads images from a directory and returns a list of images.
            
        _form_transf(R, t):
            Forms a transformation matrix from rotation matrix and translation vector.
            
        get_world_points():
            Returns the triangulated 3D points.
            
        get_matches(img1, img2):
            Finds feature matches between two images using ORB and FLANN.
            
        get_pose(q1, q2):
            Estimates the camera pose from feature matches using the essential matrix.
            
        decomp_essential_mat(E, q1, q2):
            Decomposes the essential matrix to retrieve rotation and translation.
            
        decomp_essential_mat_old(E, q1, q2):
            Decomposes the essential matrix using an alternative method.
    """
    
    
    
    def __init__(self, n_points, data_dir, skip_frames, intrinsic):
        """
        Initializes the CameraPoses object.
        
        Args:
            data_dir (str): Directory containing image frames.
            skip_frames (int): Number of frames to skip for processing.
            intrinsic (numpy.array): Intrinsic camera matrix.
        """
        
        self.K = intrinsic
        self.extrinsic = np.array(((1,0,0,0),(0,1,0,0),(0,0,1,0)))
        self.P = self.K @ self.extrinsic
        self.orb = cv2.ORB_create(n_points)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        
        self.world_points = []

        self.current_pose = None
        
    @staticmethod
    def _load_images(filepath, skip_frames):
        """
        Loads images from a directory and returns a list of images.
        
        Args:
            filepath (str): Directory containing image frames.
            skip_frames (int): Number of frames to skip for processing.
        
        Returns:
            list: List of loaded images.
        """
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        images = []
        
        for path in tqdm(image_paths[::skip_frames]):
            img = cv2.imread(path)
            if img is not None:
                #images.append(cv2.resize(img, (640,480)))
                images.append(img)
                
        return images
    

    @staticmethod
    def _form_transf(R, t):
        """
        Forms a transformation matrix from rotation matrix and translation vector.
        
        Args:
            R (numpy.array): Rotation matrix.
            t (numpy.array): Translation vector.
        
        Returns:
            numpy.array: Transformation matrix.
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        
        return T

    def get_world_points(self):
        """
        Returns the triangulated 3D points.
        
        Returns:
            numpy.array: Triangulated 3D points.
        """
        return np.array(self.world_points)
    
    def get_matches(self, img1, img2):
        """
        Finds feature matches between two images using ORB and FLANN.
        
        Args:
            img1 (numpy.array): First image.
            img2 (numpy.array): Second image.
        
        Returns:
            tuple: Feature matches in the form of keypoints for both images.
        """
   
        # Find the keypoints and descriptors with ORB
        kp1, des1 = self.orb.detectAndCompute(img1, None)
        kp2, des2 = self.orb.detectAndCompute(img2, None)
        # Find matches
        if len(kp1) > 6 and len(kp2) > 6:
            matches = self.flann.knnMatch(des1, des2, k=2)

            # Find the matches there do not have a to high distance
            good_matches = []
            try:
                for m, n in matches:
                    if m.distance < 0.5 * n.distance:
                        good_matches.append(m)
            except ValueError:
                pass
            
            # Draw matches
            img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
            #cv2.drawMatches(img1, kp1, img2, kp2, good_matches, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
            #cv2.imshow('Good Matches', img_matches)
            #cv2.waitKey(50)
            
            # Get the image points form the good matches
            #q1 = [kp1[m.queryIdx] for m in good_matches]
            #q2 = [kp2[m.trainIdx] for m in good_matches]
            q1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
            q2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        
            return q1, q2,kp2
        else:
            return None, None, None

    def get_pose(self, q1, q2):
        """
        Estimates the camera pose from feature matches using the essential matrix.
        
        Args:
            q1 (numpy.array): Keypoints in the first image.
            q2 (numpy.array): Keypoints in the second image.
        
        Returns:
            numpy.array: Transformation matrix representing the camera pose.
        """
    
        # Essential matrix
        E, mask = cv2.findEssentialMat(q1, q2, self.K)

        # Decompose the Essential matrix into R and t
        R, t = self.decomp_essential_mat_old(E, q1, q2)

        # Get transformation matrix
        transformation_matrix = self._form_transf(R, np.squeeze(t))
        
        return transformation_matrix


    def decomp_essential_mat(self, E, q1, q2):
        """
        Decomposes the essential matrix to retrieve rotation and translation.
        
        Args:
            E (numpy.array): Essential matrix.
            q1 (numpy.array): Keypoints in the first image.
            q2 (numpy.array): Keypoints in the second image.
        
        Returns:
            list: List containing rotation matrix and translation vector.
        """

        R1, R2, t = cv2.decomposeEssentialMat(E)
        T1 = self._form_transf(R1,np.ndarray.flatten(t))
        T2 = self._form_transf(R2,np.ndarray.flatten(t))
        T3 = self._form_transf(R1,np.ndarray.flatten(-t))
        T4 = self._form_transf(R2,np.ndarray.flatten(-t))
        transformations = [T1, T2, T3, T4]
        
        # Homogenize K
        K = np.concatenate((self.K, np.zeros((3,1)) ), axis = 1)

        # List of projections
        projections = [K @ T1, K @ T2, K @ T3, K @ T4]

        np.set_printoptions(suppress=True)

        # print ("\nTransform 1\n" +  str(T1))
        # print ("\nTransform 2\n" +  str(T2))
        # print ("\nTransform 3\n" +  str(T3))
        # print ("\nTransform 4\n" +  str(T4))

        positives = []
        for P, T in zip(projections, transformations):
            hom_Q1 = cv2.triangulatePoints(P, P, q1.T, q2.T)
            hom_Q2 = T @ hom_Q1
            # Un-homogenize
            Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            Q2 = hom_Q2[:3, :] / hom_Q2[3, :]
             
            total_sum = sum(Q2[2, :] > 0) + sum(Q1[2, :] > 0)
            relative_scale = np.mean(np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1)/
                                     np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1))
            positives.append(total_sum + relative_scale)
            

        # Decompose the Essential matrix using built in OpenCV function
        # Form the 4 possible transformation matrix T from R1, R2, and t
        # Create projection matrix using each T, and triangulate points hom_Q1
        # Transform hom_Q1 to second camera using T to create hom_Q2
        # Count how many points in hom_Q1 and hom_Q2 with positive z value
        # Return R and t pair which resulted in the most points with positive z

        max = np.argmax(positives)
        if (max == 2):
            # print(-t)
            return R1, np.ndarray.flatten(-t)
        elif (max == 3):
            # print(-t)
            return R2, np.ndarray.flatten(-t)
        elif (max == 0):
            # print(t)
            return R1, np.ndarray.flatten(t)
        elif (max == 1):
            # print(t)
            return R2, np.ndarray.flatten(t)
        
        
    def decomp_essential_mat_old(self, E, q1, q2):
        """
        Decomposes the essential matrix using an alternative method.
        
        Args:
            E (numpy.array): Essential matrix.
            q1 (numpy.array): Keypoints in the first image.
            q2 (numpy.array): Keypoints in the second image.
        
        Returns:
            list: List containing rotation matrix and translation vector.
        """
        def sum_z_cal_relative_scale(R, t):
            # Get the transformation matrix
            T = self._form_transf(R, t)
            # Make the projection matrix
            P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)

            # Triangulate the 3D points
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            # Also seen from cam 2
            hom_Q2 = np.matmul(T, hom_Q1)

            # Un-homogenize
            Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            Q2 = hom_Q2[:3, :] / hom_Q2[3, :]
            
            #self.world_points.append(Q1)

            # Find the number of points there has positive z coordinate in both cameras
            sum_of_pos_z_Q1 = sum(Q1[2, :] > 0)
            sum_of_pos_z_Q2 = sum(Q2[2, :] > 0)

            # Form point pairs and calculate the relative scale
            relative_scale = np.mean(np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1)/
                                     np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1))
            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

        # Decompose the essential matrix
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)

        # Make a list of the different possible pairs
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

        # Check which solution there is the right one
        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum, scale = sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)
            relative_scales.append(scale)

        # Select the pair there has the most points with positive z coordinate
        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        R1, t = right_pair
        t = t * relative_scale
        
        T = self._form_transf(R1, t)
        # Make the projection matrix
        P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)

        # Triangulate the 3D points
        hom_Q1 = cv2.triangulatePoints(P, P, q1.T, q2.T)
        # Also seen from cam 2
        hom_Q2 = np.matmul(T, hom_Q1)

        # Un-homogenize
        Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
        Q2 = hom_Q2[:3, :] / hom_Q2[3, :]
        
        self.world_points.append(Q1)

        return [R1, t]
    
    
        
        

    
