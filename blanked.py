# импортируем необходимые библиотеки

import cv2
import numpy as np
#from ultralytics import YOLO
import time
from picamera2 import Picamera2
import cv2

#импортируем кастомные модули
from VisualOdometry import CameraPoses # расчет положения камеры
from lane import Lane # детектирование полосы движения
import edge_detection as Edge
from map import Map #построение траектории по полученным данным положения камеры 
#import control_robot as Control #модуль с логикой управления 
from loop_closure import LoopClosureDetect
from motor2 import Robot 
import motor2
def main():
    robot = Robot(motor2.Motor(5, 22), motor2.Motor(17, 27))
    with open('slam-diploma/intrinsicNew.npy', 'rb') as f:
        intrinsic = np.load(f)


    skip_frames = 2
    data_dir = ''
    n_points = 1000
    vo = CameraPoses(n_points, data_dir, skip_frames, intrinsic)
    ld = LoopClosureDetect(threshold=0.99, num_clusters=10)
    
    estimated_path = []
    camera_pose_list = []
    start_pose = np.ones((3,4))
    start_translation = np.zeros((3,1))
    start_rotation = np.identity(3)
    start_pose = np.concatenate((start_rotation, start_translation), axis=1)
    
    """
    Инициализация камеры
    """
    cam = Picamera2()
    config = cam.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)})
    cam.configure(config)
    cam.start()


    #video = cv2.VideoCapture('my_slam/test_data/Road Highway Small.mp4') # инициализация камеры и получение с нее изображения

    process_frames = False
    old_frame = None
    new_frame = None
    frame_counter = 0
    frame_counter_1 = 0
    new_frame_with_keypoints = None
    keypoints = None
    cur_pose = start_pose
    x,y = [], []
    histogram1, histogram2 = None, None

    #while video.isOpened(): #заходим в цикл управления 
    try:
        while True:
            
            #ret, frame = video.read()
            
            frame = cam.capture_array()
            frame = frame[:][::-1]
            if frame:
                # print("Hello")
                
                width = int(frame.shape[1])
                height = int(frame.shape[0])
                #frame = cv2.resize(frame, (width, height))

                # необходимо настроить!!
                ROI = [
                (0.375*width, 0.125*height),
                (0, 0.55*height),
                (0.75*width, height),
                (0.625*width, 0.125*height)
                ]

                
                
                """
                ------------------------------------------
                ЗДЕСЬ РЕАЛИЗУЕТСЯ ОТРИСОВКА ЛИНИИ ДОРОГИ
                ------------------------------------------
                """

                orig_frame = frame.copy() # копия оригинального изображения для вывода на нем полосы 

                lane_obj = Lane(orig_frame=orig_frame, ROI= ROI)

                # Perform thresholding to isolate lane lines
                lane_line_markings = lane_obj.get_line_markings()

                # Plot the region of interest on the image
                lane_obj.plot_roi(plot=False)

                # Perform the perspective transform to generate a bird's eye view
                # If Plot == True, show image with new region of interest
                warped_frame = lane_obj.perspective_transform(plot=False)

                # Generate the image histogram to serve as a starting point
                # for finding lane line pixels
                histogram = lane_obj.calculate_histogram(plot=False)	
                
                # Find lane line pixels using the sliding window method 
                left_fit, right_fit = lane_obj.get_lane_line_indices_sliding_windows(
                    plot=False)

                # Fill in the lane line
                lane_obj.get_lane_line_previous_window(left_fit, right_fit, plot=False)
                
                # Overlay lines on the original frame
                frame_with_lane_lines = lane_obj.overlay_lane_lines(plot=False)

                center_line = lane_obj.center_line

                # Calculate lane line curvature (left and right lane lines)
                lane_obj.calculate_curvature(print_to_terminal=False)

                # Calculate center offset  																
                lane_obj.calculate_car_position(print_to_terminal=False)
                
                # Display curvature and center offset on image
                frame_with_lane_lines2 = lane_obj.display_curvature_offset(
                    frame=frame_with_lane_lines, plot=False)
                            
                        
                
                # Display the frame             
                cv2.imshow("Frame", frame_with_lane_lines2) 

                

            new_frame = frame.copy() # создадим копию оригинального изображения для вывода на нем контрольных точек 
            frame_counter += 1
            
            start = time.perf_counter()
            

            if process_frames:
                # print("I;m here")

                q1, q2, keypoints = vo.get_matches(old_frame, new_frame)
                if q1 is not None:
                    if len(q1) > 20 and len(q2) > 20:
                        transf = vo.get_pose(q1, q2)
                        cur_pose = cur_pose @ transf
                _, desc1 = ld.extract_features(cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY))
                _, desc2 = ld.extract_features(cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY))
                max_desc_count = max(desc1.shape[0], desc2.shape[0])

                # Обрезать или дополнить дескрипторы до одинаковой длины
                desc1 = desc1[:max_desc_count] if desc1.shape[0] > max_desc_count else np.vstack((desc1, np.zeros((max_desc_count - desc1.shape[0], desc1.shape[1]))))
                desc2 = desc2[:max_desc_count] if desc2.shape[0] > max_desc_count else np.vstack((desc2, np.zeros((max_desc_count - desc2.shape[0], desc2.shape[1]))))

                vocab = ld.construct_vocabulary(np.vstack((desc1, desc2)))
                histogram1 = ld.assign_to_visual_words(desc1, vocab)
                histogram2 = ld.assign_to_visual_words(desc2, vocab)

                
                hom_array = np.array([[0,0,0,1]])
                hom_camera_pose = np.concatenate((cur_pose,hom_array), axis=0)
                camera_pose_list.append(hom_camera_pose)
                estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
                
                x.append(cur_pose[0, 3])
                y.append(cur_pose[2, 3])
                
            
        

            new_frame_with_keypoints = cv2.drawKeypoints(new_frame, keypoints, None, color=(0, 255, 0), flags=0)
            old_frame = new_frame

            process_frames = True

            end = time.perf_counter()
            
            total_time = end - start
            fps = 1 / total_time
            
            cv2.putText(new_frame_with_keypoints, f'FPS: {int(fps)}', (20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            
            # cv2.putText(new_frame_with_keypoints, str(np.round(cur_pose[0, 0],2)), (260,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            # cv2.putText(new_frame_with_keypoints, str(np.round(cur_pose[0, 1],2)), (340,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            # cv2.putText(new_frame_with_keypoints, str(np.round(cur_pose[0, 2],2)), (420,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            # cv2.putText(new_frame_with_keypoints, str(np.round(cur_pose[1, 0],2)), (260,90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            # cv2.putText(new_frame_with_keypoints, str(np.round(cur_pose[1, 1],2)), (340,90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            # cv2.putText(new_frame_with_keypoints, str(np.round(cur_pose[1, 2],2)), (420,90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            # cv2.putText(new_frame_with_keypoints, str(np.round(cur_pose[2, 0],2)), (260,130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            # cv2.putText(new_frame_with_keypoints, str(np.round(cur_pose[2, 1],2)), (340,130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            # cv2.putText(new_frame_with_keypoints, str(np.round(cur_pose[2, 2],2)), (420,130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # cv2.putText(new_frame_with_keypoints, str(np.round(cur_pose[0, 3],2)), (540,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            # cv2.putText(new_frame_with_keypoints, str(np.round(cur_pose[1, 3],2)), (540,90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            # cv2.putText(new_frame_with_keypoints, str(np.round(cur_pose[2, 3],2)), (540,130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        

            if ld.detect_loop_closure(histogram1, histogram2):
                cv2.putText(new_frame_with_keypoints, 'Loop Detected! Change path!', (int(0.1*width),int(0.1*height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imshow('img', new_frame_with_keypoints)

            Map.update_plot(x,y)

            # выход из цикла по нажатию кнопки 
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        

        # Stop when the video is finished
        #video.release()


        # Close all windows
        cv2.destroyAllWindows() 
    
    except KeyboardInterrupt:
        pass

    finally:
        cv2.waitKey(1)
        #out.release()


main()



