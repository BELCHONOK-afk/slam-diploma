import cv2
from sklearn.cluster import KMeans
import numpy as np

class LoopClosureDetect:

    def __init__(self, threshold, num_clusters) -> None:
        self.threshold = threshold
        self.orb = cv2.ORB_create()
        self.kmeans = KMeans(n_clusters=num_clusters)

    # Step 1: Feature Detection and Description
    def extract_features(self, image):
        # Use ORB for feature detection and description        
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        return keypoints, descriptors

    # Step 2: Construct Vocabulary
    def construct_vocabulary(self, descriptors):
        self.kmeans.fit(descriptors)
        self.vocabulary = self.kmeans.cluster_centers_
        return self.vocabulary

    # Step 3: Assign Descriptors to Visual Words
    def assign_to_visual_words(self, descriptors, vocabulary):
        num_visual_words = vocabulary.shape[0]
        histogram = np.zeros(num_visual_words)
        for descriptor in descriptors:
            # Find the nearest visual word
            distances = np.linalg.norm(vocabulary - descriptor, axis=1)
            nearest_word_index = np.argmin(distances)
            # Increment the count of the nearest visual word
            histogram[nearest_word_index] += 1
        return histogram
        
    

    # Step 4: Loop Closure Detection
    def detect_loop_closure(self, histogram1, histogram2):
        if histogram1 is not None and histogram2 is not None:
            similarity = np.dot(histogram1, histogram2) / (np.linalg.norm(histogram1) * np.linalg.norm(histogram2))
        #return similarity
            if similarity > self.threshold:
                return True
            
def main():                
    ld = LoopClosureDetect(0.8, 100)
    path = 'test_data/test2.mp4'
    cap = cv2.VideoCapture(path)

    ret, old_frame = cap.read()
    if not ret:
        print("Error reading video file")
        exit()
    
    _, desc1 = ld.extract_features(cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY))

    while cap.isOpened():
        ret, new_frame = cap.read()
        if not ret:
            break
        _, desc2 = ld.extract_features(cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY))

        vocab = ld.construct_vocabulary(np.vstack((desc1, desc2)))
        histogram1 = ld.assign_to_visual_words(desc1, vocab)
        histogram2 = ld.assign_to_visual_words(desc2, vocab)

        if ld.detect_loop_closure(histogram1, histogram2):
            print("Loop Closure Detected!")
        else:
            print("No Loop Closure Detected.")

        cv2.imshow('Frame', old_frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        old_frame = new_frame




    """
    # Example usage
    image1 = cv2.imread('test_data/data/Slambook2 Ch11 Data 4 copy.png', cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread('test_data/data/Slambook2 Ch11 Data 4.png', cv2.IMREAD_GRAYSCALE)

    _, descriptors1 = ld.extract_features(image1)
    _, descriptors2 = ld.extract_features(image2)

    vocabulary = ld.construct_vocabulary(np.vstack((descriptors1, descriptors2)))

    histogram1 = ld.assign_to_visual_words(descriptors1, vocabulary)
    histogram2 = ld.assign_to_visual_words(descriptors2, vocabulary)

    threshold = 0.8
    #print(detect_loop_closure(histogram1, histogram2, threshold))

    if ld.detect_loop_closure(histogram1, histogram2):
        print("Loop Closure Detected!")
    else:
        print("No Loop Closure Detected.")
"""

if __name__ == '__main__':
    main()