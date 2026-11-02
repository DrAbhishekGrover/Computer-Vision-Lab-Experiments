Experiment 8

Aim: This experiment is to understand the utility of corners and blobs as features/keypoints in template matching

Data Files: Images: “messi.png”, “football.png”, “football2.png”, “checkerboard.png”

1. Use Harris Corner Detector on the image “checkerboard.png” and display the output.
2. Case 1: “messi.png” as the given image and “football2.png” as the template. Use cv.matchTemplate() to display the results of template matching.
3. Case 2: “messi.png” as the given image and “football.png” as the template. Use cv.matchTemplate() to display the results of template matching.
4. Case 3: Pass the “messi.png” and “football2.png” through Harris Corner Detector. Use the resultant images and cv.matchTemplate() to display the results of template matching.
5. Case 4: Pass the “messi.png” and “football.png” through Harris Corner Detector. Use the resultant images and cv.matchTemplate() to display the results of template matching.
6. Detect blobs and generate SIFT descriptors for the image and template. Use these descriptors for template matching. Display the results.
7. Pass the results obtained in point 6 through a RANSAC algorithm to filter out bad matches. Display the results.








 














