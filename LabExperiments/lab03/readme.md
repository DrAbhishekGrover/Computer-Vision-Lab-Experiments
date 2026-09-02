Aim: This experiment is to understand the process of camera calibration.

Data Files: checkerboard.zip (Link: https://www.kaggle.com/datasets/danielwe14/stereocamera-chessboard-pictures/data)

1. Read the files in the checkerboard folder. Obtain the intrinsic, extrinsic and projection matrices of the camera.

2. (Exercise) Find the calibration matrix for your smartphone camera. Take 10 images of the checkerboard sheet (Square size: 25 mm, Vertices:8x6) with your smartphone camera.
(Link: https://markhedleyjones.com/projects/calibration-checkerboard-collection) . Now run the code in S.No. 1 on these images. Obtain the intrinsic, extrinsic and projection matrices.

3. (Exercise) To verify the result in S.No.2: Find the true physical focal length (f) and pixel density (p) (in mm) of your main camera from internet sources. The ratio (f/p) should match with K(1,1) element of intrinsic matrix.

Note: For the calibration process, ensure that images captured from smartphone are raw/unprocessed images. The .raw file format is preferable. Approximate results may be obtained if your smartphone capture images in .jpg format by default.
   
