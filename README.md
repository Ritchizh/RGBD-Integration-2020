# RGBD-Integration:
Applying Open3D functions to integrate experimentally measured color and depth frames into a 3D object.
Data were obtained with Intel RealSense depth camera.

**Open3d version: 0.9.0.0**

# List of files:
- **main__TSDF_Integrate__color_depth.py** - run this Python script to perform integration of color and depth frames from Test_data folder
- **main__TSDF_Integrate__depth_only.py** - run this Python script to perform integration of depth frames from Test_data folder
- **_background_substruction_v2.py** - Segmentation relies on OpenCV morphological filter (see docs: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)
**!** A .bag record of the background should be captured along with the subject's data.
An averaged background image is removed from the subject's frames. Filters thresholds are selected empirically.
Depth outside the subject's range is cut.


- **trajectory_io.py** - Open3D class to generate Camera poses in the necessary format
- **Test_data** - Folder with depth (and color) frames (43 MB). Depth frames must be '.png' of type 'np.uint16'. 
- **expected_results** - Folder with the correct camera trajectory ('test_segm.log') and volumetric models generated by the scripts.

# Based on Open3D Tutorials:
- RGBD integration: http://www.open3d.org/docs/release/tutorial/Advanced/rgbd_integration.html
- ICP registration: http://www.open3d.org/docs/release/tutorial/Basic/icp_registration.html

# Possible application:
<img width="800" src="https://github.com/Ritchizh/RGBD-Integration/blob/master/readme_images/Slide_00.png">
<img width="650" src="https://github.com/Ritchizh/RGBD-Integration/blob/master/readme_images/Slide_01.png">
<img width="650" src="https://github.com/Ritchizh/RGBD-Integration/blob/master/readme_images/Slide_02.png">
<img width="650" src="https://github.com/Ritchizh/RGBD-Integration/blob/master/readme_images/Slide_03.png">
<img width="650" src="https://github.com/Ritchizh/RGBD-Integration/blob/master/readme_images/Slide_04.png">
