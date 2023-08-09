'''
@File    :   main__TSDF_Integrate__depth_only_11.py
@Time    :   2023/08/09 14:21:27
@Author  :   Margarita 
'''

import open3d            as o3d # v 0.17.0
import numpy             as np
import matplotlib.pyplot as plt
import itertools
import copy

plt.close('all')
pic_num = 1

# Change to your project path:
my_project_path = 'C:/Users/Margarita/Desktop/RGBD-Integration-2023/'

#==============================================================================
#==============================================================================

#==============================================================================
#==============================================================================

#  == ICP functions: ==

# check: http://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html


# ! Note: To use current open3d 0.17.0 version,
# o3d.registration was changed to o3d.pipelines.registration

def draw_registration_result(source, target, transformation, title='Title'):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp], window_name=title)


def preprocess_point_cloud(pcd, voxel_size):
#    print("\n- - Preprocessing - -\n:: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
#    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
#    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def refine_registration(source, target, voxel_size):
    distance_threshold = voxel_size * 0.4
#    print("\n - - Refine - -\n:: Point-to-plane ICP registration is applied on original point")
    radius_normal = voxel_size * 2
    source.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    target.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result


#==============================================================================
##=============##                 INTRINSICS                  ##=============## 
#==============================================================================

# D415 Camera:
width   = 640
height  = 480
fx      = 609.422
fy      = 608.482
cx      = 320.848
cy      = 239.221


# Create intrinsics matrix in the necessary format:
cameraIntrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
# print(cameraIntrinsics.intrinsic_matrix) 

#==============================================================================
#==============================================================================

'''
Let's try to test the algorithm step-by-step.
We will take only 2 depth frames (no color frames) and will try to match them.
'''

depth_frame_0 = my_project_path + '/ethyd_data/bottlei/depth/A_hat1_depth_frame0.png' 
depth_frame_1 = my_project_path + '/ethyd_data/bottlei/depth/A_hat1_depth_frame3.png' 

depth_source_0 = o3d.io.read_image(depth_frame_0)
depth_source_1 = o3d.io.read_image(depth_frame_1)


#---------- Visualize: ----------#
im_0 = np.asarray(depth_source_0)
plt.figure(pic_num)
plt.title('Depth frame_0')
plt.imshow(im_0)
plt.show()
pic_num +=1
#--------------------------------#

# Create point clouds from depth frames:

pcd_0 = o3d.geometry.PointCloud.create_from_depth_image(depth_source_0, cameraIntrinsics)
pcd_1 = o3d.geometry.PointCloud.create_from_depth_image(depth_source_1, cameraIntrinsics)


# Point cloud without depth truncation:
o3d.visualization.draw_geometries([pcd_0])
# o3d.visualization.draw_geometries([pcd_1])


#  ~~~~  CROP DISTANCE:  ~~~~ #

# [ object plane;  height;  depth]
# cropping table and background
bounds = [[-np.inf, np.inf], [-np.inf, 0.15], [0, 0.56]]  # set the bounds
bounding_box_points = list(itertools.product(*bounds))  # create limit points
bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(bounding_box_points))  # create bounding box object

# Crop the point cloud using the bounding box:
pcd_cropped_0 = pcd_0.crop(bounding_box)
pcd_cropped_1 = pcd_1.crop(bounding_box)

source = copy.deepcopy(pcd_cropped_0)
target = copy.deepcopy(pcd_cropped_1)

#copy.deepcopy to make copies and protect the original point clouds
pcd_cropped_0_temp = copy.deepcopy(pcd_cropped_0)
pcd_cropped_1_temp = copy.deepcopy(pcd_cropped_1)
# Point cloud after depth truncation:
pcd_cropped_0_temp.paint_uniform_color([1, 0.706, 0])
pcd_cropped_1_temp.paint_uniform_color([0, 0.651, 0.929])
o3d.visualization.draw_geometries([pcd_cropped_0_temp, pcd_cropped_1_temp])

# ! You can see the problem now: without texture, the geometry of the bottle
# doesn't change during rotation.
# ! One should use asymmetrical objects.


# Downsample pointclouds:
voxel_size  = 0.005   # 1cm
trunc       = np.inf

source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

# Visualize downsamped point cloud:
o3d.visualization.draw_geometries([source_down])


# Now, finally, the registration part!...

#==============================================================================
##=============##              ICP               ##=============## 
#==============================================================================

# As first approximation we run Ransac:
result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh, voxel_size)
# Visualize registration result:
draw_registration_result(source_down, target_down,
                             result_ransac.transformation)
# You can see that the down-sampled point clouds are nicely matched!


# ICP:
result_icp = refine_registration(source, target, voxel_size)
# Visualize registration result:
draw_registration_result(source, target, result_icp.transformation)

# You can see that the original point clouds are nicely matched!
