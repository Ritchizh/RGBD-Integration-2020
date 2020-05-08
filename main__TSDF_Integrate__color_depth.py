"""
Created on Fri Mar 13 12:28:52 2020
@author: Margarita Chizh


Based on Open3D Tutorial: http://www.open3d.org/docs/release/tutorial/Advanced/rgbd_integration.html

# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details
"""

import open3d            as o3d
import numpy             as np
import matplotlib.pyplot as plt

plt.close('all')
pic_num = 1

#==============================================================================
#==============================================================================

file_name   = 'A_hat1'      # ! - prefix of files names in Test_data folder

# PREPARE FRAMES NUMBERS:
num_Frames      = 132       # ! - number of Frames in Test_data folder
skip_N_frames   = 10        # ! - Can choose the range of integrated Frames - less frames = faster run
frames_nums     = np.arange(0, num_Frames+1, skip_N_frames)  

#==============================================================================
#==============================================================================

#  == ICP functions: ==

# check: http://www.open3d.org/docs/release/tutorial/Basic/icp_registration.html

def draw_registration_result(source, target, transformation, title='Title'):
    source_temp = source
    target_temp = target
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
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
#    print("\n- - Global registration - -\n:: RANSAC registration on downsampled point clouds.")
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))
    return result


def refine_registration(source, target, voxel_size):
    distance_threshold = voxel_size * 0.4
#    print("\n - - Refine - -\n:: Point-to-plane ICP registration is applied on original point")
    radius_normal = voxel_size * 2
    source.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    target.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    result = o3d.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.registration.TransformationEstimationPointToPlane())
    return result


#==============================================================================
##=============##           TSDF PARAMETERS                  ##=============## 
#==============================================================================

voxel_size  = 0.01   # 1cm
trunc       = np.inf  # Maximum depth limit. The Test depth frames were already truncated during Subject segmentation.
#==============================================================================
##=============##                 INTRINSICS                  ##=============## 
#==============================================================================

# For Intel RealSense to get Intrinsics use command: color_frame.profile.as_video_stream_profile().intrinsics
# Intrinsics:
width   = 1280
height  = 720
fx      = 920.003
fy      = 919.888
cx      = 640.124
cy      = 358.495
#scale   = 0.0010000000474974513 #converts [mm] to [meters]

# Create intrinsics matrix in the necessary format:
cameraIntrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

# print(cameraIntrinsics.intrinsic_matrix) 
 
#==============================================================================
##=============##               /ODOMETRY/  ICP               ##=============## 
#==============================================================================

# Instead of using Odometry, which is not accurate, we use ICP
# to find transformation of one frame into the next frame

num_of_poses    = len(frames_nums)

#  == Create a trajectory .log file: ==
from trajectory_io import *
metadata    = [0, 0, 0]
traj        = []
transform_0   = np.identity(4) # The first pose is the reference - we save identity transform matrix
traj.append(CameraPose(metadata, transform_0))
        
for i in range(num_of_poses-1): 
    # SOURCE frame:
    color_source = o3d.io.read_image("Test_data/%s_color_frame%s.jpg"%(file_name, frames_nums[i]))
    depth_source = o3d.io.read_image("Test_data/%s_depth_frame%s.png"%(file_name, frames_nums[i]))
    
    rgbd_source = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_source, depth_source, 
            depth_trunc=trunc, # truncate the depth at z  =
            convert_rgb_to_intensity=False)
    source         = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_source, cameraIntrinsics)
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
   
    # TARGET frame:
    color_target = o3d.io.read_image("Test_data/%s_color_frame%s.jpg"%(file_name, frames_nums[i+1]))
    depth_target = o3d.io.read_image("Test_data/%s_depth_frame%s.png"%(file_name, frames_nums[i+1]))
    
    rgbd_target = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_target, depth_target, 
            depth_trunc=trunc, # truncate the depth at z  =
            convert_rgb_to_intensity=False)
    
    target   = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_target, cameraIntrinsics) 
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    #----------------------------------------------------------------------
    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh, voxel_size)
    result_icp = refine_registration(source, target, voxel_size)
    
    #draw_registration_result(source, target, result_icp.transformation, title='ICP result')
    
    ## Append to camera poses:
    transform   = result_icp.transformation
    transform   = np.dot(transform, transform_0)
    transform_0 = transform
    traj.append(CameraPose(metadata, transform))
        

# == Generate .log file from ICP transform: ==   
write_trajectory(traj, "test_segm.log")
camera_poses = read_trajectory("test_segm.log")


#==============================================================================
##=============##         TSDF VOLUME INTEGRATION             ##=============## 
#==============================================================================

#  == TSDF volume integration: ==

volume = o3d.integration.ScalableTSDFVolume(
    voxel_length = 0.01, # meters # ~ 1cm
    sdf_trunc   =  0.05, # meters # ~ several voxel_lengths
    color_type  =  o3d.integration.TSDFVolumeColorType.RGB8)

print('\nnumber of camera_poses:', len(camera_poses),'\n')


num_cam_pose = 0
for i in frames_nums:
    print("Integrate %s-th image into the volume."%i)
    
    color = o3d.io.read_image("Test_data/%s_color_frame%s.jpg"%(file_name, i))
    depth = o3d.io.read_image("Test_data/%s_depth_frame%s.png"%(file_name, i))
    
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color, depth, 
    depth_trunc=trunc, # truncate the depth at z  =
    convert_rgb_to_intensity=False)
    
    volume.integrate(rgbd, cameraIntrinsics, camera_poses[num_cam_pose].pose)
    num_cam_pose +=1

#==============================================================================
##=============##              TRIANGULAR MESH                ##=============## 
#==============================================================================
    
# #  == Extract a triangle mesh from the volume (with the marching cubes algorithm) 
# #     and visualize it : ==
     
mesh = volume.extract_triangle_mesh()
print(mesh.compute_vertex_normals())

mesh.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) # flip!

o3d.visualization.draw_geometries([mesh])


# Save the resulting triangle mesh, load it back and visualize:

o3d.io.write_triangle_mesh('Mesh__%s__every_%sth_%sframes.ply'%(file_name,skip_N_frames,len(camera_poses)), mesh)
meshRead = o3d.io.read_triangle_mesh('Mesh__%s__every_%sth_%sframes.ply'%(file_name,skip_N_frames,len(camera_poses)))
o3d.visualization.draw_geometries([meshRead])