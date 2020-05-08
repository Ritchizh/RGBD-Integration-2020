"""
Created on Fri May  8 16:39:00 2020

@author: Margarita Chizh


Based on Open3D Tutorial: http://www.open3d.org/docs/release/tutorial/Advanced/rgbd_integration.html

# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details
"""

import open3d            as o3d

meshRead = o3d.io.read_triangle_mesh('Mesh__A_hat1__every_1th_133frames.ply')
o3d.visualization.draw_geometries([meshRead])

