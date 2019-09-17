from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import pickle
import code,os
import ctypes
import _ctypes
import pygame
import sys,cv2, torch,decimal    
import numpy as np
import open3d as o3d
from PIL import Image
from torch.autograd import Variable

def custom_draw_geometry_with_custom_fov(pcd, fov_step):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    print("Field of view (before changing) %.2f" % ctr.get_field_of_view())
    ctr.change_field_of_view(step=fov_step)
    print("Field of view (after changing) %.2f" % ctr.get_field_of_view())
    vis.run()
    vis.destroy_window()

def custom_draw_geometry(pcd):
    # The following code achieves the same effect as:
    # o3d.visualization.draw_geometries([pcd])
    """
    vis = o3d.visualization.Visualizer()

    vis.create_window(visible = False)
    vis.add_geometry(pcd)

    a  = vis.get_view_control()
    a.rotate(900,0)
    #vis.get_view_control(view_control)
    vis.run()
    vis.capture_screen_image("temp.png", do_render=False)
    vis.destroy_window()
    vis.close()
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=960,height=540)
    
    vis.add_geometry(pcd)
    a  = vis.get_view_control()

    a.rotate(900,0)
    a.change_field_of_view(500)

    #vis.update_geometry()
    #vis.poll_events()
    #vis.update_renderer()
    vis.run()
    vis.capture_screen_image("temp.png")
    code.interact(local = dict(globals(), **locals()))
    vis.destroy_window()

def execute():
    while(True):
        pcd = o3d.io.read_point_cloud("temp.ply")
        #ply = o3d.io.
        custom_draw_geometry(pcd)
        
    pass


execute()