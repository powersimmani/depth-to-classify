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

if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread

# colors for drawing different bodies 
SKELETON_COLORS = [pygame.color.THECOLORS["red"],
                    pygame.color.THECOLORS["blue"], 
                    pygame.color.THECOLORS["green"],
                    pygame.color.THECOLORS["orange"], 
                    pygame.color.THECOLORS["purple"], 
                    pygame.color.THECOLORS["yellow"], 
                    pygame.color.THECOLORS["violet"]]


class DepthRuntime(object):
    def __init__(self):
        pygame.init()
        self.save_flag_color = False
        self.save_flag_depth = False
        self.cnt =0 

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Loop until the user clicks the close button.
        self._done = False

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Kinect runtime object, we want only color and body frames 
        self._kinect_depth = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)

        # back buffer surface for getting Kinect depth frames, 8bit grey, width and height equal to the Kinect color frame size
        self._frame_surface_depth = pygame.Surface((self._kinect_depth.depth_frame_desc.Width, self._kinect_depth.depth_frame_desc.Height), 0, 24)
        # here we will store skeleton data 
        self._bodies = None
        
        # Set the width and height of the screen [width, height]
        self._infoObject = pygame.display.Info()
        self._screen_depth = pygame.display.set_mode((self._kinect_depth.depth_frame_desc.Width, self._kinect_depth.depth_frame_desc.Height), 
                                                pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)

        pygame.display.set_caption("Kinect for Windows v2 Depth")

        # Set the width and height of the screen [width, height]
        self._infoObject = pygame.display.Info()
        self._screen_color = pygame.display.set_mode((self._infoObject.current_w >> 1, self._infoObject.current_h >> 1), 
                                               pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Kinect runtime object, we want only color and body frames 
        self._kinect_color = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)


        # back buffer surface for getting Kinect color frames, 32bit color, width and height equal to the Kinect color frame size
        self._frame_surface_color = pygame.Surface((self._kinect_color.color_frame_desc.Width, self._kinect_color.color_frame_desc.Height), 0, 32)
        # here we will store skeleton data 




    def draw_color_frame(self, frame, out_path):
        if frame is None:  # some usb hub do not provide the depth image. it works with Kinect studio though
            return
        a = np.reshape(frame,(1080,1920,4))

        a[:,:,0],a[:,:,1],a[:,:,2] = a[:,:,2].copy(),a[:,:,1].copy(),a[:,:,0].copy()
        im =  Image.fromarray(a).convert("RGB")
        im.save(out_path+"/"+"color_"+str(self.cnt) + ".jpg")           

        

    def draw_depth_frame(self, frame, out_path):
        if frame is None:  # some usb hub do not provide the depth image. it works with Kinect studio though
            return
        f8=np.uint8(frame.clip(1,4000)/16.)#clip을 이용해서 최소값을 0 최대 값을 250으로 바꾼다. 
        frame8bit=np.dstack((f8,f8,f8))# RGB값 전부 같은 값으로 바꾼다. 

        a = np.reshape(frame,(424,512))
        b = np.reshape(frame8bit,(424,512,3))

        im =  Image.fromarray(b).convert("RGB")
        im.save(out_path+"/"+"depth_"+str(self.cnt) + ".jpg")

        with open(out_path+"/"+'depth_raw_'+str(self.cnt)+".pck", 'wb') as f:
            pickle.dump(a, f)

    def stream_color_frame(self, frame, target_surface):
        if frame is None:  # some usb hub do not provide the depth image. it works with Kinect studio though
            return
        target_surface.lock()
        address = self._kinect_color.surface_as_array(target_surface.get_buffer())

        a = np.reshape(frame,(1080,1920,4))
        #a = cv2.resize(a, dsize=(512, 424), interpolation=cv2.INTER_AREA)
        a = cv2.resize(a, dsize=(304, 228), interpolation=cv2.INTER_AREA)
        a = cv2.flip(a,1)
        cv2.imshow("color",a)
        cv2.waitKey(1)
        

    def stream_depth_frame(self, frame, target_surface):
        if frame is None:  # some usb hub do not provide the depth image. it works with Kinect studio though
            return
        target_surface.lock()
        #f8=np.uint8(frame.clip(1,4000)/16.)#clip을 이용해서 최소값을 0 최대 값을 250으로 바꾼다. 
        #frame8bit=np.dstack((f8,f8,f8))# RGB값 전부 같은 값으로 바꾼다. 

        f82 = np.uint8((frame / frame.max())*255) 
        frame8bit2=np.dstack((f82,f82,f82))     
        a2 = np.reshape(frame8bit2,(424,512,3))   

        b = cv2.resize(a2, dsize=(304, 228), interpolation=cv2.INTER_AREA)
        b = cv2.applyColorMap(b, cv2.COLORMAP_JET)
        cv2.imshow("depth",b)
        cv2.waitKey(1)

    def rgb_to_print(self,r,g,b):
        return r << 16 | g << 8 | b

    def point_cloud(self,depth_frame,color_frame):
        # size must be what the kinect offers as depth frame
        L = depth_frame.size
        # create C-style Type
        TYPE_CameraSpacePoint_Array = PyKinectV2._CameraSpacePoint * L

        # instance of that type
        csps = TYPE_CameraSpacePoint_Array()
        # cstyle pointer to our 1d depth_frame data
        ptr_depth = np.ctypeslib.as_ctypes(depth_frame.flatten())
        # calculate cameraspace coordinates
        error_state = self._kinect_depth._mapper.MapDepthFrameToCameraSpace(L, ptr_depth,L, csps)
        # 0 means everythings ok, otherwise failed!
        if error_state:
            raise "Could not map depth frame to camera space! "
            + str(error_state)

        # convert back from ctype to numpy.ndarray
        pf_csps = ctypes.cast(csps, ctypes.POINTER(ctypes.c_float))
        position_data = np.copy(np.ctypeslib.as_array(pf_csps, shape=(L,3)))
        
        color_data = np.zeros(shape=(L), dtype=np.int)
        color_frame = np.reshape(color_frame,(1080,1920,4))

        null_cnt = [] 
        for index in range(L):
            a = self._kinect_depth._mapper.MapCameraPointToColorSpace(csps[index]) 
            if ( 0<= a.y and a.y <=1080 and 0<= a.x  and a.x <= 1800):
                r,g,b = color_frame[int(a.y)][int(a.x)][:3]
                color_data[index]  = self.rgb_to_print(b,g,r)
            else:
                null_cnt.append(index)
        
        #code.interact(local = dict(globals(), **locals()))
        position_data = np.delete(position_data,null_cnt,axis = 0)
        color_data = np.delete(color_data,null_cnt,axis = 0)
        del pf_csps, csps, ptr_depth, TYPE_CameraSpacePoint_Array

        self._kinect_color 
        return position_data,color_data

    def write_pcd_file(self,position_data,color_data):
        L = len(color_data)
        f_out = open("temp.pcd","w")
        f_out.write("# .PCD v.7 - Point Cloud Data file format\n")
        f_out.write("VERSION .7\n")
        f_out.write("FIELDS x y z rgb\n")
        f_out.write("SIZE 4 4 4 4\n")
        f_out.write("TYPE F F F U\n")
        f_out.write("COUNT 1 1 1 1\n")
        f_out.write("WIDTH "+str(L)+"\n")
        f_out.write("HEIGHT 1\n")
        f_out.write("VIEWPOINT 180 0 180 0 0 0 0\n")
        f_out.write("POINTS "+str(L)+"\n")
        f_out.write("DATA ascii\n")

        for index in range(L):
            x,y,z = position_data[index]
            f_out.write(str(x) + " " + str(y) + " " + str(z)+ " " + str(color_data[index])+"\n")
        f_out.close()
        pass
    def write_ply_file(self,position_data,color_data):
        L = len(color_data)
        f_out = open("temp.ply","w")
        f_out.write("ply\n")
        f_out.write("format ascii 1.0\n")
        f_out.write("element vertex "+str(L) + "\n")
        f_out.write("property float x\n")
        f_out.write("property float y\n")
        f_out.write("property float z\n")
        f_out.write("property uchar red\n")
        f_out.write("property uchar green\n")
        f_out.write("property uchar blue\n")
        f_out.write("end_header\n")

        for index in range(L):
            x,y,z = position_data[index]
            f_out.write(str(x) + " " + str(y) + " " + str(z)+ " "+str(color_data[index] >> 16)+ " "+str(color_data[index] >> 8 & 255)+ " "+str(color_data[index] & 255)+"\n")
        f_out.close()

    def run(self):
        out_path = "output"
        depth_frame = None
        color_frame = None
        # -------- Main Program Loop -----------
        while not self._done:
            # --- Main event loop
            for event in pygame.event.get(): # User did something
                if event.type == pygame.QUIT: # If user clicked close
                    self._done = True # Flag that we are done so we exit this loop

                elif event.type == pygame.VIDEORESIZE: # window resized
                    self._screen = pygame.display.set_mode(event.dict['size'], 
                                                pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)
                elif event.type == pygame.KEYDOWN:
                    self.save_flag_color = True              
                    self.save_flag_depth = True              

                    

            # --- Getting frames and drawing  
            self.cnt = self.cnt + 1;
            if self._kinect_depth.has_new_depth_frame():
                depth_frame = self._kinect_depth.get_last_depth_frame()
                self.stream_depth_frame(depth_frame, self._frame_surface_depth)
                #depth_frame = None

            if self._kinect_color.has_new_color_frame():
                color_frame = self._kinect_color.get_last_color_frame()
                self.stream_color_frame(color_frame, self._frame_surface_color)
                #self.fed_model(frame)
                #color_frame = None

            if (depth_frame is not None and color_frame is not None):
                position_data,color_data = self.point_cloud(depth_frame,color_frame)
                self.write_ply_file(position_data,color_data)
                self.take_screen_shot()
                pcd = o3d.io.read_point_cloud("temp.ply")
                o3d.visualization.draw_geometries([pcd])
                code.interact(local = dict(globals(), **locals()))

                
            depth_frame = None
            color_frame = None

            if self.save_flag_depth ==True or self.save_flag_color ==True:
                if self._kinect_depth.has_new_depth_frame():
                    frame = self._kinect_depth.get_last_depth_frame()
                    self.draw_depth_frame(frame,out_path)
                    frame = None
                    self.save_flag_depth = False

                if self._kinect_color.has_new_color_frame():
                    frame = self._kinect_color.get_last_color_frame()
                    self.draw_color_frame(frame ,out_path)
                    frame = None      
                    self.save_flag_color = False
                
            # --- Limit to 60 frames per second
            self._clock.tick(60)
            pygame.display.update()
        # Close our Kinect sensor, close the window and quit.
        pygame.quit()


__main__ = "Kinect v2 Depth"
global cnt
cnt = 0
game =DepthRuntime();
game.run();