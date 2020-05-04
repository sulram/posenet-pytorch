import argparse
import pickle
import time
import torch
import cv2
import numpy as np
import posenet
import SpoutSDK
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from pythonosc import udp_client

"""parsing and configuration"""
def parse_args():
    desc = "Spout Posenet"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--model', type=int, default=101)
    parser.add_argument('--size', nargs = 2, type=int, default=[640, 480], help='Width and height of texture')   
    parser.add_argument('--spout_name', type=str, default='TDSyphonSpoutOut', help='Spout receiving name - the name of the sender you want to receive')   
    parser.add_argument('--scale_factor', type=float, default=0.7125)
    return parser.parse_args()


"""main"""
def main():

    # parse arguments
    args = parse_args()

    # init model

    model = posenet.load_model(args.model)
    model = model.cuda()
    output_stride = model.output_stride
    
    # window details
    width = args.size[0] 
    height = args.size[1] 
    display = (width,height)
    
    # window setup
    pygame.init() 
    pygame.display.set_caption('Spout Receiver')
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    # SPOUT INIT

    receiverName = args.spout_name 
    spoutReceiver = SpoutSDK.SpoutReceiver()
    spoutReceiver.pyCreateReceiver(receiverName, width, height, False)

    spoutSender = SpoutSDK.SpoutSender()
    spoutSender.CreateSender('Posenet', width, height, 0)

    # OSC
    client = udp_client.SimpleUDPClient('127.0.0.1', 7000)

    # OpenGL init
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0,width,height,0,1,-1)
    glMatrixMode(GL_MODELVIEW)
    glDisable(GL_DEPTH_TEST)
    glClearColor(0.0,0.0,0.0,0.0)
    glEnable(GL_TEXTURE_2D)

    # create texture for spout receiver
    textureReceiveID = glGenTextures(1)    
    
    # initalise receiver texture
    glBindTexture(GL_TEXTURE_2D, textureReceiveID)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

    # copy data into texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None ) 
    glBindTexture(GL_TEXTURE_2D, 0)

    frame = 0

    # loop for graph frame by frame
    while(True):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                spoutReceiver.ReleaseReceiver()
                pygame.quit()
                quit()
        
        # SPOUT receive texture
        spoutReceiver.pyReceiveTexture(receiverName, width, height, textureReceiveID.item(), GL_TEXTURE_2D, False, 0)
        glBindTexture(GL_TEXTURE_2D, textureReceiveID)

        # copy pixel byte array from received texture
        data = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, outputType=None)  #Using GL_RGB can use GL_RGBA 
        
        # swap width and height data around due to oddness with glGetTexImage. http://permalink.gmane.org/gmane.comp.python.opengl.user/2423
        data.shape = (data.shape[1], data.shape[0], data.shape[2])

        # POSENET from webcam_demo

        input_image, display_image, output_scale = posenet.read_tex(data)

        with torch.no_grad():
            input_image = torch.Tensor(input_image).cuda()

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(0),
                offsets_result.squeeze(0),
                displacement_fwd_result.squeeze(0),
                displacement_bwd_result.squeeze(0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.15)

        keypoint_coords *= output_scale

        frame = (frame+1) % 300

        #if frame % 4 == 0:
        for i, pose in enumerate(pose_scores):
            if pose_scores[i] > 0.05:
                client.send_message("/pose_{}".format(i+1), pickle.dumps(keypoint_coords[i].tolist()))
                client.send_message("/pose_{}_active".format(i+1), pickle.dumps(True))
            else:
                client.send_message("/pose_{}_active".format(i+1), pickle.dumps(False))


        # POSENET DRAW
        overlay_image = posenet.draw_skel_and_kp(
            display_image, pose_scores, keypoint_scores, keypoint_coords,
            min_pose_score=0.15, min_part_score=0.1)
        
        # bind FLIPPED POSENET texture
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, np.flipud(overlay_image))

        # SPOUT send texture
        spoutSender.SendTexture(textureReceiveID.item(), GL_TEXTURE_2D, width, height, True, 0)

        # bind POSENET texture
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, overlay_image)
        
        # setup window to draw to screen
        glActiveTexture(GL_TEXTURE0)

        # clean start
        glClear(GL_COLOR_BUFFER_BIT  | GL_DEPTH_BUFFER_BIT )
        # reset drawing perspective
        glLoadIdentity()

        # draw texture on screen
        # glPushMatrix() use these lines if you want to scale your received texture
        # glScale(0.3, 0.3, 0.3)
        glBegin(GL_QUADS)

        glTexCoord(0,0)        
        glVertex2f(0,0)

        glTexCoord(1,0)
        glVertex2f(width,0)

        glTexCoord(1,1)
        glVertex2f(width,height)

        glTexCoord(0,1)
        glVertex2f(0,height)
        
        glEnd()
        # glPopMatrix() make sure to pop your matrix if you're doing a scale        
        # update window
        pygame.display.flip()        

if __name__ == '__main__':
    main()
