import pyrealsense2 as rs
import cv2
import numpy as np
import os.path
import matplotlib.pyplot as plt

plt.close('all')

imW = 1280
imH = 720
combiImageScaleFactor = 2
recordsDir = '.\\records\\'
filename    = 'record_A_1.bag'
filename_bg = 'record_A_1_bgr.bag'

kernel = np.ones((9,9),np.uint8)

if os.path.isfile(f'{recordsDir}{filename}'):
    inputFileName = f'{recordsDir}{filename}'
else:
    print('File not found!')
    quit

if os.path.isfile(f'{recordsDir}{filename_bg}'):
    inputFileNameBg = f'{recordsDir}{filename_bg}'
else:
    print('File not found!')
    quit

imHScaled = int(imH//combiImageScaleFactor)
imWScaled = int(imW//combiImageScaleFactor)
align = rs.align(rs.stream.color)

hole_filling = rs.hole_filling_filter()

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
# fgbg.setHistory(200)
# fgbg.setBackgroundRatio(0.7)

pipeline = rs.pipeline()
config = rs.config()

config.enable_device_from_file(inputFileNameBg, True)
profile = pipeline.start(config)

bg = np.zeros((imHScaled, imWScaled, 3), dtype=np.float64)

for i in range(10):
    frames = pipeline.wait_for_frames(1000)
    frames = align.process(frames)                
    
    colorFrame = frames.get_color_frame()

    colorImageOrig = np.asanyarray(colorFrame.get_data())
    colorImage = cv2.cvtColor(colorImageOrig, cv2.COLOR_RGB2BGR)
    colorImage = cv2.resize(colorImage, dsize=(imWScaled, imHScaled), interpolation=cv2.INTER_LINEAR)
    
    fgmask = fgbg.apply(colorImage)
    
    bg = (bg*i + colorImage)/(i+1)
    bgImage = np.uint8(bg)

pipeline.stop()

cv2.namedWindow('RS_streams', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('Mask', cv2.WINDOW_AUTOSIZE)


pipeline = rs.pipeline()
config = rs.config()

config.enable_device_from_file(inputFileName, True)
profile = pipeline.start(config)

# fgbg.setHistory(10000000)
# fgbg.setBackgroundRatio(0.7)

while(True):
    frames = pipeline.wait_for_frames(1000)
    frames = align.process(frames)                
    
    colorFrame = frames.get_color_frame()

    depthFrame = frames.get_depth_frame()
    depthFrame = hole_filling.process(depthFrame)
    
    irFrame    = frames.get_infrared_frame(2)

    colorImageOrig = np.asanyarray(colorFrame.get_data())
    colorImage = cv2.cvtColor(colorImageOrig, cv2.COLOR_RGB2BGR)
    colorImage = cv2.resize(colorImage, dsize=(imWScaled, imHScaled), interpolation=cv2.INTER_LINEAR)
    
    depthColorImage = np.asanyarray(rs.colorizer().colorize(depthFrame).get_data())
    depthColorImage = cv2.resize(depthColorImage, dsize=(imWScaled, imHScaled), interpolation=cv2.INTER_LINEAR)
    depth = np.asanyarray(depthFrame.get_data())
    
    maskBG = cv2.absdiff(colorImage, bgImage)    
    maskBG = np.uint8(np.sum(maskBG, axis=2))
    
    maskBG = cv2.threshold(maskBG, 30, 1, cv2.THRESH_BINARY)[1]
    
    maskBG = cv2.morphologyEx(maskBG, cv2.MORPH_CLOSE, kernel)
    maskBG = cv2.morphologyEx(maskBG, cv2.MORPH_OPEN, kernel)    
    
    maskDepth = (depth > 300) & (depth < 2000)
    maskDepth = 255*np.uint8(maskDepth)
    maskDepth = cv2.resize(maskDepth, dsize=(imWScaled, imHScaled), interpolation=cv2.INTER_LINEAR)    
    maskDepth = maskDepth > 127
    
    maskFinal = maskBG & maskDepth
    maskFinal = 255*np.uint8(maskFinal)
    
    maskDepthColor = np.repeat(cv2.resize(maskFinal, dsize=(imWScaled, imHScaled), interpolation=cv2.INTER_LINEAR)[:, :, np.newaxis], 3, axis=2)
    maskDepthColor = np.float32(maskDepthColor)/255
    
    combiImage = np.hstack( (colorImage, np.uint8(depthColorImage*maskDepthColor)) )
    
    fgmask = fgbg.apply(colorImage)
    
    cv2.imshow('RS_streams', combiImage)
    cv2.imshow('Mask', fgmask)
    
    key = cv2.waitKey(10)
    
    if key == 27:
        cv2.destroyAllWindows()
        pipeline.stop()
        break
    


