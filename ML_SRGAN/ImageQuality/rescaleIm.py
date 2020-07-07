'''
Created on 27-Mar-2020

@author: Neeraj Badal
'''
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from PIL import Image
from skimage import exposure
from skimage import transform
import skimage
if __name__ == "__main__":    
    upScaledDir = "D:/Mtech/FY/SEM2/ML/SuperResolutionTest/upscaled/"
    
    
#     origDir = "D:/Mtech/FY/SEM2/ML/SuperResolutionTest/WashingtonDC_System-Ready_Stereo_50cm/056082264020/056082264020_01_P001_PAN/test/downscale_4/" 
#     
#     
#     destinationDir = "D:/Mtech/FY/SEM2/ML/SuperResolutionTest/WashingtonDC_System-Ready_Stereo_50cm/056082264020/056082264020_01_P001_PAN/test/upscale_4/" 
    
    
    origDir = "D:/Mtech/FY/SEM2/ML/SuperResolutionTest/test/" 
    
    destinationDir = "D:/Mtech/FY/SEM2/ML/SuperResolutionTest/upscaled_nn_4/" 
    
    
    psnr_res_list = []
    ssim_res_list = []
    
    for i_ in range(1,11):#range(1000,1486):
#         imageName = "eg_"+str(i_)+".png"
#         srImageName = "func_"+str(i_)+"_upscaled.png"
        
        imageName = str(i_)+".png"
        srImageName = "func_"+str(i_)+"_upscaled.png"
        
        origIm = np.array(Image.open(origDir+imageName))
        
        
#         origIm = np.array(origIm)
        print("...",i_)
        d_s = transform.rescale(origIm,scale=4,order=1,clip=True,preserve_range=True,anti_aliasing=True,multichannel=True)
        d_s = d_s.astype(np.uint8)
        im = Image.fromarray(d_s)
        im.save(destinationDir+"/"+srImageName)
        
        
    
    
#     plt.figure()
#     plt.imshow(origIm)
#     
#     plt.figure()
#     plt.imshow(srIm)
#     
#     plt.show()
    

