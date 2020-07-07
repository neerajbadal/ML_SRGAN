'''
Created on 27-Mar-2020

@author: Neeraj Badal
'''
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import exposure
from skimage import transform
import skimage
from skimage.util.shape import view_as_windows
if __name__ == "__main__":    
#     upScaledDir = "D:/Mtech/FY/SEM2/ML/SuperResolutionTest/downscaled/"
#     origDir = "D:/Mtech/FY/SEM2/ML/SuperResolutionTest/original images/" 
    
#     for i_ in range(1,10):
#         imageName = "000"+str(i_)+".PNG"
#         dsImageName = str(i_)+"_downscaled_2.PNG"
#     
#         origIm = np.array(Image.open(origDir+imageName)) 
#         
#         print(origIm)
#         
# #         origIm = np.array(origIm)
#         d_s = transform.rescale(origIm,scale=0.5,clip=True,preserve_range=True,anti_aliasing=True,multichannel=True)
#         print(origIm.shape,"................",d_s.shape)
#         d_s = d_s.astype(np.uint8)
# #         plt.imshow(origIm)
# #         plt.show()
#         print(d_s)
#         
#         im = Image.fromarray(d_s)
#         im.save(upScaledDir+dsImageName)
        
    
    
#     plt.figure()
#     plt.imshow(origIm)
#     
#     plt.figure()
#     plt.imshow(srIm)
#     
#     plt.show()
    
    origDir = "D:/Mtech/FY/SEM2/ML/SuperResolutionTest/WashingtonDC_System-Ready_Stereo_50cm/056082264020/056082264020_01_P001_PAN/"
    imgName = "12SEP21160913-P1BS-056082264020_01_P001.TIF"
    Image.MAX_IMAGE_PIXELS = None
    im = np.array(Image.open(origDir+imgName))
    print(im.shape)
    
    min_i = im.min()
    max_i = im.max()
    min_o = 0
    max_o = 255
     
#     hist,bins = np.histogram(im)
#     plt.plot(bins[1:],hist)
#     plt.show()
     
    cs_im = (im - min_i)*(((max_o-min_o)/(max_i-min_i))+min_o)
    cs_im = cs_im.astype(np.uint8)
     
    print(min_i,".......",max_i)
     
    print(cs_im.min(),"........",cs_im.max())
    
    img_rescale = exposure.equalize_hist(cs_im)
       
    min_i = img_rescale.min()
    max_i = img_rescale.max()
    min_o = 0
    max_o = 255
       
    cs_im = (img_rescale - min_i)*(((max_o-min_o)/(max_i-min_i))+min_o)
    cs_im = cs_im.astype(np.uint8)
    
    
    
    windowRowSize = 768
    windowColSize = 768
    window_shape = (windowRowSize,windowColSize)
    im_patches = view_as_windows(cs_im, window_shape,step=windowRowSize)
    print(im_patches.shape)
    
    im_patches = im_patches.reshape(im_patches.shape[0]*im_patches.shape[1],im_patches.shape[2],im_patches.shape[3])
    print(im_patches.shape)
    
    np.random.shuffle(im_patches)
    
    trainDir = "D:/Mtech/FY/SEM2/ML/SuperResolutionTest/WashingtonDC_System-Ready_Stereo_50cm/056082264020/056082264020_01_P001_PAN/train/"
    testDir = "D:/Mtech/FY/SEM2/ML/SuperResolutionTest/WashingtonDC_System-Ready_Stereo_50cm/056082264020/056082264020_01_P001_PAN/test/"
    downScaleDir = "D:/Mtech/FY/SEM2/ML/SuperResolutionTest/WashingtonDC_System-Ready_Stereo_50cm/056082264020/056082264020_01_P001_PAN/test/downscale_4/"
    
    count_i = 0
    for ims in im_patches:
        test_image = skimage.color.gray2rgb(ims)
        print("processed ",count_i)
        if count_i < 1000:
            im = Image.fromarray(test_image)
            im.save(trainDir+"eg_"+str(count_i)+".png")
            
        else:
            im = Image.fromarray(test_image)
            im.save(testDir+"eg_"+str(count_i)+".png")
            d_s = transform.rescale(test_image,scale=0.25,clip=True,preserve_range=True,anti_aliasing=True,multichannel=True)
            d_s = d_s.astype(np.uint8)
            im = Image.fromarray(d_s)
            im.save(downScaleDir+"eg_"+str(count_i)+".png")
            
            
        count_i = count_i + 1
        
#         if count_i > 20:
#             exit(0)
    
    
#     plt.imshow(im_patches[25304,34410],cmap='gray')
#     plt.show()
    
    exit(0)
    
    plt.imshow(cs_im[:5000,:5000],cmap='gray')
    plt.show()
