'''
Created on 27-Mar-2020

@author: Neeraj Badal
'''
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from skimage import measure

def fft2Dimensional(img,norm_mathod='DC'):
    print('Current Normalization Method : '+str(norm_mathod))
    
    
    
    # got magnitude spectrum
    f=np.fft.fft2(img)
    
#    f=np.fft.fft(img[:,0])
    fshift = np.fft.fftshift(f)
    Mag_spec= np.abs(fshift)

    print(Mag_spec.shape)
    
    rows=int(Mag_spec.shape[0])
    cols=int(Mag_spec.shape[1])
    
  ###   start conversion to polar image
    r=np.max([rows/2,cols/2])
    r=int(r)
    print(r,rows,cols)
    MpolI=np.zeros((r,360))
#    print(MpolI,Mag_spec)
    for rho in np.arange(0,r):
        for theta in np.arange(0,360):
            x=rho*np.cos(np.deg2rad(theta))+rows/2
            y=-rho*np.sin(np.deg2rad(theta))+cols/2
    
            if x<rows and y<cols and x>0 and y>0:
                MpolI[rho,theta]=Mag_spec[int(x),int(y)]
 
    print("mpolshape",MpolI.shape)
    if norm_mathod=='DC':
        avgR=np.mean(MpolI,axis=1)
        avgR/=avgR[0]
        
    if norm_mathod=='AREA':
        MpolI[1:,:]=MpolI[1:,:]/Mag_spec.mean()
        avgR=np.mean(MpolI[1:,:],axis=1)
#  to get positive values 
    avgR=20*np.log10(avgR)
    print("shape of avgR",avgR.shape,len(avgR),rows)
    avgR = np.squeeze(avgR)
    
    avgrIndex = np.arange(0,len(avgR))
    avgrIndex = avgrIndex / max(rows,cols)

    avgR = (avgR - np.min(avgR) ) / (np.max(avgR) - np.min(avgR))

    return [avgrIndex,avgR]
    
#     plt.figure()
#     plt.plot(avgrIndex,avgR)
#     plt.show()


if __name__ == "__main__":    
#     upScaledDir = "D:/Mtech/FY/SEM2/ML/SuperResolutionTest/upscaled/"
#     origDir = "D:/Mtech/FY/SEM2/ML/SuperResolutionTest/original images/" 
    
#     origDir = "D:/Mtech/FY/SEM2/ML/SuperResolutionTest/WashingtonDC_System-Ready_Stereo_50cm/056082264020/056082264020_01_P001_PAN/Old_Zips/test/" 
#     destinationDir = "D:/Mtech/FY/SEM2/ML/SuperResolutionTest/WashingtonDC_System-Ready_Stereo_50cm/056082264020/056082264020_01_P001_PAN/Old_Zips/test/upscale_4/"
#     srDir = "D:/Mtech/FY/SEM2/ML/SuperResolutionTest/WashingtonDC_System-Ready_Stereo_50cm/056082264020/056082264020_01_P001_PAN/Old_Zips/test/Satellite Upscaled New/"
    
#     origDir = "D:/Mtech/FY/SEM2/ML/SuperResolutionTest/WashingtonDC_System-Ready_Stereo_50cm/056082264020/056082264020_01_P001_PAN//test/" 
#     destinationDir = "D:/Mtech/FY/SEM2/ML/SuperResolutionTest/WashingtonDC_System-Ready_Stereo_50cm/056082264020/056082264020_01_P001_PAN//test/upscale_4/"
#     srDir = "D:/Mtech/FY/SEM2/ML/SuperResolutionTest/WashingtonDC_System-Ready_Stereo_50cm/056082264020/056082264020_01_P001_PAN/test/upscaled_fresh_dataset/"
     
#     origDir = "D:/Mtech/FY/SEM2/ML/SuperResolutionTest/original images/" 
#     destinationDir = "D:/Mtech/FY/SEM2/ML/SuperResolutionTest/upscaled_nn_4/" 
#     srDir = "D:/Mtech/FY/SEM2/ML/SuperResolutionTest/upscaled/" #sr vgg
    
    origDir = "D:/Mtech/FY/SEM2/ML/SuperResolutionTest/WashingtonDC_System-Ready_Stereo_50cm/056082264020/056082264020_01_P001_PAN//test/" 
    destinationDir = "D:/Mtech/FY/SEM2/ML/SuperResolutionTest/WashingtonDC_System-Ready_Stereo_50cm/056082264020/056082264020_01_P001_PAN//test/upscale_4/"
    srDir = "D:/Mtech/FY/SEM2/ML/SuperResolutionTest/WashingtonDC_System-Ready_Stereo_50cm/056082264020/056082264020_01_P001_PAN/test/upscaled_fresh_dataset/"
    avsr = "D:/Mtech/FY/SEM2/ML/SuperResolutionTest/WashingtonDC_System-Ready_Stereo_50cm/056082264020/056082264020_01_P001_PAN/test/avsrgan_up/"
    
    psnr_res_list = []
    ssim_res_list = []
    
    for i_ in range(9,10):#[1204,1247]:#range(1000,1485):
        imageName = "000"+str(i_)+".PNG"
        srImageName = str(i_)+"_upscaled.PNG"
        
#         print("...Test set ..",i_)
#         imageName = "eg_"+str(i_)+".png"
#         srImageName = "eg_"+str(i_)+"_upscaled.png"
        
        skImageName = "func_"+str(i_)+"_upscaled.png"
        
        avsrImageName = "up4_eg_"+str(i_)+".png"
        
        origIm = plt.imread(origDir+imageName, format='png')
        
        
        srIm = plt.imread(srDir+srImageName, format='png')
        
        skIm = plt.imread(destinationDir+skImageName, format='png')
        
        avIm = plt.imread(destinationDir+avsrImageName, format='png')
        
        origIm = np.array(origIm)
        srIm = np.array(srIm)
        skIm = np.array(skIm)
        avIm = np.array(avIm)
        
        out_psnr = measure.compare_psnr(origIm[:,:,0], srIm[:,:,0])
        ssim_val = measure.compare_ssim(origIm[:,:,0], srIm[:,:,0])
        
        out_psnr_sk = measure.compare_psnr(origIm[:,:,0], skIm[:,:,0])
        ssim_val_sk = measure.compare_ssim(origIm[:,:,0], skIm[:,:,0])
        
        out_psnr_av = measure.compare_psnr(origIm[:,:,0], avIm[:,:,0])
        ssim_val_av = measure.compare_ssim(origIm[:,:,0], avIm[:,:,0])
       
        
        psnr_res_list.append([out_psnr,out_psnr_sk])
        ssim_res_list.append([ssim_val,ssim_val_sk])
    
        orig_ps = fft2Dimensional(origIm[:,:,0])
        sr_ps = fft2Dimensional(srIm[:,:,0])
        sk_ps = fft2Dimensional(skIm[:,:,0])
         
        plt.plot(orig_ps[0],orig_ps[1],label='Orig Image')
        plt.plot(orig_ps[0],sr_ps[1],label='SR Image')
        plt.plot(orig_ps[0],sk_ps[1],label='SK Image')
        plt.xlabel("spatial frequency cycles per pixel",fontsize=16)
        plt.ylabel("normalized power",fontsize=16)
        plt.title("Power Spectrum Comparison between Super-resolved (SR) and Nearest-Neighbor Recaled (SK) Test set "+str(i_),fontsize=20)
        plt.legend()
        plt.show()
    
    
    psnr_res_list = np.array(psnr_res_list)
    ssim_res_list = np.array(ssim_res_list)
    plt.figure()
    plt.plot(psnr_res_list[:,0],marker='o',label='peak SNR SR')
    plt.plot(psnr_res_list[:,1],marker='o',label='peak SNR SK')
    plt.legend()
    plt.xlabel("Test Index",fontsize=16)
    plt.ylabel("PSNR",fontsize=16)
    plt.title("PSNR comparison for Super-Resolved (SR) and Nearest-Neighbor Scaled(SK)",fontsize=20)
    plt.figure()
    plt.plot(ssim_res_list[:,0],marker='o',label='SSIM SR')
    plt.plot(ssim_res_list[:,1],marker='o',label='SSIM SK')
    plt.legend()
    plt.xlabel("Test Index",fontsize=16)
    plt.ylabel("SSIM",fontsize=16)
    plt.title("SSIM comparison for Super-Resolved (SR) and Nearest-Neighbor Scaled(SK)",fontsize=20)
    plt.show()
    
#     plt.figure()
#     plt.imshow(origIm)
#     
#     plt.figure()
#     plt.imshow(srIm)
#     
#     plt.show()
    

