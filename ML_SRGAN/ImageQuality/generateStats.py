'''
Created on 19-May-2020

@author: Neeraj Badal
'''
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
    srDir = "D:/Mtech/FY/SEM2/ML/SuperResolutionTest/WashingtonDC_System-Ready_Stereo_50cm/056082264020/056082264020_01_P001_PAN/test/"
    
#     imVersion = "upscale_4/" #upscaled_fresh_dataset/ #upscale_4/ avsrgan_up/ 
#     imVersion = "upscaled_fresh_dataset/"
#     imVersion = "avsrgan_up/"
#     imVersion = "avsrgan_v2/"
#     imVersion = "mse_adv_v3/"
#     imVersion = "vgg_mse/"
#     imVersion = "vgg_mse_tv/"
#     imVersion = "ae_vgglike_mse/"
#     imVersion = "ae_mse_tv/"
#     imVersion = "aelikevgg_adv/"
#     imVersion = "vae/"
#     imVersion = "vae_v5/"
    imVersion = "vae_v6/"
    eg_index = [1108]
    srDir = srDir + imVersion
    
    stat_results_dir = "D:/Mtech/FY/SEM2/ML/SuperResolutionTest/WashingtonDC_System-Ready_Stereo_50cm/056082264020/056082264020_01_P001_PAN/stat/" 
    
    opFiles = [
        stat_results_dir+imVersion.split('/')[0]+"_psnr.dat",
        stat_results_dir+imVersion.split('/')[0]+"_ssim.dat",
        stat_results_dir+imVersion.split('/')[0]+"_spat_freq.dat",
        stat_results_dir+imVersion.split('/')[0]+"_ps_val.dat",
        stat_results_dir+"orig_ps_val.dat"
        ]
    
    psnr_res_list = []
    ssim_res_list = []
    spat_freq = []
    orig_ps_val = []
    sr_ps_val = []
    
    for i_ in range(1000,1485):#1485
        imageName = "eg_"+str(i_)+".png"
        srImageName = "up4_eg_"+str(i_)+".png"
        print("...Test set ..",i_)
#         imageName = "eg_"+str(i_)+".png"
#         srImageName = "eg_"+str(i_)+"_upscaled.png"
#         srImageName = "func_"+str(i_)+"_upscaled.png"
        
        
        origIm = plt.imread(origDir+imageName, format='png')
        srIm = plt.imread(srDir+srImageName, format='png')
        
        origIm = np.array(origIm)
        srIm = np.array(srIm)
        
        out_psnr = measure.compare_psnr(origIm[:,:,0], srIm[:,:,0])
        ssim_val = measure.compare_ssim(origIm[:,:,0], srIm[:,:,0])
        
        psnr_res_list.append([out_psnr])
        ssim_res_list.append([ssim_val])
        
        if i_ in eg_index:
            orig_ps = np.array(fft2Dimensional(origIm[:,:,0]))
            sr_ps = np.array(fft2Dimensional(srIm[:,:,0]))
         
            spat_freq.append(orig_ps[0])
            orig_ps_val.append(orig_ps[1])
            sr_ps_val.append(sr_ps[1])
        
        
         
         
#         plt.plot(orig_ps[0],orig_ps[1],label='Orig Image')
#         plt.plot(orig_ps[0],sr_ps[1],label='SR Image')
#         plt.plot(orig_ps[0],sk_ps[1],label='SK Image')
#         plt.xlabel("spatial frequency cycles per pixel",fontsize=16)
#         plt.ylabel("normalized power",fontsize=16)
#         plt.title("Power Spectrum Comparison between Super-resolved (SR) and Nearest-Neighbor Recaled (SK) Test set "+str(i_),fontsize=20)
#         plt.legend()
#         plt.show()
    
    
    psnr_res_list = np.array(psnr_res_list)
    ssim_res_list = np.array(ssim_res_list)
    spat_freq = np.array(spat_freq)
    orig_ps_val = np.array(orig_ps_val)
    sr_ps_val = np.array(sr_ps_val)
    
    np.savetxt(opFiles[0],psnr_res_list,delimiter='\t')
    np.savetxt(opFiles[1],ssim_res_list,delimiter='\t')
    np.savetxt(opFiles[2],spat_freq,delimiter='\t')
    np.savetxt(opFiles[3],sr_ps_val,delimiter='\t')
    np.savetxt(opFiles[4],orig_ps_val,delimiter='\t')
    
    
#     plt.figure()
#     plt.plot(psnr_res_list[:,0],marker='o',label='peak SNR SR')
#     plt.plot(psnr_res_list[:,1],marker='o',label='peak SNR SK')
#     plt.legend()
#     plt.xlabel("Test Index",fontsize=16)
#     plt.ylabel("PSNR",fontsize=16)
#     plt.title("PSNR comparison for Super-Resolved (SR) and Nearest-Neighbor Scaled(SK)",fontsize=20)
#     plt.figure()
#     plt.plot(ssim_res_list[:,0],marker='o',label='SSIM SR')
#     plt.plot(ssim_res_list[:,1],marker='o',label='SSIM SK')
#     plt.legend()
#     plt.xlabel("Test Index",fontsize=16)
#     plt.ylabel("SSIM",fontsize=16)
#     plt.title("SSIM comparison for Super-Resolved (SR) and Nearest-Neighbor Scaled(SK)",fontsize=20)
#     plt.show()
    
#     plt.figure()
#     plt.imshow(origIm)
#     
#     plt.figure()
#     plt.imshow(srIm)
#     
#     plt.show()
    

