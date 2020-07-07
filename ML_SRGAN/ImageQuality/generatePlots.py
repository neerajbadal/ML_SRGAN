'''
Created on 19-May-2020

@author: Neeraj Badal
'''
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
from numpy import genfromtxt

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
    
    
#     imVersion = "avsrgan_up/" #upscaled_fresh_dataset/ #upscaled_nn_4/ avsrgan_up/ 
    
#     imVersions = ["avsrgan_up/","upscaled_fresh_dataset/",
#                   "upscale_4/","avsrgan_v2/","mse_adv_v3/"]
    
#     imVersions = ["avsrgan_up/","upscaled_fresh_dataset/",
#                   "upscale_4/","avsrgan_v2/","mse_adv_v3/"]
    
#     imVersions = ["mse_adv_v3/","vgg_mse/","vgg_mse_tv/",
#                   "ae_vgglike_mse/","ae_mse_tv/","aelikevgg_adv/"]
    
    imVersions = ["upscale_4/","vgg_mse_tv/","vgg_mse/","mse_adv_v3/","vae/","vae_v2",
                  "ae_mse_tv/","ae_vgglike_mse/","aelikevgg_adv/","vae_v4/","vae_v5/",
                  "vae_v6/"
                  ]
    
    
    labVersions = ["Nearest Neighbor","L7","L5","L1","L8","L9","L6","L4","L2","L11"
                   ,"L12",
                   "L11_TV"
                   ]
    
    stat_results_dir = "D:/Mtech/FY/SEM2/ML/SuperResolutionTest/WashingtonDC_System-Ready_Stereo_50cm/056082264020/056082264020_01_P001_PAN/stat/" 
    
    
    plt.figure()
    plt.xlabel("Test Index",fontsize=20)
    plt.ylabel("PSNR",fontsize=20)
    plt.yticks(fontsize=16)
    plt.title("PSNR Comparison ",fontsize=20)
    
    avg_psnr = []
    count_ = 0
    for imVersion in imVersions:
    
        opFiles = stat_results_dir+imVersion.split('/')[0]+"_psnr.dat"
        psnr_res_list = genfromtxt(opFiles,delimiter='\t')
        plt.plot(psnr_res_list[:],marker='+',label=labVersions[count_])
        avg_psnr.append(np.average(psnr_res_list[:]))
        count_ += 1
    plt.legend(prop={'size': 14})    
    
    plt.figure()
    plt.xlabel("Test Index",fontsize=20)
    plt.ylabel("SSIM",fontsize=20)
    plt.yticks(fontsize=16)
    plt.title("SSIM Comparison",fontsize=20)
    avg_ssim = []
    count_ = 0
    for imVersion in imVersions:
    
        opFiles = stat_results_dir+imVersion.split('/')[0]+"_ssim.dat"
        ssim_res_list = genfromtxt(opFiles,delimiter='\t')
        plt.plot(ssim_res_list[:],marker='*',label=labVersions[count_])
        avg_ssim.append(np.average(ssim_res_list[:]))
        count_ += 1
    plt.legend(prop={'size': 14})
    
    
#     plt.plot(orig_ps[0],orig_ps[1],label='Orig Image')
#     plt.plot(orig_ps[0],sr_ps[1],label='SR Image')
#     plt.plot(orig_ps[0],sk_ps[1],label='SK Image')
    
    eg_index = 108
    
#     plt.figure()
#     plt.xlabel("spatial frequency cycles per pixel",fontsize=16)
#     plt.ylabel("normalized power",fontsize=16)
#     plt.title("Power Spectrum Comparison ",fontsize=20)
#     for i_ in range(0,len(imVersions)):
#     
#         
#         opFiles = stat_results_dir+imVersions[i_].split('/')[0]+"_spat_freq.dat"
#         spat_freq = genfromtxt(opFiles,delimiter='\t')
#         
#         opFiles = stat_results_dir+imVersions[i_].split('/')[0]+"_ps_val.dat"
#         ps_val = genfromtxt(opFiles,delimiter='\t')
#         
#         opFiles = stat_results_dir+"orig_ps_val.dat"
#         orig_ps_val = genfromtxt(opFiles,delimiter='\t')
#         
#         print(spat_freq.shape,"  ",orig_ps_val.shape,"  ",ps_val.shape)
#         
#         if i_ == 0:
#             plt.plot(spat_freq[:],orig_ps_val[:],label='Orig Image')
#         
#         plt.plot(spat_freq[:],ps_val[:],label=imVersions[i_].split('/')[0])
#         
#         
#     plt.legend()
    
    
    
    
    
    
#     stat_results_dir+imVersion.split('/')[0]+"_ssim.dat",
#     stat_results_dir+imVersion.split('/')[0]+"_spat_freq.dat",
#     stat_results_dir+imVersion.split('/')[0]+"_ps_val.dat",
#     stat_results_dir+"orig_ps_val.dat"
    
    
    
        
#     ssim_res_list = genfromtxt(opFiles[1],delimiter='\t')
#     spat_freq = genfromtxt(opFiles[2],delimiter='\t')
#     orig_ps_val = genfromtxt(opFiles[3],delimiter='\t')
#     sr_ps_val = genfromtxt(opFiles[4],delimiter='\t')
    
#     imVersions = ["mse_adv_v3","vgg_mse","vgg_mse_tv",
#                   "ae_vgglike_mse","ae_mse_tv","aelikevgg_adv"]

#     imVersions = ["Nearest Neighbor","L7","L5","L1","L6","L4","L2"]
    imVersions = ["Nearest Neighbor","L7","L5","L1","L8","L9","L6","L4","L2","L11"
                  ,"L12","L11_TV"
                  ]
    plt.figure()
    plt.xlabel("Operation",fontsize=20)
    plt.ylabel("Avg. SSIM",fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=16)
    plt.title("Average SSIM Comparison",fontsize=20)
    plt.plot(imVersions,avg_ssim,marker='o')
    
    plt.figure()
    plt.xlabel("Operation",fontsize=20)
    plt.ylabel("Avg. PSNR",fontsize=20)
    plt.title("Average PSNR Comparison",fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=16)
    plt.plot(imVersions,avg_psnr,marker='o')
    
    
    plt.show()
    
#     plt.figure()
#     plt.imshow(origIm)
#     
#     plt.figure()
#     plt.imshow(srIm)
#     
#     plt.show()
    

