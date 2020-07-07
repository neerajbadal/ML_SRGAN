'''
Created on 20-May-2020

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
#     eg_index = 1204
    eg_index = 1179
#     srDir = srDir + imVersion
    
    stat_results_dir = "D:/Mtech/FY/SEM2/ML/SuperResolutionTest/WashingtonDC_System-Ready_Stereo_50cm/056082264020/056082264020_01_P001_PAN/stat/" 
#     origDir = "D:/Mtech/FY/SEM2/ML/SuperResolutionTest/original images/" 
#     destinationDir = "D:/Mtech/FY/SEM2/ML/SuperResolutionTest/" 
#     srDir = "D:/Mtech/FY/SEM2/ML/SuperResolutionTest/" #sr vgg
    
    
#     opFiles = [
#         stat_results_dir+imVersion.split('/')[0]+"_psnr.dat",
#         stat_results_dir+imVersion.split('/')[0]+"_ssim.dat",
#         stat_results_dir+imVersion.split('/')[0]+"_spat_freq.dat",
#         stat_results_dir+imVersion.split('/')[0]+"_ps_val.dat",
#         stat_results_dir+"orig_ps_val.dat"
#         ]
    
#     imVersions = ["orig","avsrgan_up/","upscaled_fresh_dataset/",
#                   "upscale_4/","avsrgan_v2/","mse_adv_v3/"]
#     imVersions = ["orig","avsrgan_up/","upscaled_fresh_dataset/",
#                   "upscale_4/","mse_adv_v3/"]
#     imVersions = ["orig","upscaled_fresh_dataset/","mse_adv_v3/"]
#     filenames = [0,2,1]
#     imVersions = ["orig","mse_adv_v3/","vgg_mse/","vgg_mse_tv/","upscale_4/"]
#                   "ae_vgglike_mse/","ae_mse_tv/","aelikevgg_adv/"]
#     imVersions = ["orig","vgg_mse_tv/","upscale_4/"]
#     imVersions = ["orig","upscaled/","upscaled_nn_4/"]
#     filenames = [0,1,3]
    
#     imVersions = ["orig","mse_adv_v3/","vgg_mse/","vgg_mse_tv/","vae/","vae_tv_increased/",
#                   "vae_v2/","upscale_4/"]
    
    imVersions = ["orig","mse_adv_v3/","vgg_mse/","vgg_mse_tv/",
                  "vae/","vae_v2/","vae_v3/","vae_v4/","vae_v5/","vae_v6/",
                  "ae_vgglike_mse/","ae_mse_tv/","upscale_4/"
                  ]
    
#     filenames = [4,5,3]
    
#     filenames = [0,1,2,3,1]
    filenames = [0,1,1,1,1,1,1,1,1,1,1,1,3]
#     filenames = [0,1,1,1,1,1,1,3]
    slope_ef = []
    slope_fg = []
    slope_eg = []
    new_metric = []
    
    orig_ef = 0.0
    orig_fg = 0.0
    orig_eg = 0.0
    
#     fig, axs = plt.subplots(2, 3,sharex=True, sharey=True)
    
    for i_ in range(0,len(imVersions)):
        if filenames[i_] == 0:
            imageName = "eg_"+str(eg_index)+".png"
        elif filenames[i_] == 1:
            imageName = "up4_eg_"+str(eg_index)+".png"
        elif filenames[i_] == 2:
            imageName = "eg_"+str(eg_index)+"_upscaled.png"
        elif filenames[i_] == 3:
            imageName = "func_"+str(eg_index)+"_upscaled.png"
        elif filenames[i_] == 4:
            imageName = "000"+str(eg_index)+".png"
        elif filenames[i_] == 5:
            imageName = str(eg_index)+"_upscaled.png"
        
#         imageName = "eg_"+str(i_)+".png"
#         srImageName = "eg_"+str(i_)+"_upscaled.png"
#         srImageName = "func_"+str(i_)+"_upscaled.png"
        
        if i_ == 0:
            origIm = plt.imread(origDir+imageName, format='png')
        else:
            origIm = plt.imread(srDir+imVersions[i_]+imageName, format='png')
#         origIm = plt.imread(srDir+imVersions[i_]+imageName, format='png')
        origIm = np.array(origIm)
#         row_ = int(i_/3)
#         col_ = int(i_%3)
#         axs[row_, col_].imshow(origIm[:,:,0],cmap="gray")
#         axs[row_, col_].set_title(imVersions[i_].split('/')[0])
        
        
        orig_ps = np.array(fft2Dimensional(origIm[:,:,0]))
        plt.plot(orig_ps[0],orig_ps[1],label=imVersions[i_].split('/')[0])
#         plt.figure()
#         plt.title(imVersions[i_].split('/')[0])
#         plt.imshow(origIm[:,:,0],cmap="gray")
        
        e = orig_ps[1,1]
        f = orig_ps[1,39] #39 0.05 77 0.1
        g = orig_ps[1,154] #192 25 154 0.2
         
#         s1 = orig_ps[1,192]
#         s2 = orig_ps[1,231]
         
        total_area = np.sum(orig_ps[1,:])
        local_area = np.sum(orig_ps[1,192:231])# / total_area 
         
#         local_area = np.fabs(orig_ps[1,192]-orig_ps[1,231])/(0.3-0.25)
         
        slope_ef_ = np.fabs(f - e)/(0.05-0.0)
        slope_fg_ = np.fabs(g - f)/(0.2 - 0.05) #0.25 - 0.05
#         slope_eg_ = np.fabs(g - e)/(0.15 - 0.0)
         
#         local_area = np.sum(orig_ps[1,39:116])
#         temp_metric = (-0.6)*slope_ef_ + 0.8*slope_fg_ - 0.7*local_area
        temp_metric = (-0.6)*slope_ef_ + 0.8*slope_fg_
        new_metric.append(temp_metric)
         
         
#         if i_ !=0:
#             slope_ef.append(slope_ef_-orig_ef)
#             slope_fg.append(slope_fg_-orig_fg)
#             slope_eg.append(slope_eg_-orig_eg)
#         else:
#             orig_ef = slope_ef_
#             orig_fg = slope_fg_
#             orig_eg = slope_eg_
             
             
        slope_ef.append(slope_ef_)
        slope_fg.append(slope_fg_)
        slope_eg.append(local_area)
          
    plt.xlabel("spatial frequency cycles per pixel",fontsize=18)
    plt.ylabel("normalized power",fontsize=18)
    plt.title("Power Spectrum Comparison",fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=16)
    plt.legend(prop={'size': 14})
     
    plt.figure()
    plt.title("EF slope")
    plt.plot(slope_ef,marker='o')
    
#     imVersions = ["Orig","mse_adv_v3","vgg_mse","vgg_mse_tv"] 
#     imVersions = ["orig","mse_adv_v3/","vgg_mse/","vgg_mse_tv/","upscale_4/"]
#     imVersions = ["orig","L1","L5","L7","vae","vae_tv_ic",
#                   "vae_v2","Nearest Neighbor"]
    imVersions = ["orig","L1","L5","L7","L8","L9","L10","L11","L12","L11_tv",
                  "L4","L6",                 
                  "Nearest Neighbor"]
#     imVersions = ["orig","L1","L5","L7","vae","vae_tv_ic",
#                   "Nearest Neighbor"]
    
#     imVersions = ["orig","upscaled/","upscaled_nn_4/"]
#     imVersions = ["Orig","SRGAN L7","Nearest Neighbor"]
    
    plt.figure()
    plt.xlabel("Different Loss Functions",fontsize=18)
    plt.ylabel("LogPower Slope F-G",fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=16)
    plt.title("Blur Comparison for different loss function and NN over test set 1179",fontsize=20)
    plt.plot(imVersions,slope_fg,marker='o')
    
    outDataFile = "D:/ps_slope_ef.dat"
    count_ = 0
    with open(outDataFile, "w") as myfile:
        for data_ in slope_fg:
            myfile.write(imVersions[count_]+
                         "\t"+str(data_)+"\n")
             
            count_ = count_ + 1
    
    
#     new_metric = [slope_ef[j_]*slope_fg[j_] for j_ in range(0,len(slope_ef))]
     
    plt.figure()
    plt.title("New Metric")
    plt.plot(new_metric,marker='o')
     
    plt.figure()
    plt.title("Slope EG")
    plt.plot(slope_eg,marker='o')
     
    plt.show()
