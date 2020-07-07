'''
Created on 19-May-2020

@author: Neeraj Badal
'''
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import TruncatedSVD
from multiprocessing import Pool
from functools import partial
def performPCAAnalysis(data_):
    
    data_std = data_
    
    
    data_cov_ = np.cov(data_std.T)
    e_vals, e_vecs = np.linalg.eig(data_cov_)
    
    
    print(e_vecs.shape)
    total_var = np.sum(e_vals)
    var_val_wise = [(i / total_var) for i in sorted(e_vals, reverse=True)]
    cum_var = np.cumsum(var_val_wise)

    plt.plot(cum_var,marker='o',label='cum')
    plt.plot(var_val_wise,marker='o',label='indiv')
#     plt.plot(e_vals,marker='o',label='cum')
#     plt.plot(e_vals_,marker='o',label='cum2')
    plt.legend()
    plt.show()

def computeVariance(index_):
    srDir = "D:/Mtech/FY/SEM2/ML/SuperResolutionTest/WashingtonDC_System-Ready_Stereo_50cm/056082264020/056082264020_01_P001_PAN//train/"
    srImageName = "eg_"+str(index_)+".png"
    print("image inlcuded ",index_)
    srIm = plt.imread(srDir+srImageName, format='png')
    srIm = np.array(srIm)
    srIm = srIm[:,:,0]
    n_comps = [16,64,128,350]
    variance_list = []
    for k in n_comps:
        svd = TruncatedSVD(n_components=k, n_iter=7)
        svd.fit(srIm)
        variance_list.append(svd.explained_variance_ratio_.sum())
        print(" i ",index_," k ",k,"  ",svd.explained_variance_ratio_.sum())
    variance_list = np.array(variance_list)
    return variance_list

if __name__ == "__main__":    
    srDir = "D:/Mtech/FY/SEM2/ML/SuperResolutionTest/WashingtonDC_System-Ready_Stereo_50cm/056082264020/056082264020_01_P001_PAN//train/"
    stat_results_dir = "D:/Mtech/FY/SEM2/ML/SuperResolutionTest/WashingtonDC_System-Ready_Stereo_50cm/056082264020/056082264020_01_P001_PAN/stat/"
    imageSet = []
    svd_res = []
    
    pool = Pool(5)
    L = []
    L = pool.map(computeVariance,range(0,1000))
    pool.close()
    pool.join()
    
    n_comps = [16,64,128,350]
    
#     for i_ in range(0,2):#[1204,1247]:#range(1000,1485):
# #         imageName = "000"+str(i_)+".PNG"
# #         srImageName = str(i_)+"_upscaled.PNG"
#         
# #         print("...Test set ..",i_)
# #         imageName = "eg_"+str(i_)+".png"
#         srImageName = "eg_"+str(i_)+".png"
#         
# #         skImageName = "func_"+str(i_)+"_upscaled.png"
# #         
# #         avsrImageName = "up4_eg_"+str(i_)+".png"
#         
#         print("image inlcuded ",i_)
#         srIm = plt.imread(srDir+srImageName, format='png')
#         srIm = np.array(srIm)
#         srIm = srIm[:,:,0]
#         n_comps = [16,32,64,128,256,512,650,750]
#         variance_list = []
#         for k in n_comps:
#             svd = TruncatedSVD(n_components=k, n_iter=7)
#             svd.fit(srIm)
#             variance_list.append(svd.explained_variance_ratio_.sum())
#             print(" i ",i_," k ",k,"  ",svd.explained_variance_ratio_.sum())
#         
#         plt.plot(n_comps,variance_list,marker='o')
#         variance_list = np.array(variance_list)
#         svd_res.append(variance_list)
# #         plt.plot(n_comps,variance_list,marker='o')
# #         srIm = srIm[:,:,0].flatten()
# #         imageSet.append(srIm)
# #     imageSet = np.array(imageSet)
# #     performPCAAnalysis(imageSet)
# #     n_comps = [16,32,64,128,256,512,650,750,850,950]
# #     variance_list = []
# #     for k in n_comps:
# #         svd = TruncatedSVD(n_components=k, n_iter=7)
# #         svd.fit(imageSet)
# #         variance_list.append(svd.explained_variance_ratio_.sum())
# #         print(" i ",i_," k ",k,"  ",svd.explained_variance_ratio_.sum())
# 
# 
# #     n_comps = [1024,2048,3300,4068,6000,8000,12000,18000,25000]
#     
# #     svd_res = np.array(svd_res)
#     
#     
# #     plt.plot(n_comps,variance_list,marker='o')
    svd_res = np.array(L)
    np.savetxt(stat_results_dir+"/svd_dim.dat",svd_res,delimiter='\t')
    
    for i_ in range(0,1000):
        plt.plot(n_comps,svd_res[i_])
    
    plt.show()
    
    
        
#     print(imageSet.shape)
    exit(0)

# '''
# Created on 19-May-2020
# 
# @author: Neeraj Badal
# '''
# import numpy as np
# from matplotlib import pyplot as plt
# from sklearn.decomposition import TruncatedSVD
# def performPCAAnalysis(data_):
#     
#     data_std = data_
#     
#     
#     data_cov_ = np.cov(data_std.T)
#     e_vals, e_vecs = np.linalg.eig(data_cov_)
#     
#     
#     print(e_vecs.shape)
#     total_var = np.sum(e_vals)
#     var_val_wise = [(i / total_var) for i in sorted(e_vals, reverse=True)]
#     cum_var = np.cumsum(var_val_wise)
# 
#     plt.plot(cum_var,marker='o',label='cum')
#     plt.plot(var_val_wise,marker='o',label='indiv')
# #     plt.plot(e_vals,marker='o',label='cum')
# #     plt.plot(e_vals_,marker='o',label='cum2')
#     plt.legend()
#     plt.show()
#     
# if __name__ == "__main__":    
#     srDir = "D:/Mtech/FY/SEM2/ML/SuperResolutionTest/WashingtonDC_System-Ready_Stereo_50cm/056082264020/056082264020_01_P001_PAN//train/"
#     
#     imageSet = []
#     svd_res = []
#     
#     for i_ in range(0,10):#[1204,1247]:#range(1000,1485):
# #         imageName = "000"+str(i_)+".PNG"
# #         srImageName = str(i_)+"_upscaled.PNG"
#         
# #         print("...Test set ..",i_)
# #         imageName = "eg_"+str(i_)+".png"
#         srImageName = "eg_"+str(i_)+".png"
#         
# #         skImageName = "func_"+str(i_)+"_upscaled.png"
# #         
# #         avsrImageName = "up4_eg_"+str(i_)+".png"
#         
#         print("image inlcuded ",i_)
#         srIm = plt.imread(srDir+srImageName, format='png')
#         srIm = np.array(srIm)
#         srIm = srIm[:,:,0]
#         n_comps = [16,32,64,128,256,512,650,750]
#         variance_list = []
#         for k in n_comps:
#             svd = TruncatedSVD(n_components=k, n_iter=7)
#             svd.fit(srIm)
#             variance_list.append(svd.explained_variance_ratio_.sum())
#             print(" i ",i_," k ",k,"  ",svd.explained_variance_ratio_.sum())
#         
# #         variance_list = np.array(variance_list)
# #         svd_res.append(variance_list)
#         plt.plot(n_comps,variance_list,marker='o')
# #         srIm = srIm[:,:,0].flatten()
# #         imageSet.append(srIm)
# #     imageSet = np.array(imageSet)
# #     performPCAAnalysis(imageSet)
# 
# #     n_comps = [1024,2048,3300,4068,6000,8000,12000,18000,25000]
#     
# #     svd_res = np.array(svd_res)
#     
#     
# #     plt.plot(n_comps,variance_list,marker='o')
#     plt.show()
#     
#     
#         
# #     print(imageSet.shape)
#     exit(0)
        

        