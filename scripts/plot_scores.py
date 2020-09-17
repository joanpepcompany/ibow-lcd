# import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from numpy import trapz
from sklearn import preprocessing
import scipy.signal
import math

from os import listdir
from os.path import isfile, join

# Compute Area under the line of a given vector
def integrate(y_vals, h):
    i = 1
    total = y_vals[0] + y_vals[-1]
    for y in y_vals[1:-1]:
        if i % 2 == 0:
            total += 2 * y
        else:
            total += 4 * y
        i += 1
    return total * (h / 3.0)

  
# Load scores from file
# pose_filename = '/home/joanpep/catkin_ws/cpp_code/lc_storm/results/scores/200.yaml'
path_filename = '/home/joanpep/catkin_ws/cpp_code/lc_storm/results/scores_Lip6In/'
onlyfiles = [f for f in listdir(path_filename) if isfile(join(path_filename, f))]
# Sort the filenames
onlyfiles = [item.replace(".yaml", "") for item in onlyfiles]
onlyfiles.sort(key = int) 
onlyfiles = [item + ".yaml" for item in onlyfiles]

h_lc_pts = []
h_lc_lines = []

h_non_lc_pts = []
h_non_lc_lines = []

for m in onlyfiles:

    pose_filename = path_filename + m

    print(pose_filename)
    if (not os.path.exists(pose_filename)):
        print('Filename doesn\'t exist')
        sys.exit(0)

    fs = cv2.FileStorage(pose_filename, cv2.FILE_STORAGE_READ)
    pts_list = fs.getNode("pts_score").mat()
    lines_list = fs.getNode("lines_score").mat()
    is_lc = fs.getNode("is_LC").mat()
    print ("is_lc: ", is_lc)

    min_size = 10
    evaluate_pts = False
    evaluate_lines = False
    if(not pts_list is None):
        print ("Pts length:", len(pts_list))
        if (len(pts_list) > min_size): 
            evaluate_pts = True
    
    # if(not is_lc):
    #     continue
    if( not lines_list is None):
        print ("Lines length:", len(lines_list))
        if (len(lines_list) > min_size):
            evaluate_lines = True

    if ((not evaluate_pts) and (not evaluate_lines)):
        if(is_lc):
            input("Press Enter to continue...")
        continue
    # print("pts_list: ", pts_list)
    # print("lines_list: ", lines_list)

    # Normalize between 0 and 1 both lists and delete 0.0 value numbers from the lists
    if (evaluate_pts):
        if(pts_list.min(axis=0) != pts_list.max(axis=0) ):
            pts_list = (pts_list - pts_list.min(axis=0)) / (pts_list.max(axis=0) - pts_list.min(axis=0))
    
        pts_list = pts_list[~np.all(pts_list == 0, axis=1)]

    if (evaluate_lines):
        lines_list = lines_list[~np.all(lines_list == 0, axis=1)]
    
        if(lines_list.min(axis=0) != lines_list.max(axis=0) ):
            lines_list = (lines_list - lines_list.min(axis=0)) / (lines_list.max(axis=0) - lines_list.min(axis=0))

    pts_list = pts_list[:40]
    lines_list = lines_list[:40]

    # Pts and Lines format conversion
    # s_pts_list = []
    # for m in pts_list:
    #     s_pts_list.insert(len(s_pts_list),m[0])

    # s_lines_list = []
    # for m in lines_list:
    #     s_lines_list.insert(len(s_lines_list),m[0])

    # Smooth pts curve
    # TODO: 
    # pts_length = int(len(s_pts_list)/2)
    # if (pts_length % 2) == 0:  
    #     pts_length += 1

    # # window size 51, polynomial order 3
    # pts_hat = scipy.signal.savgol_filter(s_pts_list, pts_length , 3) 

    # # Smooth lines curve
    # lines_length = int(len(s_lines_list)/2)
    # print ("pts_lenght: ", pts_length, "  lines_lenght: ", lines_length)
    # if (lines_length % 2) == 0:  
    #     lines_length += 1
    # # window size,polynomial order 3
    # lines_hat = scipy.signal.savgol_filter(s_lines_list, lines_length , 3) 

   

    # Compute area under the curve
    dx = 1.0

    area_pts = [0.0]
    if (evaluate_pts):
        area_pts = 1/(integrate(pts_list, dx))
        print("1/Area pts=", area_pts)
    
    area_lines = [0.0]
    if (evaluate_lines):
        area_lines = 1/(integrate(lines_list, dx))
        print("1/Area lines=", area_lines)

    X = np.asarray([[area_pts[0],area_lines[0]]]
                    , dtype=np.float)
                    
    # l2-normalize the samples (rows). 
    X_normalized = preprocessing.normalize(X, norm='l1')
    print("L1 [Pts, Lines]: ", X_normalized)

    sum_inv_areas = area_pts + area_lines
    X_normalized[0][0] = area_pts/sum_inv_areas
    X_normalized[0][1] = area_lines/sum_inv_areas
    print("Tolchi [Pts, Lines]: ", X_normalized)

    A = 0.0
    if (X_normalized[0][0] > X_normalized[0][1]):
        if (X_normalized[0][0] == 1.0):
            A = 1.0
        else:
            A = X_normalized[0][0] /X_normalized[0][1]
    else:
        if (X_normalized[0][1] == 1.0):
            A = 1.0
        else:
            A = X_normalized[0][1] /X_normalized[0][0]
    print("A:", A)

    if (math.isfinite(A)):
        if (is_lc):
            if (X_normalized[0][0] > X_normalized[0][1]):
                h_lc_pts.insert(len(h_lc_pts),A)
            else:
                h_lc_lines.insert(len(h_lc_lines),A)

        else:
            if (X_normalized[0][0] > X_normalized[0][1]):
                h_non_lc_pts.insert(len(h_lc_pts),A)
            else:
                h_non_lc_lines.insert(len(h_lc_lines), A)
   



    # print("x area: ", X)
    
    # Plot Data
    fig, ax = plt.subplots()
    # POINTS Data for plotting
    # t = np.arange(0.0, len(pts_hat), 1.0)
    # ax[0].plot(t, pts_hat,label='points filtered')

    # t = np.arange(0.0, len(lines_hat), 1.0)
    # ax[0].plot(t, lines_hat,label='lines filtered')
    # ax[0].set(xlabel='Samples', ylabel='Normalized Score',
    #     title='Smoothed')
    # ax[0].grid()
    if (evaluate_pts):
        t_pts = np.arange(0.0, len(pts_list), 1.0)
        ax.plot(t_pts, pts_list,label='points')
    

    # ax[1].text(21, 0.82, 'Pts score     = ' +str(X_normalized[0][0])[:5], fontsize=12)
    # ax[1].text(21, 0.76, 'Lines score = ' +str(X_normalized[0][1])[:5], fontsize=12)

    # LINES Data for plotting
    if (evaluate_lines):
        v = np.arange(0.0, len(lines_list), 1.0)
        ax.plot(v, lines_list,label='Lines')

    ax.set(xlabel='Samples', ylabel='Normalized Score',
        title='Points Score Evaluation')
    ax.grid()
    plt.legend()
    plt.show()


 

# plt.hist(h_lc_pts, bins=50)
# plt.hist(h_non_lc_pts, bins=50)

#     # fig.savefig("Score_Evaluation.png")
# plt.show()
fig, ax = plt.subplots(2)

bins = 30
ax[0].hist(h_lc_pts, bins, alpha=0.5, label='LC Pts')
ax[0].hist(h_non_lc_pts, bins, alpha=0.5, label='Non LC Pts')
ax[1].hist(h_lc_lines, bins, alpha=0.5, label='LC Lines')
ax[1].hist(h_non_lc_lines, bins, alpha=0.5, label='Non LC Lines')
ax[0].legend(loc='upper right')
ax[1].legend(loc='upper right')
ax[0].set_title('L1 normalization for the inverted areas')
plt.show()
