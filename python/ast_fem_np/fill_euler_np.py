# This file is part of the AST_MFEM project (https://github.com/ShMonem/AST_MFEM).
# Copyright AST_MFEM to all developers and contributors. All rights reserved.
# License: Apache-2.0

import numpy as np

def fill_euler(num_handles):
    eulers = np.zeros ((num_handles, 3))
    
    eulers [0, :] = [0.496 * np.pi / 180, -0.240 * np.pi / 180 , 34.824 * np.pi / 180]
    eulers [1, :] = [-0.326 * np.pi / 180, 0.444 * np.pi / 180, 6.113 * np.pi / 180] 
    eulers [2, :] = [-0.000 * np.pi / 180, 0.000 * np.pi / 180, -8.028 * np.pi / 180] 
    eulers [3, :] = [-0.000 * np.pi / 180, 0.000 * np.pi / 180, -0.646 * np.pi / 180] 
    eulers [4, :] = [-0.000 * np.pi / 180, 0.000 * np.pi / 180, -16.763 * np.pi / 180] 
    eulers [5, :] = [-0.000 * np.pi / 180, 0.000 * np.pi / 180, -16.763 * np.pi / 180] 
    eulers [7, :] = [-10.522 * np.pi / 180, 3.782 * np.pi / 180,  -0.064* np.pi / 180] 
    eulers [8, :] = [11.857 * np.pi / 180, 19.232 * np.pi / 180,  -0.043* np.pi / 180] 
    eulers [9, :] = [7.249 * np.pi / 180, -84.351 * np.pi / 180,  -12.732* np.pi / 180] 
    eulers [10, :] = [39.435 * np.pi / 180, -15.434 * np.pi / 180,  -42.359* np.pi / 180] 
    eulers [11, :] = [-2.922 * np.pi / 180, -1.575 * np.pi / 180,  14.948* np.pi / 180] 
    eulers [12, :] = [3.150 * np.pi / 180, 71.789 * np.pi / 180,  14.948* np.pi / 180] 
    eulers [13, :] = [0.939 * np.pi / 180, 57.664 * np.pi / 180,  -7.207* np.pi / 180] 
    eulers [15, :] = [-59.886 * np.pi / 180, 13.435 * np.pi / 180,  -16.710* np.pi / 180] 
    eulers [16, :] = [-53.086 * np.pi / 180, 3.137 * np.pi / 180,  -7.679* np.pi / 180] 
    eulers [17, :] = [-26.277 * np.pi / 180, 0.946 * np.pi / 180,  -4.038* np.pi / 180] 
    eulers [19, :] = [-52.249 * np.pi / 180, 1.336 * np.pi / 180,  -3.353* np.pi / 180] 
    eulers [20, :] = [-62.738 * np.pi / 180, 4.453 * np.pi / 180,  -8.225* np.pi / 180] 
    eulers [21, :] = [-75.802 * np.pi / 180, 6.044 * np.pi / 180,  -7.158* np.pi / 180] 
    eulers [23, :] = [-47.143 * np.pi / 180, -4.318 * np.pi / 180,  3.026* np.pi / 180] 
    eulers [24, :] = [-68.595 * np.pi / 180, 5.456 * np.pi / 180,  -7.322* np.pi / 180] 
    eulers [25, :] = [-57.079 * np.pi / 180, 3.924 * np.pi / 180,  -7.328* np.pi / 180] 
    eulers [27, :] = [-56.440 * np.pi / 180, -14.385 * np.pi / 180,  11.675* np.pi / 180] 
    eulers [28, :] = [-63.762 * np.pi / 180, 4.931 * np.pi / 180,  -8.774* np.pi / 180] 
    eulers [29, :] = [-43.621 * np.pi / 180, 2.528 * np.pi / 180,  -6.324* np.pi / 180] 
    eulers [31, :] = [10.511 * np.pi / 180, -3.817 * np.pi / 180,  0.151* np.pi / 180] 
    eulers [32, :] = [-11.969 * np.pi / 180, -18.817 * np.pi / 180,  0.074* np.pi / 180] 
    eulers [33, :] = [-7.026 * np.pi / 180, 84.204 * np.pi / 180,  12.507* np.pi / 180] 
    eulers [34, :] = [-37.413 * np.pi / 180, -13.117 * np.pi / 180,  -41.196* np.pi / 180] 
    eulers [35, :] = [-2.161 * np.pi / 180, -4.455 * np.pi / 180,  -14.972* np.pi / 180] 
    eulers [36, :] = [-3.729 * np.pi / 180, -73.384 * np.pi / 180,  -0.664* np.pi / 180] 
    eulers [37, :] = [-4.098 * np.pi / 180, -63.254 * np.pi / 180,  3.120* np.pi / 180] 
    eulers [39, :] = [57.500 * np.pi / 180, -12.010 * np.pi / 180,  12.536* np.pi / 180] 
    eulers [40, :] = [55.546 * np.pi / 180, -2.572 * np.pi / 180,  5.517* np.pi / 180] 
    eulers [41, :] = [29.469 * np.pi / 180, -0.802 * np.pi / 180,  2.780* np.pi / 180] 
    eulers [43, :] = [54.676 * np.pi / 180, -0.553 * np.pi / 180,  1.582* np.pi / 180] 
    eulers [44, :] = [60.994 * np.pi / 180, -3.105 * np.pi / 180,  6.275* np.pi / 180] 
    eulers [45, :] = [76.197 * np.pi / 180, -4.444 * np.pi / 180,  5.076* np.pi / 180] 
    eulers [47, :] = [52.804 * np.pi / 180, -5.117 * np.pi / 180,  7.070* np.pi / 180] 
    eulers [48, :] = [59.703 * np.pi / 180, -3.248 * np.pi / 180,  6.184* np.pi / 180] 
    eulers [49, :] = [62.782 * np.pi / 180, -3.305 * np.pi / 180,  4.802* np.pi / 180] 
    eulers [51, :] = [56.742 * np.pi / 180, -15.420 * np.pi / 180,  14.076* np.pi / 180] 
    eulers [52, :] = [61.818 * np.pi / 180, -3.537 * np.pi / 180,  6.836* np.pi / 180] 
    eulers [53, :] = [47.705 * np.pi / 180, -2.199 * np.pi / 180,  4.500* np.pi / 180] 
    eulers [55, :] = [2.638 * np.pi / 180, 5.466 * np.pi / 180, -50.313 * np.pi / 180] 
    eulers [56, :] = [-4.662 * np.pi / 180, -10.203 * np.pi / 180, 99.772 * np.pi / 180] 
    eulers [57, :] = [7.360 * np.pi / 180, -1.938 * np.pi / 180, -85.511 * np.pi / 180] 
    eulers [58, :] = [-0.054 * np.pi / 180, -0.271 * np.pi / 180, -2.847 * np.pi / 180] 
    eulers [60, :] = [-12.450 * np.pi / 180, 5.272 * np.pi / 180, -126.553 * np.pi / 180,] 
    eulers [61, :] = [0.769 * np.pi / 180, 9.669 * np.pi / 180, 77.890 * np.pi / 180] 
    eulers [62, :] = [5.019 * np.pi / 180, -6.404 * np.pi / 180, 14.526 * np.pi / 180] 
    return eulers


