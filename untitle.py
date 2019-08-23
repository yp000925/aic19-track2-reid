import re
import numpy as np

# pattern = re.compile(r'\d+')
#
# file = np.loadtxt('track2.txt',delimiter=' ',dtype=str)
# (row,col) = file.shape
# output=[]
# for i in range(row):
#     r =[]
#     for j in range(col):
#         r.append(re.findall(pattern, file[i][j]))
#
#     if output == []:
#         output = r
#     else:
#         output = np.concatenate([output,r],axis = 0)
#

# file = open('track2.txt')
# matches=[]
# for line in file.readlines():
#     match =re.findall(pattern,line)
#     match = [int(m) for m in match]
#     matches=np.append(matches,match)
# file.close()
# matches= matches.reshape(1052,100)
# np.savetxt('mat.txt', matches, delimiter=' ', fmt='%d')

file = np.loadtxt('match28000_Mslm.txt',dtype=int)
for file_name in range(1,1053):
    match = file[file_name-1][0:50]
    match = match.reshape(50,1)
    np.savetxt('./tool/result28000/%06d.txt' % file_name, match, fmt='%d')
