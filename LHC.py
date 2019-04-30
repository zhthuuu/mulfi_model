from samplingplan import samplingplan
import numpy as np

num = 35

sp = samplingplan(4)
X = sp.optimallhc(num,iterations = 20,generation=False)
X = np.array(X)
X[:,0] = 4.*X[:,0]+5.
X[:,1] = 25.*X[:,1]+25.
X[:,2] = 18.*X[:,2]+2.
X[:,3] = 6.*X[:,3]+3.
print(X)

file_abs = 'C:\Users\Hao\Desktop\LHC.txt'
with open(file_abs,'w') as f:
    for i in range(num):
        for j in range(4):
            f.write('{}    '.format(float('%.2f' % X[i,j])))
        f.write('\n')

print('succeed to write file')