import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from pyquaternion import Quaternion
from segment import *

plt.close("all")

file_to_read = '../data/trial1.txt'
freq = 50
sample_period = 1/freq


def get_header_data(filename, header_part_name):
    all_headers = np.genfromtxt(filename, delimiter=',',names=True, deletechars='').dtype.names
    all_data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    header_data = []
    for index, header in enumerate(all_headers):
        if header_part_name in header:
            header_data.append(all_data[:,index])
    return np.array(header_data).transpose()
    
def reduce_still(pose, force):
    start = pose[0]
    stop = pose[-1]
    threshold = 5
    time_before = 1.323
    time_after = 0.8789
    first_index = 0
    last_index = -1
    for index, p in enumerate(pose):
        if np.linalg.norm(p-start) > threshold:
            first_index = index
            break
    for index, p in enumerate(pose[::-1]):
        if np.linalg.norm(p-stop) > threshold:
            last_index = -index
            break
    first_index -= 0
    last_index += 16
    return pose[first_index:last_index], force[first_index:last_index]



force = get_header_data(file_to_read,'ati2rob_2.forcesTorques')
time = np.cumsum(sample_period*np.ones(force.shape[0]))
nbr_time_steps = time.size


T44 = get_header_data(file_to_read,'T44_rightbase')

pos = np.zeros((nbr_time_steps, 3))
quats = np.zeros((nbr_time_steps, 3))
for time_step in range(nbr_time_steps):
    trans = T44[time_step].reshape(4,4)
    pos[time_step] = trans[0:3,-1]
    quats[time_step] = Quaternion(matrix=trans).imaginary
pos = pos-pos[0]
force = force-force[0]
force = force/1000

print(force)

pose = np.concatenate((pos,quats), axis=1)

pose, force = reduce_still(pose, force)
time = np.cumsum(sample_period*np.ones(force.shape[0]))


gmm_segment(time, pose, 5)
force_segment(time, force)



plt.show()


