#/usr/bin/python
import GPUtil
import time
import numpy as np
import matplotlib.pyplot as plt

print(' ID GPU MEM')
print('---------------------')
iCount = 0
gpurate = np.zeros(1)
gmem = np.zeros(1)

plt.figure(figsize=(8, 6), dpi=80)
plt.ion()

while True:
    # GPUtil.showUtilization()
    GPUs = GPUtil.getGPUs()
    for gpu in GPUs:
        print('{0:2d} {1:3.0f}% {2:3.0f}%'.format(gpu.id, gpu.load*100, gpu.memoryUtil*100))
        gpurate = np.append(gpurate, gpu.load)
        gmem = np.append(gmem, gpu.memoryUtil)
        plt.cla()
        plt.title('GPU Utilization')
        plt.grid(True)
        plt.ylim(0, 1)
        i = len(gpurate)
        if (i>120):
            plt_gpu = gpurate[i-120:i]
            plt_gmem = gmem[i-120:i]
            iCnt = 120
            plt.plot(range(iCnt), plt_gpu, 'r-', label='GPU rate')
            plt.plot(range(iCnt), plt_gmem, 'b-', label='GMEM rate')
        else:
            plt.plot(range(i), gpurate, 'r-', label='GPU rate')
            plt.plot(range(i), gmem, 'b-', label='GMEM rate')
        plt.legend(loc='upper right', shadow=True)

    if (iCount % 20 == 0):
        print(' ID GPU MEM')
        print('---------------------')
    iCount = iCount + 1
    # time.sleep(1)
    plt.pause(0.5)
