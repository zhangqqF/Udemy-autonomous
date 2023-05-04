import os
import glob



for i in glob.glob(r'F:\Applied Control Systems 1 autonomous cars Math + PID + MPC\字幕\3. Vehicle modelling for lateral control using equations of motion/**'):
    if '-en_US_CN' in i:
        print(i)
        os.rename(i, i.replace('l-en_US_CN', ''))