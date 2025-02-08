import numpy as np

with open('targets.npy', 'rb') as f:
    try:
        while 1:
            item = np.load(f)
            print(item)
    except:
        print("EoF")

with open('states.npy', 'rb') as f:
    try:
        while 1:
            item = np.load(f)
            print(item)
    except:
        print("EoF")
