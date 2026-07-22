import numpy as np

for dimension in [2,10,50,100,500]:

    data = np.random.rand(100,dimension)

    distances = np.linalg.norm(data[0]-data,axis=1)

    print("Dimension =",dimension)

    print("Average Distance =",np.mean(distances))
