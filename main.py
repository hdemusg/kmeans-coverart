import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.image as img
import sys
import skimage
from skimage import io

# Base of this algorithm is from https://www.geeksforgeeks.org/image-compression-using-k-means-clustering/

def distance(x1, y1, x2, y2):
      
    dist = np.square(x1 - x2) + np.square(y1 - y2)
    dist = np.sqrt(dist)
  
    return dist

def main():
    if len(sys.argv) < 4:
        exit()
    img = io.imread(sys.argv[1])
    img = img / 255
    clusters = int(sys.argv[2])
    threshold = float(sys.argv[3])
    name = sys.argv[1].split('.')[0]

    pts = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
    m, n = pts.shape
    means = np.zeros((clusters, n))

    for i in range(clusters):
        rand = [int(np.random.random(1))*10, int(np.random.random(1))*8]
        means[i, 0] = pts[rand[0], 0]
        means[i, 1] = pts[rand[1], 1]

    iterations = 0
    delta = float('inf')
    m, n = pts.shape
      
    # these are the index values that
    # correspond to the cluster to
    # which each pixel belongs to.
    index = np.zeros(m) 
  
    # k-means algorithm.
    while(delta > threshold):
  
        for j in range(len(pts)):
              
            # initialize minimum value to a large value
            minv = 1000
            temp = None

            for k in range(clusters):
                  
                x1 = pts[j, 0]
                y1 = pts[j, 1]
                x2 = means[k, 0]
                y2 = means[k, 1]
                  
                if(distance(x1, y1, x2, y2) < minv):         
                    minv = distance(x1, y1, x2, y2)
                    temp = k
                    index[j] = k 

        d = 0.0
          
        for k in range(clusters):
              
            sumx = 0
            sumy = 0
            count = 0
              
            for j in range(len(pts)):
                  
                if(index[j] == k):
                    sumx += pts[j, 0]
                    sumy += pts[j, 1] 
                    count += 1
              
            if(count == 0):
                count = 1    

            oldx = means[k, 0]  
            oldy = means[k, 1]
            
            means[k, 0] = float(sumx / count)
            means[k, 1] = float(sumy / count) 

            d += distance(oldx, oldy, means[k, 0], means[k, 1])
        delta = d / k
        iterations += 1 
        print(f"Delta after iteration {iterations}: {delta}")

    centroid = np.array(means)
    recovered = centroid[index.astype(int), :]
      
    # getting back the 3d matrix (row, col, rgb(3))
    recovered = np.reshape(recovered, (img.shape[0], img.shape[1],
                                                     img.shape[2]))
  
    # plotting the compressed image.
    plt.imshow(recovered)
    plt.show()
  
    # saving the compressed image.
    skimage.io.imsave(name + '-' + str(clusters) +
                        '_colors.jpg', recovered)

if __name__ == "__main__":
    main()
