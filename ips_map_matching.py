'''
***************************************************************************************************

* Created on      :    Oct 18, 2019
* Filename        :    Position_Estimator
* Description     :    Estimat position of Cars with use of Particle Filters, Beacons and IMU
* Author          :    Ali Bemani
* Module version  :    1.0
* Compiler Version:    PyDev 3.7
* Email           :    ali.bemani@hig.se

***************************************************************************************************
'''

'''*************************************** Import Modules **************************************'''
#from bluepy.btle import Scanner, DefaultDelegate

import numpy as np
import scipy as scipy
from numpy.random import uniform
import scipy.stats

np.set_printoptions(threshold=3)
np.set_printoptions(suppress=True)
import cv2
import random


'''***************************************** Definitions ***************************************'''
def drawLines(img, points, r, g, b):
    cv2.polylines(img, [np.int32(points)], isClosed=False, color=(r, g, b))

def drawCross(img, center, r, g, b):
    d = 5
    t = 2
    LINE_AA = cv2.LINE_AA
    color = (r, g, b)
    ctrx = center[0,0]
    ctry = center[0,1]
    cv2.line(img, (ctrx - d, ctry - d), (ctrx + d, ctry + d), color, t, LINE_AA)
    cv2.line(img, (ctrx + d, ctry - d), (ctrx - d, ctry + d), color, t, LINE_AA)
    

def mouseCallback(event, x, y, flags,null):
    global center
    global trajectory
    global previous_x
    global previous_y
    global pp_x
    global pp_y
    
    
    global zs
    
    center=np.array([[x,y]])
    trajectory=np.vstack((trajectory,np.array([previous_x,previous_y])))
    #noise=sensorSigma * np.random.randn(1,2) + sensorMu
    
    #if pp_x >0:
    #print('enter pp_x>0')
    heading=np.arctan2(np.array([previous_y-pp_y]), np.array([pp_x-previous_x ]))

    if heading>0:
        heading=-(heading-np.pi)
    else:
        heading=-(np.pi+heading)
        
    distance=np.linalg.norm(np.array([[pp_x,pp_y]])-np.array([[previous_x,previous_y]]) ,axis=1)
    
    #print("heading",heading)
    #print("distance",distance)
    
    
    std=np.array([2,4])
    u=np.array([heading,distance])
    predict(particles, u, std, landmarks=landmarks, dt=1.)
    zs = (np.linalg.norm(landmarks - center, axis=1) + (np.random.randn(NL) * sensor_std_err))
    #print(zs)
    update(particles, weights, z=zs, R=50, landmarks=landmarks)
    
    indexes = systematic_resample(weights)
    resample_from_index(particles, weights, indexes)

    pp_x = previous_x
    pp_y = previous_y



    #print(np.array([np.average(particles[:,0]), np.average(particles[:,1])]))
    robot_pos = estimate(particles, weights)
    z = int(np.around(robot_pos[0,0]))
    s = int(np.around(robot_pos[0,1]))

    
#    previous_x=x
#    previous_y=y   
    previous_x=z
    previous_y=s
         
    print("x_cal:",z)
    print("x_real:",x)
    print("y_cal:",s)
    print("y_real:",y)


WIDTH=300
HEIGHT=200
WINDOW_NAME="Particle Filter"

#sensorMu=0
#sensorSigma=3

sensor_std_err=5


def create_uniform_particles(road_map, N):
    particles = np.empty((N, 2))
    for i in range(N):
        particles[i] = random.choice(road_map)  
        
    #particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    #particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    return particles



def predict(particles, u, std, landmarks, dt=1.):
    N = len(particles)
    dist = (u[1] * dt) + (np.random.randn(N) * std[1])
    particles[:, 0] += np.cos(u[0]) * dist
    particles[:, 1] += np.sin(u[0]) * dist

    for i, landmark in enumerate(landmarks):       
        dis_map_matching=np.power((particles[:,0] - landmark[0])**2 +(particles[:,1] - landmark[1])**2,0.5)
        for index in range(N):
            if dis_map_matching[index]>200:
                    particles[index] = random.choice(road_map) 

   
def update(particles, weights, z, R, landmarks):
    weights.fill(1.)
    for i, landmark in enumerate(landmarks):
       
        distance=np.power((particles[:,0] - landmark[0])**2 +(particles[:,1] - landmark[1])**2,0.5)
        weights *= scipy.stats.norm(distance, R).pdf(z[i])
        
        #print(distance)
        
    weights += 1.e-300 # avoid round-off to zero
    weights /= sum(weights)
    
def neff(weights):
    return 1. / np.sum(np.square(weights))

def systematic_resample(weights):
    N = len(weights)
    positions = (np.arange(N) + np.random.random()) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N and j<N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes
    
def estimate(particles, weights):
    #pos = particles[:, 0:1]
    #mean = np.average(pos, weights=weights, axis=0)
    #var = np.average((pos - mean)**2, weights=weights, axis=0)
    #return mean, var
    position =  np.empty((1,2))
    position[0,0] = np.average(particles[:,0], weights=weights, axis=0)
    position[0,1] = np.average(particles[:,1], weights=weights, axis=0)
    return position 

def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights /= np.sum(weights)

'''*************************************** Global Variables ************************************'''

'''*************************************** Static Variables ************************************'''

'''******************************************** Classes ****************************************'''

'''********************************************** Main *****************************************'''
road_map = np.array([[0,100],[0,96],[0,104],[0,91],[0,109],[1,87],[1,113],[1,83]\
                    ,[1,117],[2,79],[2,121],[3,74],[3,126],[4,70],[4,130],[5,66]\
                    ,[5,134],[6,62],[7,138],[8,58],[8,142],[9,54],[10,146],[11,50]\
                    ,[12,150],[13,46],[13,154],[15,42],[16,157],[17,39],[18,161],[20,35]\
                    ,[20,165],[22,32],[22,168],[25,28],[25,172],[27,25],[28,175],[30,22]\
                    ,[31,178],[33,18],[34,181],[36,16],[37,184],[40,13],[40,187],[43,10]\
                    ,[44,190],[47,8],[47,192],[51,6],[51,194],[54,4],[55,196],[58,3]\
                    ,[59,197],[63,1],[63,199],[67,1],[67,199],[71,0],[72,200],[75,0]\
                    ,[76,200],[80,0],[80,200],[84,1],[84,199],[88,2],[89,198],[92,3]\
                    ,[93,197],[96,4],[97,196],[100,6],[101,194],[104,7],[105,193],[108,9]\
                    ,[109,191],[112,11],[113,190],[116,12],[117,188],[120,14],[121,186],[124,15]\
                    ,[125,185],[129,16],[129,184],[133,18],[133,182],[137,19],[137,181],[141,19]\
                    ,[142,181],[145,20],[146,180],[150,20],[150,180],[154,20],[154,180],[158,19]\
                    ,[159,181],[162,19],[163,182],[167,18],[167,183],[171,16],[171,184],[175,15]\
                    ,[175,185],[179,13],[179,187],[183,12],[183,188],[187,10],[187,190],[191,9]\
                    ,[191,191],[195,7],[195,193],[199,6],[199,194],[203,4],[203,196],[207,3]\
                    ,[208,197],[211,2],[212,198],[215,1],[216,199],[220,0],[220,200],[224,0]\
                    ,[224,200],[228,0],[229,200],[233,1],[233,199],[237,1],[237,199],[241,3]\
                    ,[241,197],[245,4],[245,196],[249,6],[249,194],[253,8],[253,192],[256,10]\
                    ,[257,189],[260,13],[260,187],[263,16],[263,184],[266,19],[266,181],[269,22]\
                    ,[269,178],[272,25],[272,175],[275,28],[275,172],[277,32],[278,168],[280,35]\
                    ,[280,165],[282,39],[282,161],[284,42],[285,157],[286,46],[287,154],[288,50]\
                    ,[289,150],[290,54],[290,146],[292,58],[292,142],[293,62],[293,138],[295,66]\
                    ,[295,134],[296,70],[296,130],[297,74],[297,125],[298,78],[298,121],[299,83]\
                    ,[299,117],[299,87],[299,113],[300,91],[300,108],[300,96],[300,104],[300,100]])
    
x_range=np.array([0,300])
y_range=np.array([0,200])

#Number of partciles
N=100

landmarks=np.array([ [0,0], [300,0], [0,200], [300,200], [150,100]  ])
NL = len(landmarks)

#particles=create_uniform_particles(x_range, y_range, N) # uniform distribution of all particles
particles=create_uniform_particles(road_map, N) # uniform distribution of all particles

weights = np.array([1.0]*N)


# Create a black image, a window and bind the function to window
img = np.zeros((HEIGHT,WIDTH,3), np.uint8)
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME,mouseCallback)

center=np.array([[2,1]])

trajectory=np.zeros(shape=(0,2))
robot_pos=np.zeros(shape=(0,2))

previous_x=1
previous_y=1

pp_x = 0
pp_y = 0
DELAY_MSEC=50

z = 1
s = 1

print(cv2.__version__)
print("Test IS Ok")



'''****************************************** Infnit Loop **************************************'''

while(1):

    cv2.imshow(WINDOW_NAME,img)
    img = np.zeros((HEIGHT,WIDTH,3), np.uint8)
    drawLines(img, trajectory,   0,   255, 0)
    drawCross(img, center, r=255, g=0, b=0)
    drawLines(img, road_map,   25,   25, 45)    
    #landmarks
    for landmark in landmarks:
        cv2.circle(img,tuple(landmark),10,(255,0,0),-1)
    
    #draw_particles:
    for particle in particles:
        cv2.circle(img,tuple((int(particle[0]),int(particle[1]))),1,(255,255,255),-1)

    if cv2.waitKey(DELAY_MSEC) & 0xFF == 27:
        break

#    cv2.circle(img,(10,10),10,(255,0,0),-1)
#    cv2.circle(img,(10,30),3,(255,255,255),-1)
#    cv2.putText(img,"iBeacons_Landmark",(30,20),1,1.0,(255,0,0))
#    cv2.putText(img,"Particles_Filter",(30,40),1,1.0,(255,255,255))
#    cv2.putText(img,"Car_Trajectory(Ground truth)",(30,60),1,1.0,(0,255,0))

#    drawLines(img, np.array([[10,55],[25,55]]), 0, 255, 0)
    


cv2.destroyAllWindows()
#print(trajectory)
print("Terminated Programe")