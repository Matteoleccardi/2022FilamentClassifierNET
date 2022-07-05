'''
    Utilities used to build the dataset.

    ----------------------
    DATASET FILES CONTENTS
    ----------------------

    Every file is composed of the following columns:
    | x | y | z | class | id |
    --------------------------
    |    .... row 1 ...      |
    |    .... row N ...      |
    --------------------------
    A point cloud is defined in an unstructured way (the row index has no meaning)
    
    x, y, z
    -------
        Cartesian coordinates of each point in the point cloud.
        The reference frame, and the point cloud position with respect to 0, has no importance.
    class
    -----
        Classes of filament points:
            0: Isolated point
            1: Endpoint of filament
            2: Central point of sigle filament (classic filament)
            3: Central point of filament inside conjunction of 3 filaments (3-streets conjunction)
            4: Central point of filament inside conjunction of 4 (or more) filaments (4-strees conjunction, or more)
            Additional classes (TBD):
            5: Point on surface (flat or curved)
            6: Point on the edge of a surface (straght or curved by any shape or curvature)
            7: Point on edge between 2 surfaces
    id
    --
        An id which identifies all points belonging to the same filament.
        It is needed to distinguish filaments in case of a file containing many filaments.
        Points belonging to no filament or structure will have index -1, all the others will have 0, 1, 2, ..., sequentially.

    -------------
    DATASET TYPES
    -------------

    SYNTHETIC:
        Synthetic dataset organization:
            /synthetic_dataset/data0000000.csv
            /synthetic_dataset/data0000001.csv
            ....
        Considered filaments shapes:
        - Straight
        - Splines
        - Random knots
        - Tri/quadri/quinti-centered stars
            - Staright, curved (x3)
        - Ribbons (no start, no end, just interseptions)
        - 1,2,3,4, 5 straights and then a circle/ellipse
            - circle -> ellipse
        - Leaf-like filaments
        - Fibonacci tree-like filaments
        - Spirals
        - Only one - N isolated points
        Each of them:
        - Can be found alone in the 3D space, or together with other filaments (clusters of disjointed filaments)
        - Randomly oriented/rotated in 3D space
        - Noisy, with different noise levels
        - Different densities

    CAT08 REFERENCE CENTERLINES:
        CAT08 dataset organization:
            /CAT08_dataset/data0000.csv
            /CAT08_dataset/dat00001.csv
            ....
        Considered filaments shapes:
        - Every reference vessel centerline (x8)
        Each of them:
        - No noise (10%)
        - Small noise (15%)
        - Medium noise (50%)
        - High noise (25%)
        Each of them:
        - Rotated (x8, randomly)

    SHAPENET DATASET (?)
'''

import numpy as np
from scipy import interpolate as scipy_interpolate
import matplotlib.pyplot as plt
import os
from queue import Queue
from threading import Thread
import time


# Utilities 
# ---------

def matrix_rotate3D(alpha, beta, gamma) -> np.ndarray:
    '''
        Rotation matrices:
            alpha: yaw angle   (around Z axis)
            beta : pitch angle (around Y axis)
            gamma: roll angle  (around X axis)
        Angles have to be expressed in radians : [0, 2*pi]
    '''
    yaw =   [[np.cos(alpha), -np.sin(alpha),       0     ],
             [np.sin(alpha),  np.cos(alpha),       0     ],
             [     0,             0,               1     ]]
    pitch = [[ np.cos(beta),      0,         np.sin(beta)],
             [     0,             1,               0     ],
             [-np.sin(alpha),     0,         np.cos(beta)]]
    roll =  [[     1,             0,            0   ],
             [     0,        np.cos(gamma), -np.sin(gamma)],
             [     0,        np.sin(gamma),  np.cos(gamma)]]
    rot_mat = np.matmul(yaw, pitch)
    rot_mat = np.matmul(rot_mat, roll)
    return rot_mat


def makefilament_noise3D(d_max, n_points, noise_type:str="random"):
    ''' Function to generate noise
        Input:
            d_max : The maximum amount of noise. The actual maximum is drawn from an uniform distribution noise_max: [d_max/10, d_max]
            n_points : number of points to which the noise must be added (number of points in the filament)
            noise_type:
                "random" (default) -> chooses randomly between the following types of noise distributions
                "uniform" -> generates uniform noise in a bounding box of -noise_max, +noise_max
                "normal" -> generates normal (gaussian) noise in a bounding box of -noise_max, +noise_max
                "triangular" -> generates triangular noise in a bounding box of -noise_max, +noise_max
        Output: noise, noise_max
            noise: numpy.ndarray points
            noise_max: the maximum amount of noise for this filament (see above noise_max: [d_max/10, d_max])
    '''
    if noise_type=="random":
        noise_choice = np.random.randint(0,2+1)
    elif noise_type=="uniform":
        noise_choice = 0
    elif noise_type=="normal":
        noise_choice = 1
    elif noise_type=="triangular":
        noise_choice = 2
    else:
        print("ERROR: incorrect option <"+noise_type+"> passed in makefilament_noise3D")
        quit()
    noise_max = np.random.uniform((d_max*0.99)/10, (d_max*0.99))
    if noise_choice == 0:
        noise = np.random.uniform(-noise_max, noise_max, size=(n_points,3))
    if noise_choice == 1:
        noise = np.random.normal(0, noise_max/3, size=(n_points,3))
        noise[np.abs(noise)>noise_max] = noise_max
    if noise_choice == 2:
        noise = np.random.triangular(-noise_max, 0, noise_max, size=(n_points,3))
    noise[0,:], noise[-1,:] = [0,0,0], [0,0,0]
    return noise, noise_max


def makefilaments_isolated_points(n_isolated: int, threshold_distance, filament_points: np.ndarray) -> np.ndarray:
    ''' Creates isolated points
        Input:
            n_isolated: number of total isolated points
            threshold_distance: Minimum distance between isolated points and filament points
            filament: a (N,3) filament
    '''
    f = np.empty((0,5))
    ptp_max =  np.ptp(filament_points)
    x_mean, x_stdev = np.mean(filament_points[:,0]), np.mean([ptp_max,np.ptp(filament_points[:,0])])/4
    y_mean, y_stdev = np.mean(filament_points[:,1]), np.mean([ptp_max,np.ptp(filament_points[:,1])])/4
    z_mean, z_stdev = np.mean(filament_points[:,2]), np.mean([ptp_max,np.ptp(filament_points[:,2])])/4
    while f.shape[0] < n_isolated:
        x = np.random.normal(x_mean, x_stdev)
        y = np.random.normal(y_mean, y_stdev)
        z = np.random.normal(z_mean, z_stdev)
        point = np.array([x,y,z])
        d_i = np.linalg.norm( filament_points[:,0:3]-point , axis=1)
        if np.min(d_i) > threshold_distance:
            if len(f) > 0:
                d_i = np.linalg.norm(f[:,0:3]-point, axis=1)
                if np.min(d_i) > threshold_distance:
                    f = np.append(f, np.append(point, [0, -1], axis=0).reshape((1,5)), axis=0)
            else:
                f = np.append(f, np.append(point, [0, -1], axis=0).reshape((1,5)), axis=0)
    return f


def regularize_points_distances(points: np.ndarray, method="regularise") -> np.ndarray:
    ''' This function takes as input a N x 3 numpy.ndarray and regularises
        the distances between consecutive points, making them converge to the mean.
        Inputs:
            points: the points to regularise/augment
            methods:
                "regularise" : it attempts to regularise the points by using a combination of mean and minimum distance
                "regularise strong" : regularises so that every point is distant from the next maximum the current minimum/2 distance between two points
                "augment" : it makes sure that each point is distant from the next one so that, in the final array, there are around 10'000 points
    '''
    if method == "regularise":
        d_vector = [np.linalg.norm(points[i,:]-points[i+1,:]) for i in range(points.shape[0]-1)]
        d_thresh = (3*np.mean(d_vector) + np.min(d_vector))/4
    if method == "regularise strong":
        d_vector = [np.linalg.norm(points[i,:]-points[i+1,:]) for i in range(points.shape[0]-1)]
        d_thresh = np.min(d_vector)/2
    if method == "augment":
        d_vector = [np.linalg.norm(points[i,:]-points[i+1,:]) for i in range(points.shape[0]-1)]
        d_tot = np.sum(d_vector)
        d_thresh = d_tot/7000
    # Forward pass
    i = 0
    condition = True
    while condition:
        d = np.linalg.norm(points[i,:]-points[i+1,:])
        if d > d_thresh:
            points = np.insert(points, i+1, (points[i,:]+points[i+1,:])/2, axis=0)
            i = np.max([i-1, 0])
        else:
            i += 1
        if i == points.shape[0]-1: condition = False
    # Backward pass
    i =  points.shape[0] - 1
    condition = True
    while condition:
        d = np.linalg.norm(points[i,:]-points[i-1,:])
        if d > d_thresh:
            points = np.insert(points, i, (points[i,:]+points[i-1,:])/2, axis=0)
            i = np.min([i+1, points.shape[0] - 1])
        else:
            i -= 1
        if i == 0: condition = False
    return points

def make_fibonacci_sphere(radious: float=1., n_points: int=300) -> np.ndarray:
    ''' Creates a Fibonacci sphere - points evenly distributed on a sphere.
        Inputs:
            radious: the radious of the sphere
            n_points: the number of points in the Fibonacci sphere
        Outputs:
            A numpy.ndarray of shape (n_points,3)
    '''
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    for i in range(n_points):
        y = 1 - (i / float(n_points - 1)) * 2
        r = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * r
        z = np.sin(theta) * r
        points.append([x, y, z])
    points = np.array(points)*radious
    return points



# Demos 
# -----

def makefilament_classes_demo():
    np.random.seed(444)
    # Filament 1
    t = np.linspace(0,25,num=1000)
    x = t
    y = 0.1*t**2 - 5*t**0.5
    z = 0*t
    p_class = 2*np.ones(t.shape)
    p_class[0], p_class[-1] = 1, 1
    p_class[539:547] = 3
    p_class[694:701] = 3
    p_class[802:807] = 4
    noise = np.random.normal(0, 0.05/3, (1000,3))
    filament = np.array([x+noise[:,0], y+noise[:,1], z+noise[:,2], p_class, np.zeros(p_class.shape)]).T
    # Filament 2
    t = np.linspace(0,11,num=50)
    x = t + 13.55
    y = -0.1*t**2
    z = 0*t
    p_class = 2*np.ones(t.shape)
    p_class[0:4] = 3
    p_class[-1] = 1
    noise = np.random.normal(0, 0.07/3, (50,3))
    f = np.array([x+noise[:,0], y+noise[:,1], z+noise[:,2], p_class, np.zeros(p_class.shape)]).T
    filament = np.append(filament, f, axis=0)
    # Filament 3
    t = np.linspace(0,6,num=100)
    x = 20.1 - t
    y = 0.2*t**3 - 0.9*t**2 + 0.8*t + 18
    z = 0*t
    p_class = 2*np.ones(t.shape)
    p_class[0:4] = 4
    p_class[-1] = 1
    noise = np.random.normal(0, 0.03/3, (100,3))
    f = np.array([x+noise[:,0], y+noise[:,1], z+noise[:,2], p_class, np.zeros(p_class.shape)]).T
    filament = np.append(filament, f, axis=0)
    # Filament 4
    t = np.linspace(0,15,num=300)
    x = t + 20.05
    y = 18.9 + np.sin((0.9*2**(-(t-6)**2)+1.5)*t - 2)
    z = 0*t
    p_class = 2*np.ones(t.shape)
    p_class[0:4] = 4
    p_class[-1] = 1
    noise = np.random.normal(0, 0.03/3, (300,3))
    f = np.array([x+noise[:,0], y+noise[:,1], z+noise[:,2], p_class, np.zeros(p_class.shape)]).T
    filament = np.append(filament, f, axis=0)
    # Filament 5
    t = np.linspace(0,15,num=150)
    x = t + 17.4
    y = 0.1*(t-15)**2 - 13
    z = 0*t
    p_class = 2*np.ones(t.shape)
    p_class[0:3] = 3
    p_class[-1] = 1
    noise = np.random.normal(0, 0.05/3, (150,3))
    f = np.array([x+noise[:,0], y+noise[:,1], z+noise[:,2], p_class, np.zeros(p_class.shape)]).T
    filament = np.append(filament, f, axis=0)
    # Isolated points
    n_isolated = 20
    d_thresh = 2
    f = makefilaments_isolated_points(n_isolated, d_thresh, filament_points=filament[:,0:3])
    filament = np.append(filament, f, axis=0)
    # Rotation
    alpha = np.random.uniform(0, 2*np.pi)
    beta  = np.random.uniform(0, 2*np.pi)
    gamma = np.random.uniform(0, 2*np.pi)
    rotation_matrix = matrix_rotate3D(alpha, beta, gamma)
    for row in filament:
        row[0:3] = np.matmul(rotation_matrix, row[0:3])
    # Out
    return filament


# Synthetic data 
# --------------

def makefilament_aggregator(makefilament_function, n_points: int=30, length=10, n_filaments_max=8) -> np.ndarray:
    ''' 
        Function to crete multiple instances of makefilament_003 and makefilament_004 in just one space
        Input:
            makefilament_function: must be a function which take as input just n_points, length and outputs a Nx4 numpy.ndarray
            n_points : number of points in the filament [14, 500]
            length: total length of the segment
            n_filaments_max: maximum number of filaments
    '''
    n_filaments = np.random.randint(1,n_filaments_max+1) # Max 27 
    filaments_list = []
    d_thresh = 3*np.sqrt(11)*(length/n_points)
    c_max = 15
    translation_axies_hist = [[0,0,0]]
    x_lim_l, y_lim_l, z_lim_l =  1e30,  1e30,  1e30
    x_lim_h, y_lim_h, z_lim_h = -1e30, -1e30, -1e30
    while len(filaments_list) < n_filaments:
        is_ok, c = False, 0
        while (not is_ok) and c < c_max:
            filament = makefilament_function(n_points=n_points, length=length)
            [dx, dy, dz] = np.random.uniform(2*d_thresh,np.mean(np.abs(filament[:,0:3])), 3)*np.random.choice([1,-1])
            filament[:,0] = filament[:,0] + dx
            filament[:,1] = filament[:,1] + dx
            filament[:,2] = filament[:,2] + dx
            # Distances check
            d_min = 1e20
            for f in filaments_list:
                for pfil in filament:
                    d_min_temp = np.min( np.linalg.norm(pfil[0:3]-f[:,0:3], axis=1) )
                    if d_min_temp < d_min : d_min = d_min_temp
            if d_min > d_thresh: is_ok = True
            else: c+=1
        if (len(filaments_list) == 0) or not (c==c_max):
            x_lim_l = np.min([x_lim_l, np.min(filament[:,0])] )-2*d_thresh
            y_lim_l = np.min([y_lim_l, np.min(filament[:,1])] )-2*d_thresh
            z_lim_l = np.min([z_lim_l, np.min(filament[:,2])] )-2*d_thresh
            x_lim_h = np.max([x_lim_h, np.max(filament[:,0])] )+2*d_thresh
            y_lim_h = np.max([y_lim_h, np.max(filament[:,1])] )+2*d_thresh
            z_lim_h = np.max([z_lim_h, np.max(filament[:,2])] )+2*d_thresh
        if (c == c_max):
            translation_axies = np.random.choice([1, 0, -1], 3).tolist()
            while translation_axies in translation_axies_hist:
                translation_axies = np.random.choice([1, 0, -1], 3).tolist()
            translation_axies_hist.append(translation_axies)
            [fmin_x, fmin_y, fmin_z] = np.min(filament[:,0:3], axis=0)
            [fmax_x, fmax_y, fmax_z] = np.max(filament[:,0:3], axis=0)
            if (translation_axies[0] == 1) and (fmin_x < x_lim_h):
                filament[:,0] = filament[:,0] + np.abs(x_lim_h-fmin_x)
            elif (translation_axies[0] == -1) and (fmax_x > x_lim_l):
                filament[:,0] = filament[:,0] - np.abs(x_lim_l-fmax_x)
            if (translation_axies[1] == 1) and (fmin_y < y_lim_h):
                filament[:,1] = filament[:,1] + np.abs(y_lim_h-fmin_y)
            elif (translation_axies[1] == -1) and (fmax_y > y_lim_l):
                filament[:,1] = filament[:,1] - np.abs(y_lim_l-fmax_y)
            if (translation_axies[2] == 1) and (fmin_z < z_lim_h):
                filament[:,2] = filament[:,2] + np.abs(z_lim_h-fmin_z)
            elif (translation_axies[2] == -1) and (fmax_z > z_lim_l):
                filament[:,2] = filament[:,2] - np.abs(z_lim_l-fmax_z)
            # Distances check
            d_min = 1e20
            for f in filaments_list:
                for pfil in filament:
                    d_min_temp = np.min( np.linalg.norm(pfil[0:3]-f[:,0:3], axis=1) )
                    if d_min_temp < d_min : d_min = d_min_temp
            is_ok = True if d_min > d_thresh else False
        if is_ok:
            filaments_list.append(filament)
    # Build final set of points
    filament = np.empty((0,5))
    filamentID = 0
    for f in filaments_list:
        f[ f[:,4] != -1, 4] = filamentID
        filament = np.append(filament, f, axis=0)
        filamentID += 1
    # OUT
    return filament

#   Linear

def makefilament_linear(p0, angles, length, n_points: int, yaw_angle=0, pitch_angle=0, roll_angle=0) -> np.ndarray:
    '''
    Input:
        p0 = [x0, y0, z0] : starting point of the filament
        angles = [alpha, beta] : angles for polar coordinates:
            alpha : angle on the XY plane with respect to X axis. Domain: [0, 2*pi]
            beta  : angle running along the Z axis (outside XY plane). beta=0 means in XY plane. Domain : [-pi/2, pi/2]
        length : length of the filament
        n_points: number of points in the filament
        yaw_angle : rotation around Z axis
        pitch_angle : rotation around Y axis
        roll_angle : rotation around X axis
    Output: points, noise_max
            points: numpy.ndarray points of the noisy filament, (N,3)
            noise_max: the maximum amount of noise for this filament (see above noise_max: [d_max/10, d_max])
    '''
    # Geometry
    alpha, beta = angles[0], angles[1]
    t = np.linspace(0,length,n_points)
    x = np.cos(alpha)*np.sin(beta+np.pi/2)*t + p0[0]
    y = np.sin(alpha)*np.sin(beta+np.pi/2)*t + p0[1]
    z = np.cos(beta+np.pi/2)*t + p0[2]
    # Noise
    d = (length/n_points)
    noise, noise_max = makefilament_noise3D(d_max=d, n_points=n_points, noise_type="random")
    points = np.array([x, y, z]).T + noise
    # Rotation
    if (yaw_angle != 0) or (pitch_angle != 0) or (roll_angle != 0):
        rotation_matrix = matrix_rotate3D(yaw_angle, pitch_angle, roll_angle)
        for row in points:
            row[0:3] = np.matmul(rotation_matrix, row[0:3])
    return points, noise_max


def makefilament_001(n_points: int=30, length=10) -> np.ndarray:
    ''' 
        Linear filaments in 3D
        One filament
        Created on the XY plane and then rotated randomly through yaw, pitch, roll.
        Input:
            n_points : number of points in the filament [3, 500]
            length: total length of the segment 
    '''
    # Geometry
    points, _ = makefilament_linear(p0=[0,0,0], angles=[0,0], length=length, n_points=n_points)
    # Classification
    p_class     = 2*np.ones((points.shape[0],1))
    p_class[0], p_class[-1]  = 1, 1
    # Filament
    filament = np.append(points, p_class, axis=1)
    filament = np.append(filament, np.zeros(p_class.shape), axis=1)
    # ISOLATED POINTS
    n_isolated = int( np.random.uniform(2,10) )
    d_thresh = 1.5*np.sqrt(11)*(length/n_points)
    f = makefilaments_isolated_points(n_isolated, d_thresh, filament_points=filament[:,0:3])
    filament = np.append(filament, f, axis=0)
    # Rotation
    [alpha, beta, gamma] = np.random.uniform(0, 2*np.pi, 3)
    rotation_matrix = matrix_rotate3D(alpha, beta, gamma)
    for row in filament:
        row[0:3] = np.matmul(rotation_matrix, row[0:3])
    # Out
    return filament


def makefilament_002(n_points: int=30, length=10, n_filaments: int=1) -> np.ndarray:
    ''' 
        Linear filaments in 3D
        Multiple filament - not overlapping
        Created on the XY plane and then rotated randomly through yaw, pitch, roll.
        The neighbouring filaments can be not on the XY  plane, and can take up any orientation
        Input:
            n_points : number of points in the filament [3, 500]
            length: total length of the segment 
            n_filaments: number of additional filaments [1, 5]
    '''
    # FILAMENT 1
    # Geometry
    points, _ = makefilament_linear(p0=[0,0,0], angles=[0,0], length=length, n_points=n_points)
    # Classification
    p_class     = 2*np.ones((n_points,1))
    p_class[0], p_class[-1]  = 1, 1
    # Filament
    filament = np.append(points, p_class, axis=1)
    filament = np.append(filament, np.zeros(p_class.shape), axis=1)
    # OTHER FILAMENTS
    #d_thresh = 1.5*np.sqrt(11*((1.2*length)/int(0.8*n_points))**2)
    d_thresh = 1.5*np.sqrt(11)*(length/int(n_points))
    fil_count = 0
    while fil_count < n_filaments:
        # Geometry
        l   = np.random.triangular(0.2*length,length,1.8*length)
        n_p = int( np.max([4, np.random.triangular(0.5*n_points,n_points,1.5*n_points)]) )
        x = np.random.uniform(-0.5*length, 1.5*length)
        [y, z] = np.random.normal(0, length, 2)
        x_0   = np.array([x, y, z])
        phi   = np.random.uniform(0, 2*np.pi)
        theta = np.random.uniform(-np.pi/2,   np.pi/2)
        points, _ = makefilament_linear(p0=x_0, angles=[phi,theta], length=l, n_points=n_p)
        # Classification
        p_class     = 2*np.ones((n_p,1))
        p_class[0], p_class[-1]  = 1, 1
        # Candidate new filament
        f_new = np.append(points, p_class, axis=1)
        f_new = np.append(f_new, np.ones(p_class.shape)+fil_count, axis=1)
        # Update threshold distance iteratively
        new_d_thresh = 1.5*np.sqrt(11)*(l/int(n_p))
        d_thresh = new_d_thresh if d_thresh < new_d_thresh else d_thresh
        # Distance check from every point in the already saved filaments
        d_min = 1e20
        for p_new in f_new:
            dist_vector = np.linalg.norm(filament[:,0:3]-p_new[0:3])
            dist_vector_min = np.min(dist_vector)
            d_min = dist_vector_min if dist_vector_min < d_min else d_min
        if d_min > d_thresh:
            filament = np.append(filament, f_new, axis=0)
            fil_count += 1
    # ISOLATED POINTS
    n_isolated = int( np.random.uniform(2+n_filaments,20+n_filaments) )
    f = makefilaments_isolated_points(n_isolated, d_thresh, filament_points=filament[:,0:3])
    filament = np.append(filament, f, axis=0)
    # Rotation
    [alpha, beta, gamma] = np.random.uniform(0, 2*np.pi, 3)
    rotation_matrix = matrix_rotate3D(alpha, beta, gamma)
    for row in filament:
        row[0:3] = np.matmul(rotation_matrix, row[0:3])
    # Out
    return filament


def makefilament_003(n_points: int=30, length=10) -> np.ndarray:
    ''' 
        Linear filaments in 3D with 3-ways junctions
        One filament with one/two 3-way junctions. Points of 2 filaments are of a 3-way junction if they are closer than the maximum distance between 2 points, given the cubic noise used here in the dataset.
        Created on the XY plane and then rotated randomly through yaw, pitch, roll.
        Input:
            n_points : number of points in the filament [14, inf)
            length: total length of the segment 
    '''
    n_points = 14 if n_points < 14 else n_points
    # BASE FILAMENT
    # Geometry
    points, noise_prev = makefilament_linear(p0=[0,0,0], angles=[0,0], length=length, n_points=n_points)
    alpha_prev = np.pi
    noises = [noise_prev]
    # Classification
    p_class     = 2*np.ones((n_points,1))
    p_class[0], p_class[-1]  = 1, 1
    # Filament
    filament = np.append(points, p_class, axis=1)
    filament = np.append(filament, np.zeros(p_class.shape), axis=1)
    filaments_list = [filament]
    # BRANCHING FILAMENTS
    ln_ratio = (length/n_points)
    d_max = np.sqrt(11)*ln_ratio
    max_alpha_correction = 4
    # Initialisation
    x0 = np.random.triangular(2*d_max, np.min([4*d_max, length-2.1*d_max]), length-2*d_max)
    [y0, z0] = np.random.uniform(-noise_prev, noise_prev, 2)
    alpha_correction_factor = max_alpha_correction * ( (2*((x0/length)**2 + (1-x0/length)**2 - 0.5))**2)
    alpha = np.random.uniform(np.pi/(8-alpha_correction_factor), np.pi*(1-1/(8-alpha_correction_factor)))
    beta =  np.random.uniform(-np.pi/2, np.pi/2)
    n_points0 = int( np.max([np.ceil(np.random.uniform(0.5*length, 5*length)/ln_ratio), 14]) )
    length0 = ln_ratio*n_points0
    while (x0 < length-2*d_max) and (alpha<alpha_prev):
        # Geometry
        points, noise_prev = makefilament_linear(p0=[x0,y0,z0], angles=[alpha,beta], length=length0, n_points=n_points0)
        # Classification
        p_class     = 2*np.ones((n_points0,1))
        p_class[0], p_class[-1]  = 1, 1
        # Filament
        filament = np.append(points, p_class, axis=1)
        filament = np.append(filament, np.zeros(p_class.shape), axis=1)
        filaments_list.append(filament)
        noises.append(noise_prev)
        # Initialisation for next filament - if any
        x0_prev = x0
        alpha_prev = alpha
        x0 = np.random.uniform(x0_prev+3*d_max, length-2*d_max)
        [y0, z0] = np.random.uniform(-noise_prev, noise_prev, 2)
        alpha_correction_factor = max_alpha_correction * ( (2*(((x0-x0_prev)/(length-x0_prev))**2 + ((length-x0)/(length-x0_prev))**2 - 0.5))**2)
        if (np.pi/(8-alpha_correction_factor)) < (alpha_prev - np.pi*1/(8-alpha_correction_factor)): 
            alpha = np.random.uniform(np.pi/(8-alpha_correction_factor), alpha_prev - np.pi*1/(8-alpha_correction_factor))
        else:
            break
        beta =  np.random.uniform(-np.pi/2, np.pi/2)
        n_points0 = int( np.max([np.ceil(np.random.uniform(0.5*length, 4*length)/ln_ratio), 14]) )
        length0 = ln_ratio*n_points0
    # CLASSIFICATION OF ALL FILAMENTS
    for fi, ni in zip(filaments_list, noises):
        for fj, nj in zip(filaments_list, noises):
            if np.array_equal(fi,fj): continue
            ''' Filament "fj" to be classified with respect to the points of "fi" '''
            d_thresh = np.sqrt(11)*np.max([ni, nj])
            for k in range(len(fj[:,0])):
                d_base_min = np.min( np.linalg.norm(fj[k,0:3]-fi[:,0:3], axis=1) )
                if d_base_min < d_thresh:
                    fj[k,3] = 3
    # FILAMENTS AGGREGATION
    filament = np.empty((0,5))
    for f in filaments_list:
        filament = np.append(filament, f, axis=0)
    # ISOLATED POINTS
    n_isolated = int( np.random.uniform(len(filaments_list)+2,len(filaments_list)+20) )
    f = makefilaments_isolated_points(n_isolated, 1.5*d_max, filament_points=filament[:,0:3])
    filament = np.append(filament, f, axis=0)
    # Rotation
    [alpha, beta, gamma] = np.random.uniform(0, 2*np.pi, 3)
    rotation_matrix = matrix_rotate3D(alpha, beta, gamma)
    for row in filament:
        row[0:3] = np.matmul(rotation_matrix, row[0:3])
    # OUT
    return filament


def makefilament_004(n_points: int=30, length=10) -> np.ndarray:
    ''' 
        Linear filaments in 3D with 4-ways junctions
        One filament with one 4-way junction and, possibly, another 3/4 way junction.
        Points of 2 filaments are of a 4-way junction if they are closer than the maximum distance between 2 points in any of the filaments, given the cubic noise used here in the dataset.
        Main filament on XY plane, others can be outside of plane through a random rotation around the x-axis
        Created on the XY plane and then rotated randomly through yaw, pitch, roll.
        Input:
            n_points : number of points in the filament [14, 500]
            length: total length of the segment 
    '''
    n_points = 14 if n_points < 14 else n_points
    # BASE FILAMENT
    # Geometry
    points, noise_prev = makefilament_linear(p0=[0,0,0], angles=[0,0], length=length, n_points=n_points)
    noises = [noise_prev]
    # Classification
    p_class     = 2*np.ones((n_points,1))
    p_class[0], p_class[-1]  = 1, 1
    # Filament
    filament = np.append(points, p_class, axis=1)
    filament = np.append(filament, np.zeros(p_class.shape), axis=1)
    # BRANCHING FILAMENTS
    filaments_list = [filament]
    ln_ratio = (length/n_points)
    d_max = np.sqrt(11)*ln_ratio
    max_alpha_correction = 4
    # FILAMENT 1
    x0 = np.random.triangular(2*d_max, np.min([3*d_max, length-2.1*d_max]), length-2*d_max)
    [y0, z0] = np.random.uniform(-noise_prev, noise_prev, 2)
    alpha_correction_factor = max_alpha_correction * ( (2*((x0/length)**2 + (1-x0/length)**2 - 0.5))**2)
    alpha = np.random.uniform(np.pi/(8-alpha_correction_factor), np.pi*(1-1/(8-alpha_correction_factor)))
    beta =  np.random.uniform(-np.pi/2, np.pi/2)
    n_points0 = int( np.max([np.ceil(np.random.uniform(0.5*length, 5*length)/ln_ratio), 14]) )
    length0 = ln_ratio*n_points0
    # Geometry
    points, noise_prev = makefilament_linear(p0=[x0,y0,z0], angles=[alpha,beta], length=length0, n_points=n_points0)
    noises.append(noise_prev)
    # Classification
    p_class     = 2*np.ones((n_points0,1))
    p_class[0], p_class[-1]  = 1, 1
    # Filament
    filament = np.append(points, p_class, axis=1)
    filament = np.append(filament, np.zeros(p_class.shape), axis=1)
    filaments_list.append(filament)
    # FILAMENT 2 - same root point of filament 1
    [y0, z0] = np.random.uniform(-noise_prev, noise_prev, 2)
    end_is_too_near = True
    while end_is_too_near is True:
        alpha = np.random.uniform(np.pi/(8-alpha_correction_factor), np.pi*(1-1/(8-alpha_correction_factor)))
        beta =  np.random.uniform(-np.pi/2, np.pi/2)
        n_points0 = int( np.max([np.ceil(np.random.uniform(0.5*length, 5*length)/ln_ratio), 14]) )
        length0 = ln_ratio*n_points0
        # Geometry
        points, noise_prev = makefilament_linear(p0=[x0,y0,z0], angles=[alpha,beta], length=length0, n_points=n_points0)
        # Distance calculator
        dist_min = np.min( np.linalg.norm(points[-1,:] - (filaments_list[0])[:,0:3], axis=1) )
        end_is_too_near = True if dist_min < 5*d_max else False
    noises.append(noise_prev)
    # Classification
    p_class     = 2*np.ones((n_points0,1))
    p_class[0], p_class[-1]  = 1, 1
    # Filament
    filament = np.append(points, p_class, axis=1)
    filament = np.append(filament, np.zeros(p_class.shape), axis=1)
    filaments_list.append(filament) 
    # CLASSIFICATION OF FIRST 3 FILAMENTS FILAMENTS
    for fi, ni in zip(filaments_list, noises):
        for fj, nj in zip(filaments_list, noises):
            if np.array_equal(fi,fj): continue
            ''' Filament "fj" to be classified with respect to the points of "fi" '''
            d_thresh = np.sqrt(11)*np.max([ni, nj])
            for k in range(len(fj[:,0])):
                d_base_min = np.min( np.linalg.norm(fj[k,0:3]-fi[:,0:3], axis=1) )
                if d_base_min < d_thresh:
                    fj[k,3] = 4
    # FILAMENT 3
    x0_prev = x0
    if x0_prev+3*d_max < length-2*d_max:
        x0 = np.random.uniform(x0_prev+3*d_max, length-2*d_max)
        [y0, z0] = np.random.uniform(-noise_prev, noise_prev, 2)
        alpha_correction_factor = max_alpha_correction * ( (2*(((x0-x0_prev)/(length-x0_prev))**2 + ((length-x0)/(length-x0_prev))**2 - 0.5))**2)
        is_too_near = True
        counter = 0
        while is_too_near and counter < 1000:
            alpha = np.random.uniform(np.pi/(8-alpha_correction_factor), np.pi*(1-1/(8-alpha_correction_factor)))
            beta =  np.random.uniform(-np.pi/2, np.pi/2)
            n_points0 = int( np.max([np.ceil(np.random.uniform(0.5*length, 5*length)/ln_ratio), 14]) )
            length0 = ln_ratio*n_points0
            # Geometry
            points, noise_prev = makefilament_linear(p0=[x0,y0,z0], angles=[alpha,beta], length=length0, n_points=n_points0)
            # Distance calculator
            dist_min_1 = 1e20
            for p in points:
                dist_min_1_temp = np.min( np.linalg.norm(p - (filaments_list[1])[:,0:3], axis=1) )
                dist_min_1 = dist_min_1_temp if dist_min_1_temp < dist_min_1 else dist_min_1
            dist_min_2 = 1e20
            for p in points:
                dist_min_2_temp = np.min( np.linalg.norm(p - (filaments_list[2])[:,0:3], axis=1) )
                dist_min_2 = dist_min_2_temp if dist_min_2_temp < dist_min_2 else dist_min_2
            dist_min = np.min([dist_min_1, dist_min_2])
            is_too_near = True if dist_min < 3*d_max else False
            counter+=1
        noises.append(noise_prev)
        # Classification
        p_class     = 2*np.ones((n_points0,1))
        p_class[0], p_class[-1]  = 1, 1
        # Filament
        filament = np.append(points, p_class, axis=1)
        filament = np.append(filament, np.zeros(p_class.shape), axis=1)
        filaments_list.append(filament)
        # Filament 4 - same root as filament 3
        if np.random.rand() < 0.3:
            [y0, z0] = np.random.uniform(-noise_prev, noise_prev, 2)
            is_too_near = True
            counter = 0
            while is_too_near and counter < 1000:
                alpha = np.random.uniform(np.pi/(8-alpha_correction_factor), np.pi*(1-1/(8-alpha_correction_factor)))
                beta =  np.random.uniform(-np.pi/2, np.pi/2)
                n_points0 = int( np.max([np.ceil(np.random.uniform(0.5*length, 5*length)/ln_ratio), 14]) )
                length0 = ln_ratio*n_points0
                # Geometry
                points, noise_prev = makefilament_linear(p0=[x0,y0,z0], angles=[alpha,beta], length=length0, n_points=n_points0)
                # Distance calculator
                dist_min_1 = 1e20
                for p in points:
                    dist_min_1_temp = np.min( np.linalg.norm(p - (filaments_list[1])[:,0:3], axis=1) )
                    dist_min_1 = dist_min_1_temp if dist_min_1_temp < dist_min_1 else dist_min_1
                dist_min_2 = 1e20
                for p in points:
                    dist_min_2_temp = np.min( np.linalg.norm(p - (filaments_list[2])[:,0:3], axis=1) )
                    dist_min_2 = dist_min_2_temp if dist_min_2_temp < dist_min_2 else dist_min_2
                dist_min = np.min([dist_min_1, dist_min_2])
                is_too_near = True if dist_min < 3*d_max else False
                counter+=1
            noises.append(noise_prev)
            # Classification
            p_class     = 2*np.ones((n_points0,1))
            p_class[0], p_class[-1]  = 1, 1
            # Filament
            filament = np.append(points, p_class, axis=1)
            filament = np.append(filament, np.zeros(p_class.shape), axis=1)
            filaments_list.append(filament)
        # CLASSIFICATION OF LAST 1/2 FILAMENTS FILAMENTS
        if len(filaments_list) > 3:
            indexes = (0,3) if len(filaments_list)==4 else (0,3,4)
            for fi, ni in zip([filaments_list[i] for i in indexes], [noises[i] for i in indexes]):
                for fj, nj in zip([filaments_list[i] for i in indexes], [noises[i] for i in indexes]):
                    if np.array_equal(fi,fj): continue
                    ''' Filament "fj" to be classified with respect to the points of "fi" '''
                    d_thresh = np.sqrt(11)*np.max([ni, nj])
                    for k in range(len(fj[:,0])):
                        d_base_min = np.min( np.linalg.norm(fj[k,0:3]-fi[:,0:3], axis=1) )
                        if d_base_min < d_thresh:
                            fj[k,3] = 3 if len(filaments_list) == 4 else 4
    # FILAMENTS AGGREGATION
    filament = np.empty((0,5))
    for f in filaments_list:
        filament = np.append(filament, f, axis=0)
    # ISOLATED POINTS
    n_isolated = int( np.random.uniform(len(filaments_list)+2,len(filaments_list)+20) )
    f = makefilaments_isolated_points(n_isolated, 1.5*d_max, filament_points=filament[:,0:3])
    filament = np.append(filament, f, axis=0)
    # Rotation
    [alpha, beta, gamma] = np.random.uniform(0, 2*np.pi, 3)
    rotation_matrix = matrix_rotate3D(alpha, beta, gamma)
    for row in filament:
        row[0:3] = np.matmul(rotation_matrix, row[0:3])
    # OUT
    return filament


#   Splines

def make_side_branches_control_points(base_filament: np.ndarray, start_point: np.ndarray, n_points: int=4, d_mean: float=1., d_thresh: float=0) -> np.ndarray:
    '''  
        Base function to create control points for splines branching off other filaments
        Inputs:
            base_filament: np.ndarray with shape (N,3). All the filaments which constiture an "obstacle" for the new filament which fillows the control points
            start_point: the 3D point (shape (1,3)) from which to start the control points search
            n_points: total number of points in the final spline. n_points > 3
            d_thresh: the minimum distance one control point must have from any other point (of filament, of control points).
                      If no point is fount which satisfies this distance requirement, the d_mean is temporarily increased
        Outputs:
            control_points_data: a N x 3 numpy array, where N>3. Row 0: X. Row 1: Y. Row 2: Z.
    '''
    # Initialise
    if n_points < 4: n_points = 4
    control_points_data = start_point.reshape((1,3))
    curr_center = control_points_data[-1,:]
    d_mean_iter = d_mean
    # Cycle
    i = 0
    r = -1
    while i < n_points-1:
        # Make sphere
        if i == 0:
            r = 0.5*d_thresh if r == -1 else 1.05*r
        else:
            r = np.random.triangular(0.5*d_mean_iter, d_mean_iter, 1.5*d_mean_iter)
        sphere = make_fibonacci_sphere(radious=r) + curr_center
        # Compute distances
        d_sphere = []
        for s in sphere:
            d_min = np.min( np.linalg.norm(s - base_filament, axis=1) )
            d_sphere.append(d_min)
        d_sphere = np.array(d_sphere)
        # Choose step
        idx_max = np.argmax(d_sphere)
        sigma = np.min(d_sphere)/4 if i != 0 else 0.00001
        # Compute next point
        new_control_point = sphere[idx_max,:].reshape((1,3)) + np.random.normal(0, sigma, size=(1,3))
        # Final distance check
        if np.min( np.linalg.norm(new_control_point - base_filament, axis=1) ) < d_thresh:
            # Too near the filament -> try again
            d_mean_iter = 1.1*d_mean_iter
        else:
            # Append coordinates of the sphere point which is the furtherst from the base filament and from the previous control points
            control_points_data = np.append(control_points_data, new_control_point, axis=0)
            d_mean_iter = d_mean
            # Re-init for next cycle
            curr_center = control_points_data[-1,:]
            i += 1
    return control_points_data


def makefilament_spline(control_points_data: np.ndarray, n_points: int, self_crossing=False) -> np.ndarray:
    '''
        Base function to create noisy splines
        Noise and spline smoothness (distance from control points) are random
        Inputs:
            control_points_data: a 3xN numpy array, where N>3. Row 0: X. Row 1: Y. Row 2: Z.
            n_points: total number of points in the final spline. n_points >= 3*N
        Outputs:
            points: n_points x 3 numpy.array |x|y|z|
            d_noise_max: The minimum distance between two consecutive points in the spline filament. It also defines the maximum amount of noise
    '''
    if n_points < 3*control_points_data.shape[1]: n_points = 3*control_points_data.shape[1] + 1
    # Make spline
    smoothness = np.random.exponential(scale=6.0)+0.1
    tck, _ = scipy_interpolate.splprep(control_points_data, s=smoothness)
    out = scipy_interpolate.splev(np.linspace(0, 1, n_points), tck)
    points = np.array(out).T
    points = regularize_points_distances(points)
    # Make noise
    d_noise_max_1 = np.min(np.linalg.norm(points[0:-1,:]-points[1:,:], axis=1))
    d_noise_max_2 = np.min([np.linalg.norm(points[0,:]-points[1,:]) , np.linalg.norm(points[-2,:]-points[-1,:])])
    d_noise_max = (4/5)*np.max([d_noise_max_1, d_noise_max_2]) + (1/5)*np.min([d_noise_max_1, d_noise_max_2])
    noise, d_noise_max_used = makefilament_noise3D(d_max=d_noise_max, n_points=points.shape[0])
    # Check for self-crossing
    is_ok = False if not self_crossing else True
    while not is_ok:
        points_check = regularize_points_distances(points, method="augment")
        idx_offset = int(15*points_check.shape[0]/points.shape[0])
        is_ok=True
        # Check for unwanted crossings
        for i in range(points_check.shape[0]):
            idx_l_g, idx_h_g = np.max([0, i-idx_offset]), np.min([points_check.shape[0]-1, i+idx_offset])
            d = np.linalg.norm(points_check[i,:]-points_check, axis=1)
            search = np.argwhere(d<3.5*d_noise_max_used)
            self_crossing_condition = np.sum( np.logical_not(np.logical_and((search <= idx_h_g),(search >= idx_l_g))) ) > 0
            if self_crossing_condition:
                is_ok=False
                break
        if not is_ok:
            # Retry filament
            smoothness = np.random.exponential(scale=6.0)+0.1
            control_points_data += np.random.normal(0, np.mean(np.ptp(control_points_data, axis=1)) )
            tck, _ = scipy_interpolate.splprep(control_points_data, s=smoothness)
            out = scipy_interpolate.splev(np.linspace(0, 1, n_points), tck)
            points = np.array(out).T
            points = regularize_points_distances(points)
            # Retry noise
            d_noise_max_1 = np.min(np.linalg.norm(points[0:-1,:]-points[1:,:], axis=1))
            d_noise_max_2 = np.min([np.linalg.norm(points[0,:]-points[1,:]) , np.linalg.norm(points[-2,:]-points[-1,:])])
            d_noise_max = (4/5)*np.max([d_noise_max_1, d_noise_max_2]) + (1/5)*np.min([d_noise_max_1, d_noise_max_2])
            noise, d_noise_max_used = makefilament_noise3D(d_max=d_noise_max, n_points=points.shape[0])
    # Add Noise
    points += noise
    # Out
    return points, d_noise_max_used


def makefilament_005(n_points: int=50) -> np.ndarray:
    ''' 
        Splines in 3D
        One filament
        Created on the XY plane and then rotated randomly through yaw, pitch, roll.
        Input:
            n_points : number of points in the filament. Should be at least 14
    '''
    # Control points
    if n_points < 14: n_points = 14
    n_cp = np.random.randint(low=4, high=np.min([8,int(n_points/3)]) )
    xyz_max = np.random.uniform(0.1, 1000)
    cp = np.random.uniform(0, xyz_max, (3, n_cp))
    if np.random.rand() < 0.1: cp = np.sort(cp, axis=1)
    # Geometry
    points, d_max = makefilament_spline(control_points_data=cp, n_points=n_points)
    n_points = points.shape[0]
    # Classification
    p_class     = 2*np.ones((n_points,1))
    p_class[0], p_class[-1]  = 1, 1
    # Filament
    filament = np.append(points, p_class, axis=1)
    filament = np.append(filament, np.zeros(p_class.shape), axis=1)
    # ISOLATED POINTS
    n_isolated = np.random.randint(1,15)
    d_thresh = 1.5*np.sqrt(11)*d_max
    f = makefilaments_isolated_points(n_isolated, d_thresh, filament_points=filament[:,0:3])
    filament = np.append(filament, f, axis=0)
    # ROTATION
    [alpha, beta, gamma] = np.random.uniform(0, 2*np.pi, 3)
    rotation_matrix = matrix_rotate3D(alpha, beta, gamma)
    for row in filament:
        row[0:3] = np.matmul(rotation_matrix, row[0:3])
    # OUTPUT
    return filament


def makefilament_006(n_points: int=80, n_branching_points: int=3) -> np.ndarray:
    ''' 
        Splines in 3D with filaments branching off in n_branching_points random positions
        Branching filaments may be 3,4 or even 5- ways junctions
        Input:
            n_points : number of points in the filament. Should be at least 60
            n_branching_points : number of locations where filaments should branch off the main filament. 0 < n_branching_points <= 3
    '''
    # MAIN FILAMENT
    # Control points
    if n_points < 60: n_points = 60
    n_cp = np.random.randint(low=4, high=np.min([8,int(n_points/3)]) )
    xyz_max = np.random.uniform(0.1, 1000)
    cp = np.random.uniform(0, xyz_max, (3, n_cp))
    if np.random.rand() < 0.1: cp = np.sort(cp, axis=1)
    # Geometry
    n_points_base = 2*n_points if n_points/n_cp < 30 else n_points
    points_base, d_max = makefilament_spline(control_points_data=cp, n_points=n_points_base)
    d_max_list = [d_max]
    n_branching_list = [-1]
    idx_branching_list = [-1]
    # Classification
    p_class     = 2*np.ones((points_base.shape[0],1))
    p_class[0], p_class[-1]  = 1, 1
    # Filament
    filament = np.append(points_base, p_class, axis=1)
    filament = np.append(filament, np.zeros(p_class.shape), axis=1)
    filaments_list = [filament]
    # BRANCHING FILAMENTS
    if points_base.shape[0] < 150:
        n_branching_points = np.min([n_branching_points, 2])
    ln_ratio_base = np.mean([np.linalg.norm(points_base[i,:]-points_base[i+1]) for i in range(points_base.shape[0]-2)])
    n_branching_points = np.max([1, np.min([3, n_branching_points])])
    idxs_branching = np.random.choice(np.arange(5, points_base.shape[0]-5, 8), n_branching_points)
    d_mean = np.sum([np.linalg.norm(points_base[i,:] - points_base[i+1,:]) for i in range(points_base.shape[0]-2)] ) / 50
    for idx_branching in idxs_branching:
        n_branchings = np.random.choice([1,2,3], p=[0.6, 0.30, 0.10])
        for nb in range(n_branchings):
            d_thresh = 3*np.max(d_max_list)
            filament_temp = np.empty((0,5))
            for f in filaments_list:
                filament_temp = np.append(filament_temp, f, axis=0)
            # Geometry
            is_ok = False
            while not is_ok:
                is_ok = True
                n_cp = np.random.randint(4, 7)
                start_point = points_base[idx_branching,:] + np.random.uniform(-np.min(d_max_list), np.min(d_max_list), (1,3))
                cp = make_side_branches_control_points(base_filament=filament_temp[:,:3], start_point=start_point, n_points=n_cp, d_mean=d_mean, d_thresh=d_thresh)
                n_points_new = int( np.sum([np.linalg.norm(cp[i,:]-cp[i+1]) for i in range(cp.shape[0]-1)]) / ln_ratio_base )
                cp = cp.T
                points, d_max = makefilament_spline(control_points_data=cp, n_points=n_points_new )
                # Overlap check
                for i_fil in range(len( filaments_list )):
                    pdist_list = []
                    for p in points:
                        dp = np.min( np.linalg.norm(p - (filaments_list[i_fil])[:,:3], axis=1) )
                        pdist_list.append(dp) # per ogni punto del nuovo filamento, la distanza minima dal filamento considerato
                    if i_fil == 0:
                        # Base filament -> overlap allowed just on the first points
                        dp_i = np.argwhere(np.array(pdist_list) < d_thresh)
                        for j in range(len(dp_i)):
                            if j == 0:
                                if dp_i[j] > 3: is_ok = False; break
                            else:
                                if dp_i[j] - dp_i[j-1] > 4: is_ok = False; break
                    else:
                        # Other filaments
                        if idx_branching_list[-1] == idx_branching:
                            # The two filaments are spawned from the same point -> overlap allowed just on the first points
                            dp_i = np.argwhere(np.array(pdist_list) < d_thresh)
                            for j in range(len(dp_i)):
                                if j == 0:
                                    if dp_i[j] > 3: is_ok = False; break
                                else:
                                    if dp_i[j] - dp_i[j-1] > 4: is_ok = False; break
                        else:
                            # The two filaments should be completely disjointed
                            if np.min(pdist_list) < d_thresh: is_ok = False; break
                    if not is_ok: break
            d_max_list.append(d_max)
            n_branching_list.append(n_branchings)
            idx_branching_list.append(idx_branching)
            # Classification
            p_class     = 2*np.ones((points.shape[0],1))
            p_class[-1] = 1
            # Filament
            filament = np.append(points, p_class, axis=1)
            filament = np.append(filament, np.zeros(p_class.shape), axis=1)
            filaments_list.append(filament)
    # FILAMENTS CLASSIFICATION
    indexes = np.arange(len(filaments_list))
    for ii, fi, ni, nbi, idbi in zip(indexes, [filaments_list[i] for i in indexes], [d_max_list[i] for i in indexes], n_branching_list, idx_branching_list):
        for jj, fj, nj, nbj, idbj in zip(indexes, [filaments_list[i] for i in indexes], [d_max_list[i] for i in indexes], n_branching_list, idx_branching_list):
            if ii == jj: continue
            ''' Filament "fj" to be classified with respect to the points of "fi" '''
            if (idbi == idbj) or (idbi == -1) or (idbj == -1):
                # filaments should be correlated
                d_thresh = np.sqrt(11)*np.max([ni, nj])
                p_c = np.max([nbi, nbj]) + 2
                p_class = int( np.min([4,p_c]) )
                for k in range(len(fj[:,0])):
                    d_base_min = np.min( np.linalg.norm(fj[k,0:3]-fi[:,0:3], axis=1) )
                    if d_base_min < d_thresh:
                        fj[k,3] = p_class
    # FILAMENTS AGGREGATION
    filament = np.empty((0,5))
    for f in filaments_list:
        filament = np.append(filament, f, axis=0)
    # ISOLATED POINTS
    n_isolated = np.random.randint(4,15)
    d_thresh = 1.5*np.sqrt(11)*np.max(d_max_list)
    f = makefilaments_isolated_points(n_isolated, d_thresh, filament_points=filament[:,0:3])
    filament = np.append(filament, f, axis=0)
    # ROTATION
    [alpha, beta, gamma] = np.random.uniform(0, 2*np.pi, 3)
    rotation_matrix = matrix_rotate3D(alpha, beta, gamma)
    for row in filament:
        row[0:3] = np.matmul(rotation_matrix, row[0:3])
    # OUTPUT
    return filament



# Data from CAT08 dataset

def makefilamentCAT08_001(vessel, noise) -> np.ndarray:
    file_data = np.loadtxt(f"./CAT08_dataset/.../vessel{int(vessel):01}.txt", delimiter=",", skiprows=1, usecols=3)


# Final function to create the entire dataset 
# --------------------------------------

def save_pointcloud_file(points: np.ndarray, save_index: int, subdirectory: str):
    '''
        Saves dataset file.
        Input:
            points: the Nx5 list of points (x, y, z, class, id)
            save_index: an index used to save the files
            subdirectory: a subdirectory of the Dataset folder
                Available subdirectories:
                    "synthetic_dataset/"
                    "CAT08_dataset/"
    '''
    if save_index < 0: raise ValueError("save_index must be >= 0")
    np.random.shuffle(points)
    fname = "./Dataset/" + subdirectory + f'data{save_index:06}' + ".csv"
    np.savetxt(fname, points, delimiter=",", header="x,y,z,class,id", fmt=["%.9f","%.9f","%.9f","%d","%d",])


def build_main_dataset_log(n_files):
    global stopper
    total = int( np.sum( [x for x in n_files.values()] ) )
    pulse = True
    count_tot = 0
    print( "Building FilmentClassifierNET dataset")
    print( "-----------------------------------------")
    print(f"    Total files to be generated: {total}")
    print(f"    Total synthetic files : {n_files['synthetic']}")
    print(f"    Total cat08 files : {n_files['cat08']}")
    print( "-----------------------------------------")
    print(" ")
    t_start = time.time()
    while total > count_tot:
        # Counts
        count_dir1 = len(os.listdir("./Dataset/synthetic_dataset/"))
        count_dir2 = len(os.listdir("./Dataset/CAT08_dataset/"))
        count_tot_prev = count_tot
        count_tot  = count_dir1 + count_dir2
        # Prints
        estim = (total-count_tot)*(time.time()-t_start)/count_tot - 1 if (count_tot != 0) else 0
        pulser = " | " if pulse else " - "
        pulse = not pulse
        print(f"Around {100*count_dir1/total:3.0f}% completed in {int(np.floor((time.time()-t_start)/60)):3d} min {int((time.time()-t_start)%60):2d} s.  " + pulser + " ", end="\r")
        time.sleep(1)
        if stopper: break
    print(f"\nTotal elapsed time: {int(np.floor((time.time()-t_start)/60)):3d} min {int((time.time()-t_start)%60):2d} s.")

def build_main_dataset(n_files, options="new"):
    '''
        Main function to create the whole dataset.
        Input:
            n_files: dictionary with key values defined as:
                "synthetic"
                "cat08"
            options: can be either:
                "new" -> (default) create a whole new dataset and delete all previous files
                "add" -> add files on top of the ones previously existing
    '''
    # Variables
    dirs = ["./Dataset/synthetic_dataset/", "./Dataset/CAT08_dataset/"]
    n_files_minimum = [8, 0]
    # Clean n_files
    for k, nfm in zip(n_files, n_files_minimum):
        if n_files[k] < nfm: n_files[k] = nfm
    # Manage options
    if options == "new":
        n_files_log = n_files
        for d in dirs:
            filelist = [ f for f in os.listdir(d) if f.endswith(".csv") ]
            for f in filelist: os.remove(os.path.join(d, f))
        counter = 0
    elif options == "add":
        n_files_log = n_files
        for d,k in zip(dirs, n_files):
            nf = len([f for f in os.listdir(d) if f.endswith(".csv") ])
            n_files_log[k] += nf
    else:
        print("Option \"" + options + "\" is not supported. Quitting...")
        quit()
    # Setup parallel log system
    global stopper
    stopper = False
    worker = Thread(target=build_main_dataset_log, args=[n_files_log])
    worker.daemon = True
    worker.start()
    # START SYNTHETIC
    # ---------------
    syn_rates = [5, 5, 10, 10, 15, 10, 20, 50]
    syn_rates = np.array(syn_rates)/np.sum(syn_rates)
    # -- LINEAR
    for i in range( int(np.ceil(np.max([1, syn_rates[0] *n_files["synthetic"]]))) ):
        n = int( np.random.triangular(3, 80, 200) )
        l = np.random.uniform(0.01, 1000)
        pts = makefilament_001(n, l)
        save_pointcloud_file(pts, save_index=counter, subdirectory="synthetic_dataset/")
        counter += 1
    for i in range( int(np.ceil(np.max([1, syn_rates[1] *n_files["synthetic"]]))) ):
        n = int( np.random.triangular(3, 80, 200) )
        l = np.random.uniform(0.01, 1000)
        n_f = int( np.random.triangular(1, 5, 15) )
        pts = makefilament_002(n, l, n_f)
        save_pointcloud_file(pts, save_index=counter, subdirectory="synthetic_dataset/")
        counter += 1
    for i in range( int(np.ceil(np.max([1, syn_rates[2] *n_files["synthetic"]]))) ):
        n = int( np.random.triangular(14, 80, 200) )
        l = np.random.uniform(0.01, 1000)
        pts = makefilament_003(n, l)
        save_pointcloud_file(pts, save_index=counter, subdirectory="synthetic_dataset/")
        counter += 1
    for i in range( int(np.ceil(np.max([1, syn_rates[3] *n_files["synthetic"]]))) ):
        n = int( np.random.triangular(14, 80, 200) )
        l = np.random.uniform(0.01, 1000)
        pts = makefilament_aggregator(makefilament_003, n, l, 10)
        save_pointcloud_file(pts, save_index=counter, subdirectory="synthetic_dataset/")
        counter += 1
    for i in range( int(np.ceil(np.max([1, syn_rates[4] *n_files["synthetic"]]))) ):
        n = int( np.random.triangular(14, 80, 200) )
        l = np.random.uniform(0.01, 1000)
        pts = makefilament_004(n, l)
        save_pointcloud_file(pts, save_index=counter, subdirectory="synthetic_dataset/")
        counter += 1
    for i in range( int(np.ceil(np.max([1, syn_rates[5] *n_files["synthetic"]]))) ):
        n = int( np.random.triangular(14, 80, 200) )
        l = np.random.uniform(0.01, 1000)
        pts = makefilament_aggregator(makefilament_004, n, l)
        save_pointcloud_file(pts, save_index=counter, subdirectory="synthetic_dataset/")
        counter += 1
    # -- SPLINES
    for i in range( int(np.ceil(np.max([1, syn_rates[6] *n_files["synthetic"]]))) ):
        n = int( np.random.triangular(14, 80, 200) )
        pts = makefilament_005(n_points=n)
        save_pointcloud_file(pts, save_index=counter, subdirectory="synthetic_dataset/")
        counter += 1
    for i in range( int(np.ceil(np.max([1, syn_rates[7] *n_files["synthetic"]]))) ):
        n = int( np.random.triangular(60, 120, 300) )
        n_b = np.random.randint(1, 4)
        pts = makefilament_006(n_points=n, n_branching_points=n_b)
        save_pointcloud_file(pts, save_index=counter, subdirectory="synthetic_dataset/")
        counter += 1
    # End parallel log system
    stopper = True
    time.sleep(3)
    worker.join()
    # Visualisa dataset statistics
    view_dataset_statistics()

# Statistics
# ----------

def view_dataset_statistics():
    # Variables
    dirs = ["./Dataset/synthetic_dataset/", "./Dataset/CAT08_dataset/"]
    classes = [0, 1, 2, 3, 4]
    # Retrieve data
    tot_files = 0
    classes_list = []
    structures_list = []
    for d in dirs:
        f_list = [ f for f in os.listdir(d) if f.endswith(".csv") ]
        tot_files += len(f_list)
        for f in f_list:
            content_classes = np.loadtxt(os.path.join(d, f), delimiter=",", skiprows=1, usecols=3).flatten()
            content_structures = np.loadtxt(os.path.join(d, f), delimiter=",", skiprows=1, usecols=4).flatten()
            classes_list.append(content_classes)
            structures_list.append(np.max(content_structures)+1)
    # Files numerosity
    print("\n\nDATASET STATISTICS")
    print("------------------------------------------------------------------")
    print(f"-  {int(tot_files)} total files, of which:")
    for d in dirs:
        print(f"     {len([f for f in os.listdir(d) if f.endswith('.csv') ]):5d} inside \""+d+"\" ")
    # Filaments
    print(f"-  {int(np.sum(structures_list)):d} total filaments.")
    # Classes
    print(f"-  {int(np.concatenate(classes_list).flatten().shape[0]):d} total points subdivided as follows:")
    for cl in classes:
        n = 0
        for content_classes in classes_list:
            n += np.sum(content_classes == cl)
        print(f"    Class {cl:d}: {n:8d} points")
    print("------------------------------------------------------------------\n\n")





# Visualization
# -------------

def visualize_dataset_3D(p: np.ndarray, title=""):
    fig = plt.figure()
    if np.max(p[:,4]) == 0:
        ax = fig.add_subplot(projection='3d')
    elif np.max(p[:,4]) > 0:
        ax  = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
    # PLOT 1
    # ------
    # Isolated points
    idx = p[:,3]==0 
    if np.sum(idx) != 0: ax.scatter(p[idx,0], p[idx,1], p[idx,2], c='tab:olive', s=15, label=f"Class 0: {np.sum(idx)} points")
    # Filament endpoints
    idx = p[:,3]==1 
    if np.sum(idx) != 0: ax.scatter(p[idx,0], p[idx,1], p[idx,2], c='tab:red', s=35, label=f"Class 1: {np.sum(idx)} points")
    # Filament internal point
    idx = p[:,3]==2 
    if np.sum(idx) != 0: ax.scatter(p[idx,0], p[idx,1], p[idx,2], c='tab:cyan', s=5, label=f"Class 2: {np.sum(idx)} points", alpha=0.8)
    # Filament points inside/near a 3-way junction
    idx = p[:,3]==3 
    if np.sum(idx) != 0: ax.scatter(p[idx,0], p[idx,1], p[idx,2], c='tab:blue', s=55, label=f"Class 3: {np.sum(idx)} points")
    # Filaments points inside/ near a 4-way junction
    idx = p[:,3]==4 
    if np.sum(idx) != 0: ax.scatter(p[idx,0], p[idx,1], p[idx,2], c='tab:purple', s=55, label=f"Class 4: {np.sum(idx)} points")
    # Other visualisation stuff
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.suptitle("Filament point cloud visualisation\nColored classes")
    ax.set_title(title)
    # Bounding box for visualization purposes
    x_c = np.mean([p[:,0].min(), p[:,0].max()])
    y_c = np.mean([p[:,1].min(), p[:,1].max()])
    z_c = np.mean([p[:,2].min(), p[:,2].max()])
    rm = np.array([np.ptp(p[:,0]), np.ptp(p[:,1]), np.ptp(p[:,2])]).max()
    ax.scatter(x_c+rm/2, y_c+rm/2, z_c+rm/2, c='white', alpha=0.01)
    ax.scatter(x_c+rm/2, y_c+rm/2, z_c-rm/2, c='white', alpha=0.01)
    ax.scatter(x_c+rm/2, y_c-rm/2, z_c+rm/2, c='white', alpha=0.01)
    ax.scatter(x_c+rm/2, y_c-rm/2, z_c-rm/2, c='white', alpha=0.01)
    ax.scatter(x_c-rm/2, y_c+rm/2, z_c+rm/2, c='white', alpha=0.01)
    ax.scatter(x_c-rm/2, y_c+rm/2, z_c-rm/2, c='white', alpha=0.01)
    ax.scatter(x_c-rm/2, y_c-rm/2, z_c+rm/2, c='white', alpha=0.01)
    ax.scatter(x_c-rm/2, y_c-rm/2, z_c-rm/2, c='white', alpha=0.01)
    # PLOT 2
    # ------
    if np.max(p[:,4]) > 0:
        n_classes = int(np.max(p[:,4])+2)
        cmap_array = np.linspace(0,0.8, num=n_classes)
        ax2.scatter(p[:,0], p[:,1], p[:,2], c=p[:,4], cmap="rainbow")
        # Other visualisation stuff
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("z")
        ax2.set_title("Multi-filaments classification")
        # Bounding box for visualization purposes
        ax2.scatter(x_c+rm/2, y_c+rm/2, z_c+rm/2, c='white', alpha=0.01)
        ax2.scatter(x_c+rm/2, y_c+rm/2, z_c-rm/2, c='white', alpha=0.01)
        ax2.scatter(x_c+rm/2, y_c-rm/2, z_c+rm/2, c='white', alpha=0.01)
        ax2.scatter(x_c+rm/2, y_c-rm/2, z_c-rm/2, c='white', alpha=0.01)
        ax2.scatter(x_c-rm/2, y_c+rm/2, z_c+rm/2, c='white', alpha=0.01)
        ax2.scatter(x_c-rm/2, y_c+rm/2, z_c-rm/2, c='white', alpha=0.01)
        ax2.scatter(x_c-rm/2, y_c-rm/2, z_c+rm/2, c='white', alpha=0.01)
        ax2.scatter(x_c-rm/2, y_c-rm/2, z_c-rm/2, c='white', alpha=0.01)
    # Plot
    plt.tight_layout(pad=0)
    plt.show()
