#!/usr/bin/python

import cv2.cv
import numpy as np
import sys
from os import listdir
from os.path import isfile, join, splitext
from scipy.misc import imread, imsave
from matplotlib.pyplot import plot, show, scatter, title, xlabel, ylabel, savefig
from matplotlib.colors import rgb_to_hsv
from scipy.sparse.linalg import spsolve

def read_images(path):
    '''Desc'''
    #TODO Apply filter to select just image files from the input directory
    files = [f for f in listdir(path) if isfile(join(path, f))]
    imgs = dict()
    for f in files:
        print '[create_hdri] Reading image: ' + f
        shutter_time_text = splitext(f)[0]
        shutter_time = float(shutter_time_text[1:])
        #imgs[shutter_time] = rgb_to_hsv(imread(join(path, f)))
        imgs[shutter_time] = imread(join(path, f))

    return imgs

def get_samples(imgs_array, channel, num_points):
    '''Desc'''
    #Samples points
    img_shape = imgs_array[imgs_array.keys()[0]].shape
    sp_x = np.random.randint(0, img_shape[0]-1, (num_points, 1))
    sp_y = np.random.randint(0, img_shape[1]-1, (num_points, 1))
    sp = np.concatenate((sp_x, sp_y), axis=1)

    n = len(sp)
    p = len(imgs_array)
    Z = np.zeros((n, p))
    B = np.zeros((p, 1))

    for i in range(0, n):
        j = 0
        for key in sorted(imgs_array):
            img = imgs_array[key][:, :, channel]
            row = sp[i, 0]
            col = sp[i, 1]
            Z[i, j] = img[row, col]
            B[j, 0] = key
            j += 1

    return Z, B

def fit_response(Z, B, l, w):
    '''Desc'''

    num_gray_levels = 256
    n = Z.shape[0]
    p = Z.shape[1]
    num_rows = n*p + num_gray_levels -2 + 1
    num_cols = num_gray_levels + n

    A = np.zeros((num_rows, num_cols))
    b = np.zeros((num_rows, 1))

    k = 0
    for j in range(0, p):
        for i in range(0, n):
            z_value = Z[i, j]
            w_value = w(z_value)
            A[k, z_value] = w_value
            A[k, num_gray_levels + i] = -w_value
            b[k, 0] = w_value*B[j]
            k += 1

    #Setting the middle value of the G function as '0'
    A[k, 128] = 1
    k += 1

    #Add the smoothness constraints
    for i in range(1, num_gray_levels-1):
        w_value = w(i)
        A[k, i-1] = l*w_value
        A[k, i] = -2*l*w_value
        A[k, i+1] = l*w_value
        k += 1

    #U, s, V = np.linalg.svd(A, full_matrices=True)
    #x_hat = spsolve(A, b)
    U, s, V = np.linalg.svd(A, full_matrices=False)
    m = np.dot(V.T, np.dot( np.linalg.inv(np.diag(s)), np.dot(U.T, b)))
    #m = np.linalg.lstsq(A, b)[0]

    return m[0:256], m[256:]

# Assumes you have a np.array((height,width,3), dtype=float) as your HDR image
def write_hdr(filename, image):
    '''Desc'''
    f = open(filename, "wb")
    f.write("#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
    f.write("-Y {0} +X {1}\n".format(image.shape[0], image.shape[1]))

    brightest = np.maximum(np.maximum(image[...,0], image[...,1]), image[...,2])
    mantissa = np.zeros_like(brightest)
    exponent = np.zeros_like(brightest)
    np.frexp(brightest, mantissa, exponent)
    scaled_mantissa = mantissa * 256.0 / brightest
    rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    rgbe[..., 0:3] = np.around(image[..., 0:3] * scaled_mantissa[..., None])
    rgbe[..., 3] = np.around(exponent + 128)

    rgbe.flatten().tofile(f)
    f.close()

def create_radiance_map(imgs, G, delta_t, w):
    '''Desc'''

    img_shape = imgs[imgs.keys()[0]].shape

    R = np.zeros(img_shape)
    W = np.zeros(img_shape, dtype=float)
    for dt in imgs:
        print '[create_hdri] Processing image with dt = ', dt
        R += np.array([w_hat(z)*(G[z] - dt) for z in np.ravel(imgs[dt])]).reshape(img_shape)
        W += np.array([w_hat(z) + 1 for z in np.ravel(imgs[dt])]).reshape(img_shape)

    return np.exp(R / W)

if __name__=='__main__':

    R_CHANNEL = 0
    G_CHANNEL = 1
    B_CHANNEL = 2

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    #read images
    imgs_array = read_images(input_folder)

    #sampling
    channel = G_CHANNEL
    num_samples = 250
    Z, B = get_samples(imgs_array, channel, num_samples)
    n, p = Z.shape

    #least squares
    Zmin = 0.0      #np.amin(Z)
    Zmax = 255.0    #np.amax(Z)
    w_hat = lambda z: z - Zmin if z <= (Zmin + Zmax)/2 else Zmax - z
    l = 250
    G, E = fit_response(Z, B, l, w_hat)

    #creating radiance map for the channel
    print '[create_hdri] Creating radiance map (could take a while...)'
    R = create_radiance_map(imgs_array, G, B, w_hat)

    output_filename = join(output_folder, 'output.png')
    print '[create_hdri] Saving HDR image on: ', output_filename
    #write_hdr(output_filename, R)
    #Gamma compression
    gamma = 0.5
    imsave(output_filename, np.power(R, gamma))

    print '[create_hdri] Showing plot'
    #Creates a plot for the response curve
    plot(G, np.arange(256))
    title('RGB Response function')
    xlabel('log exposure')
    ylabel('Z value')
    savefig(join(output_folder, 'response_curve.png'))
    #show()

    #tuning

    print '[create_hdri] Done.'