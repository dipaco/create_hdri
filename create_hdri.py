#!/usr/bin/python

import numpy as np
import sys
from os import listdir
from os.path import isfile, join, splitext, basename
from scipy.misc import imread, imsave
from matplotlib.pyplot import plot, show, scatter, title, xlabel, ylabel, savefig
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from scipy.sparse.linalg import spsolve

APP_NAME = splitext(basename(sys.argv[0]))[0]
APP_PREFIX = '[' + APP_NAME + '] '

def read_images(path):
    '''Reads all the images in folder <path> (any common format), and convert every
        image into a numpy array. Every image is then stored in a dictionary in the
        following way:
        {
            t_exposure_image1 : numpy.array(rows, cols, 3),
            t_exposure_image2 : numpy.array(rows, cols, 3),
            ...
        }'''
    files = [f for f in listdir(path) if isfile(join(path, f))]
    imgs = dict()
    for f in files:
        print APP_PREFIX + ' Reading image: ' + f
        shutter_time_text = splitext(f)[0]
        shutter_time = float(shutter_time_text[1:])
        imgs[shutter_time] = imread(join(path, f))

    return imgs

def save_pfm(filename, image, scale=1):
    '''
    Save a Numpy array to a PFM file.
    '''
    color = None
    file = open(filename, "wb")
    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n' % scale)

    image.tofile(file)

def get_samples(imgs_array, channel, num_points):
    '''Returns a matrix with intensity values (0-255) with many rows as sample points,
        and many columns as exposure times (num. of images)'''
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
    '''Finds all values for the discrete log of exposure function, as well as the Radiance
        values (E) for each point'''

    #max num of intensity levels
    num_gray_levels = 256
    n = Z.shape[0]
    p = Z.shape[1]
    num_rows = n*p + num_gray_levels -2 + 1
    num_cols = num_gray_levels + n

    A = np.zeros((num_rows, num_cols))
    b = np.zeros((num_rows, 1))

    #Fill the coeficients in matrix A, according to the equation: G(z) = ln(E) + ln(dt)
    #multiplied by a weighting function (w)
    k = 0
    for j in range(0, p):
        for i in range(0, n):
            z_value = Z[i, j]
            w_value = w(z_value)
            A[k, z_value] = w_value
            A[k, num_gray_levels + i] = -w_value
            b[k, 0] = w_value * np.log(B[j])
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

    #Solve the equation system
    U, s, V = np.linalg.svd(A, full_matrices=False)
    m = np.dot(V.T, np.dot( np.linalg.inv(np.diag(s)), np.dot(U.T, b)))
    #m = np.linalg.lstsq(A, b)[0]

    return m[0:256], m[256:]

def write_hdr(filename, image):
    '''Writes a HDR image into disk. Assumes you have a np.array((height,width,3), dtype=float)
        as your HDR image'''
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

def create_radiance_map(imgs, G, w):
    '''Using the log. exposure function (G) create a radiance map
        for every channel in the image'''

    #returns a function to calculate the Radiance of a associated
    #to an intensity value
    def map_z_values(exposure_time):
        return np.vectorize(lambda z: G[z] - np.log(exposure_time))

    #vector form of weighting function
    get_w_values = np.vectorize(w)

    #Reduce noise by weighting the E values
    img_shape = imgs[imgs.keys()[0]].shape
    R = np.zeros(img_shape)
    W = np.zeros(img_shape, dtype=float)
    for dt in imgs:
        print APP_PREFIX + 'Processing image with dt = ', dt
        W_aux = get_w_values(imgs[dt])
        R += W_aux * (map_z_values(dt))(imgs[dt]).reshape(img_shape)
        W += W_aux

    return R / W

def tonemap(R):
    '''
        convert R in range rmin to rmax to the range 0..240 degrees which
        correspond to the colors red..blue in the HSV colorspace
        lower values will have the value 0 and greater values will have
        the value 240. Then convert hsv color (h,1,1) to its rgb equivalent
        note: the hsv_to_rgb() function expects h to be in the range 0..1 not 0..360
    '''
    rmax = np.amax(R)
    rmin = np.amin(R)
    H = 240 * (rmax - R) / (rmax - rmin)
    TM_aux = np.ones(R.shape + (3,), dtype=float)
    TM_aux[:, :, 0] = H / 360

    return hsv_to_rgb(TM_aux)

if __name__=='__main__':

    #Print some (hopefully) useful documentation
    if ( len(sys.argv) != 3 ):
        print   'Usage: ' + APP_NAME + ' <input_folder> <output_folder>\n' \
                '<input_folder> ->  Directory containing only images. Each one has to be named as follows:\n' \
                '                   t<time_exposure_1>.png\n' \
                '                   t<time_exposure_2>.jpg\n' \
                '                   t<time_exposure_3>.png\n' \
                '                   ...\n' \
                '\n' \
                '                   <time_exposure> has to be in seconds.\n' \
                '\n' \
                '<output_folder> -> Directoy to store the results. HDR image, PFM image, Radiance map,\n' \
                'PNG file, and the fitted curve for log exposure.\n'
        exit(-1)

    L_CHANNEL = 0
    a_CHANNEL = 1
    b_CHANNEL = 2

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    #read images
    imgs_array = read_images(input_folder)

    #sampling
    channel = L_CHANNEL
    num_samples = 1500.0 / len(imgs_array.keys())
    Z, B = get_samples(imgs_array, channel, num_samples)
    n, p = Z.shape

    #Fitting the curve
    Zmin = 0.0      #np.amin(Z)
    Zmax = 255.0    #np.amax(Z)
    w_hat = lambda z: z - Zmin + 1 if z <= (Zmin + Zmax)/2 else Zmax - z + 1
    l = 550
    print APP_PREFIX + 'Fitting log exposure curve...'
    G, E = fit_response(Z, B, l, w_hat)

    #creating radiance map for the channel
    print APP_PREFIX + 'Creating radiance map (could take a while...)'
    relative_R = create_radiance_map(imgs_array, G, w_hat)
    R = np.exp(relative_R)


    tonemap_filename = join(output_folder, 'tonemap.png')
    hdr_filename = join(output_folder, 'output.hdr')
    pfm_filename = join(output_folder, 'output.pfm')
    png_filename = join(output_folder, 'output.png')
    print APP_PREFIX + 'Saving HDR image on: ', hdr_filename
    write_hdr(hdr_filename, R)
    print APP_PREFIX + 'Saving PFM image on: ', pfm_filename
    save_pfm(pfm_filename, np.float32(R))
    print APP_PREFIX + 'Saving Tonemap for the scene on: ', tonemap_filename
    imsave(tonemap_filename, tonemap(relative_R[:, :, channel]))
    print APP_PREFIX + 'Saving HDR image on: ', png_filename
    #Gamma compression
    gamma = 0.5
    imsave(png_filename, np.power(relative_R, gamma))

    print APP_PREFIX + 'Creating and saving response function plot'
    #Creates a plot for the response curve
    plot(G, np.arange(256))
    title('RGB Response function')
    xlabel('log exposure')
    ylabel('Z value')
    savefig(join(output_folder, 'response_curve.png'))

    print APP_PREFIX + 'Done.'