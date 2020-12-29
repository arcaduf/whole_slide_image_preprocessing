''' Library of functions to segment the foreground or tissue in WSIs '''

import sys
import numpy as np
import cv2
import scipy
from skimage.morphology import remove_small_objects, rectangle, binary_dilation , disk
from skimage import measure
from itertools import combinations



def extract_foreground( rgb_image , gaussian_kernel=15 , thres_small_obj=200 , 
                        fill_holes=False , remove_black=False ):
    """Extract the foreground from an RGB image using Otsu threshold and binary operations.
    rgb_image       : [height, width, channels]. 
    gaussian_kernel : int, size of the Gaussian kernel 
    thresh_small_obj: int, number of pixel to consider for an object.
    use_hsv         : boolean, if False transform the RGB into grayscale, if True transform the RGB into HSV
    Returns         : boolean, mask array.
    """
   
    # Convert image to HSV
    img_hsv = cv2.cvtColor( rgb_image , cv2.COLOR_RGB2HSV )
    img_t   = img_hsv[:,:,1]
    
    # Apply Gaussian smoothing
    blur = cv2.GaussianBlur( img_t ,( gaussian_kernel , gaussian_kernel ) , 0 )

    # Threshold image with OTSU 
    _, mask = cv2.threshold( blur , 0 , 255 , cv2.THRESH_BINARY + cv2.THRESH_OTSU )

    # Fill holes and remove small objects
    if fill_holes:
        mask = scipy.ndimage.binary_fill_holes( mask ) # fill holes

    mask = remove_small_objects( mask , min_size=thres_small_obj , connectivity=1 ) # remove small objects    
    
    # Remove black background areas such as those in the Camelyon-17 
    if remove_black:
        img_v  = img_hsv[:,:,2]
        blur_v = cv2.GaussianBlur(img_v, ( gaussian_kernel , gaussian_kernel ) , 0 )
        _ , mask_black = cv2.threshold( blur_v , 50 , 255 , cv2.THRESH_BINARY )
        mask_black     = 255 - mask_black
        mask_black     = binary_dilation( mask_black , selem=disk(5) )
        im, n_labels   = measure.label( mask_black , return_num=True , connectivity=1 )

        for i in range( 1 , n_labels+1 ):
            ind = np.where( im == i )
            if np.count_nonzero( mask[ind] ):
                mask[ind] = 0
    
    # Convert binary mask to binary image
    mask_out             = np.zeros( mask.shape , dtype=np.uint8 )
    mask_out[ mask > 0 ] = 255

    return mask_out



def split_big_rectangle( bbox , thres ):
    """Split big bounding boxes into smaller bounding boxes, the splitting
       is performed along the longest size to ensure that each sub-box
       contains both foreground and background.

    Parameters
    ==========

    bbox : list of ints, verteces of the bounding box, min row, min column,
           max row and max column, respectively
    thres: maximum area for each of the sub bounding boxes

    Return
    ======

    bboxes_split: list of lists of int, each list contains the verteces of newly
                  generated bounding boxes
    areas_split : areas associated to the newly generated bounding boxes
    """

    side1   = bbox[2] - bbox[0]
    side2   = bbox[3] - bbox[1]
    area    = side1 * side2 
    n_split = np.ceil( area * 1.0 / thres )

    bboxes_split = []
    areas_split  = []

    if side1 > side2:
        step = np.int( np.ceil( side1 * 1.0 / n_split ) )
        
        for i in range( bbox[0] , bbox[2] , step ):
            bbox = [ i , bbox[1] , i+step , bbox[3] ]
            area = step * (  bbox[3] - bbox[1] )

            bboxes_split += [ bbox ]
            areas_split  += [ area ]

    else:
        step = np.int( np.ceil( side2 * 1.0 / n_split ) )
        
        for i in range( bbox[1] , bbox[3] , step ):
            bbox = [ bbox[0] , i , bbox[2] , i+step ]
            area = step * (  bbox[2] - bbox[0] )

            bboxes_split += [ bbox ]
            areas_split  += [ area ]

    return bboxes_split , areas_split



def remove_bbox_within_bbox( bboxes , areas ):
    """Remove bounding boxes within bounding boxes

    Parameters
    ==========

    bboxes: list of lists of ints, each list contains the verteces of a bounding box, min row, min column, max row, max column, respectively.
    areas : list of ints, areas associated to the input bounding boxes

    Return
    ======

    inds_keep: list of ints, indices corresponding to the bounding boxes to keep
    """
    
    inds_keep = []

    for i in range( len( bboxes ) ):
        bbox1 = bboxes[i]
        flag  = True
        for j in range( len( bboxes ) ):
            bbox2 = bboxes[j]
            if j != i:
                if areas[i] < areas[j]:
                    if ( bbox1[0] >= bbox2[0] ) and ( bbox1[2] <= bbox2[2] ) and \
                           ( bbox1[1] >= bbox2[1] ) and ( bbox1[3] <= bbox2[3] ):
                        flag = False
                        break

        if flag:
            inds_keep.append( i )

    return inds_keep



def rescale_mask( mask , target_shape ):
    return transform.resize( mask , target_shape , order=0 ,
                             preserve_range=True )



def merge_masks( mask_a , mask_b , content=None ):
    '''Merge 2 masks

    Parameters
    ==========

    mask_a : 2D uint8 numpy array, pixels !=0 represent foreground pixels
    mask_b : 2D uint8 numpy array, pixels !=0 represent foreground pixels
    content: 2D uint8 numpy array, choose which mask to use to set the pixel
             values of the output mask; if None, the 1st mask will be selected

    Return
    ======

    mask_merge: 2D uint8 numpy array
    '''

    if mask_a.shape != mask_b.shape:
        mask_b = rescale_mask( mask_b , target_shape=mask_a.shape )

    mask_merge = np.zeros( mask_a.shape , dtype=np.uint8 )

    fg               = ( mask_a != 0 ) & ( mask_b != 0 ) 
    mask_merge[ fg ] = content[ fg ]

    return mask_merge



def split_mask_in_bboxes( mask , height=100 , buffer=10 ):
    '''Split mask in quasi-fitting bounding boxes
    
    Parameters
    ==========

    mask         : 2D numpy array, grey-level mask with values from 0 to 255
    height       : int, height along the Y-direction of each individual box
    buffer       : int, percentage of length to add on both side along the
                   X-axis to include a bit of background
    
    Return
    ======

    bboxes: list of lists of ints, each list contains the top left and bottom
            right corners of a single bounding box; for each list the convention
            is : [ x_top_left , y_top_left , x_bottom_right , y_bottom_right ]
    labels: list on ints, list of integer labels transferred from the mask values
    '''

    # Connected component analysis
    n_values = len( np.unique( mask ) ) 

    if n_values > 2:
        comps = measure.label( mask )
    elif n_values == 2:
        comps              = mask.copy()
        comps[ comps !=0 ] = 1
    
    props = measure.regionprops( comps )
    
    # For loop on the bounding boxes
    bboxes_raster_all = [];  labels_all = []

    for num , prop in enumerate( props ):
        mask_aux = mask.copy()
        mask_aux[ comps != num + 1 ] = 0
        bboxes_raster_comp , labels = split_component_in_bboxes( mask_aux        ,
                                                                 prop.bbox       ,
                                                                 height = height ,
                                                                 buffer = buffer )
        bboxes_raster_all += bboxes_raster_comp
        labels_all        += labels 
    
    return bboxes_raster_all , labels_all



def split_component_in_bboxes( mask , bbox , height=10 , buffer=10 ):
    '''Split component in quasi-fitting bounding boxes
    
    Parameters
    ==========

    mask  : 2D numpy array, grey-level mask with values from 0 to 255
    bbox  : list of ints, bounding box top left and right bottom corners
            using the convention: [ x_top_left , y_top_left , 
                                    x_bottom_right , y_bottom_right ]
    height: int, height along the Y-direction of each individual box
    buffer: int, percentage of length to add on both side along the
            X-axis to include a bit of background
    
    Return
    ======

    bboxes: list of lists of ints, each list contains the top left and bottom
            right corners of a single bounding box; for each list the convention
            is : [ x_top_left , y_top_left , x_bottom_right , y_bottom_right ]
    labels: list on ints, list of integer labels transferred from the mask values
    '''

    # Select ROI from mask 
    roi = mask[ bbox[0]:bbox[2] , bbox[1]:bbox[3] ]
    
    # Check whether how many types of foreground pixels
    values = np.unique( roi )
    values = values[ values > 0 ]

    # For loop across the ROI
    bboxes = []; labels = []
  
    for value in values:
        for y in range( 0 , roi.shape[0] , height ):
            stripe   = roi[y:y+height,:]
            stripe[ stripe != value ] = 0

            pos = np.argwhere( stripe > 0 )
           
            if pos.shape[0]:
                x_min = np.min( pos[:,1] )
                x_max = np.max( pos[:,1] )
            
                bbox_new = [ bbox[0] + y , max( bbox[1] + x_min - buffer , 0 ) ,
                             bbox[0] + y + height , max( bbox[1] + x_max + buffer , 0 ) ]
                bboxes  += [ bbox_new ]
                labels  += [ value ]

    return bboxes , labels
