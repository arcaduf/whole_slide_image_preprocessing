''' Extract tiles from WSIs '''

import os
import numpy as np
import large_image
from lib.utils.utils import get_image_from_slide , convert_coordinates , create_hash_128
from lib.segment_foreground.segment_foreground import extract_foreground
from PIL import Image
from PIL import ImageEnhance
import random
import skimage
from skimage import color



def extract_tiles( file_wsi , bbox , mags=[1.25,5,10,20] , mag_ref=5 , 
                   tile_size=512 , bbox_mag=0.5 , stride=0 , 
                   overlap=0 , thres=60 , pil_enhance_factor=0.0 , n_bad_tiles=-1 ):
    '''Extract tiles from WSI give a bounding box

    Parameters
    ==========
    file_wsi     : string, filepath to WSI
    bbox         : list of ints, it contains the minimum row, the minumum column,
                   the maximum row and the maximum column of the bounding box,
                   respectively.
    bbox_mag     : magnification of the input bounding box
    mags         : list of floats, magnifications at which to extract the tiles
    mag_ref      : float, reference magnification setting the grid
    tile_size    : int, size of the square tiles to extract
    stride       : int, tiling stride
    overlap      : float, percentage of overlap
    thres        : float, percentage of foreground characterizing the patch
                   under which to discard the patch
    pil_enhance_factor: PIL color enhancement factor
    n_bad_tiles  : int, number of tiles in a tile set allowed not to have enough
                   foreground
    
    Return
    ======
    info_tiles: list of lists of ints, each list contains the coordinates of
                each tile, its magnification and size.
    '''

    # Default value for n_bad_tiles
    if n_bad_tiles == -1:
        n_bad_tiles = len( mags )

    # Read the WSI
    slide = large_image.getTileSource( file_wsi )

    # Make sure every entry of bbox is positive
    bbox_pos = [ max( pos , 0 ) for pos in bbox ]

    # Convert bounding to the selected magnification
    bbox_new = convert_coordinates( bbox_pos , mag_start=bbox_mag , mag_end=mag_ref )

    # Extract bounding box at given magnification
    roi = [ bbox_new[1] , bbox_new[0] ,
            bbox_new[3]-bbox_new[1]  ,
            bbox_new[2]-bbox_new[0]  ]
    
    img_roi , _ = slide.getRegion( region = dict( left=roi[0] , top=roi[1] , width=roi[2] , height=roi[3] , units='mag_pixels'  ) ,
                                   scale  = dict( magnification = mag_ref ) ,
                                   format = large_image.tilesource.TILE_FORMAT_NUMPY )

    # Enhance color to facilitate the extraction of the tissue foreground
    if pil_enhance_factor:
        enhancer    = ImageEnhance.Color( Image.fromarray( img_roi ) )
        img_enhance = np.array( enhancer.enhance( pil_enhance_factor ) )


    # Get foreground mask
    if pil_enhance_factor:
        mask = extract_foreground( img_enhance , fill_holes=False , remove_black=True )
    else:
        mask = extract_foreground( img_roi , fill_holes=False , remove_black=True )
    img_gray   = color.rgb2gray( img_roi[:,:,:3] )
    mean_value = np.mean( img_gray[ mask == 255 ] )


    # Initialize output lists
    info_tiles = []


    # For loop to extract patches
    n_rows , n_cols , _ = img_roi.shape

    if overlap:
        if overlap > 0 and overlap < 100:
            stride = -np.int( overlap * tile_size / 100.0 )

    tile_size_h = np.int( 0.5 * tile_size )
    
    for ir in range( 0 , n_rows , tile_size + stride ):
        for ic in range( 0 , n_cols , tile_size + stride ):
            x_centroid = bbox_new[0] + ir + tile_size_h
            y_centroid = bbox_new[1] + ic + tile_size_h
            flag       = 0
            rows       = []

            for mag in mags:
                x0 , y0 = convert_coordinates( [ x_centroid ,  y_centroid ] , 
                                                mag_start=mag_ref , mag_end=mag )
                img_patch , _ = slide.getRegion( region = dict( left=y0-tile_size_h , top=x0-tile_size_h , width=tile_size , height=tile_size , units='mag_pixels'  ) ,
                                                   scale  = dict(magnification=mag) ,
                                                   format = large_image.tilesource.TILE_FORMAT_NUMPY )

                if len( img_patch.shape ):
                    img_patch = img_patch[:,:,:3]
                    img_patch = color.rgb2gray( img_patch )

                    mask_patch = np.zeros( ( img_patch.shape[0] , img_patch.shape[1] ) )
                    mask_patch[ img_patch <= mean_value ] = 255
                
                    if not has_enough_foreground( mask_patch , thres ):
                        flag += 1
                        
                    row = [ x0 - tile_size_h        ,
                            y0 - tile_size_h        ,
                            x0 + tile_size_h        ,
                            y0 + tile_size_h        ,
                            mag , tile_size         ,
                            x_centroid , y_centroid ]

                    rows += [ row ]

            if ( flag < n_bad_tiles ) and ( len( rows ) == len( mags ) ):
                info_tiles += rows

    print( 'N .of tiles produced: ', len( info_tiles ) )

    return info_tiles



def has_enough_foreground( mask , thres ):
    '''Has enough foreground

    Parameters
    =========

    mask : 2D array, gray-level image with 2 unique values, oand 255.
    thres: int, amount of foreground. i.e. pixels == 255, under which the
           patch should be discarded

    Return
    ======

    answer: boolean
    '''
    
    perc_255 = len( mask[ mask == 255 ] ) * 100.0 / ( mask.shape[0] * mask.shape[1] )
    
    if perc_255 < thres:
        return False
    else:
        return True



def create_tile_output_filenames( file_in , p_out1 , p_out2 , mag , 
                                  x_top_left , x_size , y_top_left , 
                                  y_size ):
    hash  = create_hash_128()
    bname = os.path.splitext( os.path.basename( file_in ) )[0]

    if p_out1 is None:
        file_tile = None
    else:
        file_tile = os.path.join( p_out1 , bname + '_tile_mag' + str( mag ) + '_xtl' + str( x_top_left ) + \
                                  '_ytl' + str( y_top_left ) + '_size' + str( x_size ) + 'x' + str( y_size ) + '_' + str( hash ) + '.png' )
    
    if p_out2 is None:
        file_mask = None
    else:
        file_mask = os.path.join( p_out1 , bname + '_mask_mag' + str( mag ) + '_xtl' + str( x_top_left ) + \
                                  '_ytl' + str( y_top_left ) + '_size' + str( x_size ) + 'x' + str( y_size ) + '_' + str( hash ) + '.png' )
 
    return file_tile , file_mask , hash
