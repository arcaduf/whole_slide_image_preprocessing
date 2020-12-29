''' Utilities to wrangle WSI '''

import os
import large_image
import numpy as np
import cv2
from skimage import io , color
import random
import xml.etree.ElementTree as ET
import webcolors



WSI_FORMATS = ( '.ndpi' , '.svs' , '.tif' )



def get_base_magnification( file ):
    slide = large_image.getTileSource( file )
    return slide.getNativeMagnification()[ 'magnification' ]        



def get_image_from_slide( file , mag=1.25 ):
    """ Extract an image at the desired magnification from a slide file
    
    Parameters
    ==========
    file: path to the slide (str) or the slide (large_image)
    mag : magnification to extract the image (float)
    
    Return
    ======
    image: array 
    """ 

    # ----- Check if slide is already the slide or the input name -----
    if type(file) is str:
        # Read the slide 
        slide = large_image.getTileSource( file )
    else:
        slide = file    
    
    # ----- Get slide at given magnification -----
    if mag == 'base':
        mag = slide.getNativeMagnification()[ 'magnification' ]        

    image , _ = slide.getRegion( scale  = dict( magnification=mag ),
                                 format = large_image.tilesource.TILE_FORMAT_NUMPY )

    return image[:,:,:3]



def from_linux_to_windows_path( fpath ):
    """Get in input a linux filepath and
       transform it into a windows one.
    
    Parameters
    ==========
    fpath : string, linux filepath
    
    Return
    ======
    fpath_win: string, windows filepath 
    """

    fpath     = os.path.abspath( fpath )
    fpath_win = '\\' + fpath.replace( '/' , '\\' )

    return fpath_win



def overlay_segmentation_to_image( ref , segm , alpha=0.3 ):
    """Overlay segmentation to original image

    Parameters
    ==========

    ref  : array, input RGB image
    segm : array, input binary segmentation ranging in [0,255]
    alpha: float, number ranging in [0,1]

    Return
    ======

    img_masked: array, input RGB image with overlaid segmentation
    """

    # Convert color image to grey with 3 channels
    ref_new = np.zeros( ( ref.shape[0] , ref.shape[1] , 3 ) , dtype=np.float32 )

    if len( ref.shape ) == 3:
        gray = color.rgb2gray( ref ) * 255

    else:
        gray = ref.copy()

    ref_new[:,:,0] = gray;  ref_new[:,:,1] = gray;  ref_new[:,:,2] = gray
    ref_new = ref_new.astype( np.uint8 )
        
        
    # Transform prediction into a heatmap
    if segm.shape[0] != ref.shape[0] or ref.shape[1] != ref.shape[1]:
        segm_shape = segm.shape
        segm       = cv2.resize( segm , ( ref.shape[1] , ref.shape[0] ) )
    else:
        segm_shape = None

    heat_map = cv2.applyColorMap( np.uint8( segm ) , cv2.COLORMAP_JET )


    # Overlay heatmap to image
    img_hsv         = color.rgb2hsv( ref_new )
    color_mask_hsv  = color.rgb2hsv( heat_map )    
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha
    
    overlay = color.hsv2rgb( img_hsv ) * 255
    overlay = overlay.astype( np.uint8 )
   
    return overlay



def get_visualization_colors( n , format='RGB' ):
    '''Select n colors for visualization purposes

    Parameters
    ==========

    n     : int, number of colors to generate
    format: string, either 'RGB' or 'BGR'

    Return
    ======

    colors: list of tuples, each tuple represent a distinct RGB or BGR color
    '''
    
    list_names = [ 'red' , 'blue' , 'green' , 'orange' , 'black' ,
                   'darkviolet' , 'saddlebrown' , 'slategray' ,
                   'gold' , 'lightcoral' , 'deepskyblue' , 'lime' ,
                   'pink' , 'olive' , 'royalblue', 'lightgreen' ]

    if n > len( list_names ):
        print( 'Warning: selected number of colors is bigger than available list of colors!' )

    colors = []

    for i in range( n ):
        num_code = webcolors.name_to_rgb( list_names[i] ) 

        if format == 'BGR':
            num_code = num_code[::-1]

        colors.append( num_code )

    return colors



def draw_rectangles_on_thumbnail( file_in , bboxes , mag=0.5 , bbox_mag=20 ,
                                  labels=None , legend=False ):
    '''Draw rectangle on thumbnail of WSI at a given magnification

    Parameters
    ==========

    file_in : string, filepath to WSI or image
    bboxes  : list on ints, list of bounding box verteces formed in order by
              the minimum row, the minimum column, the maximum row and the
              maximum column value
    mag     : magnification at which to extract the thumbnail
    bbox_mag: magnification of the bounding boxes
    labels  : list of integers, if provided different colors will be used for
              the bounding boxes
    legend  : boolean, draw a legend on the bottom right

    Return
    ======

    thumb: RGB image
    '''

    # Read slide or image
    if file_in.endswith( WSI_FORMATS ):
        thumb = get_image_from_slide( file_in , mag=mag ).astype( np.uint8 )
    else:
        thumb = io.imread( file_in )
    
    thumb = cv2.cvtColor( thumb , cv2.COLOR_RGB2BGR )
    
    # Select colors
    if labels is None:
        color      = ( 0 , 0 , 255 )
        labels     = np.zeros( len( bboxes ) )
        color_dict = { 0 : color }
        values     = np.unique( labels )
    else:
        values     = np.unique( np.array( labels ) )
        n_colors   = len( values )
        colors     = get_visualization_colors( n_colors , format='BGR' )
        color_dict = {}
        
        for i in range( n_colors ):
            color_dict.update( { values[i]: colors[i] } )

    # For loop on the ROIs
    if mag < 1:
        thick = 1
    elif mag >= 1 and mag < 2:
        thick = 2
    else:
        thick = 3
    
    for num , bbox in enumerate( bboxes ):
        bbox_new = convert_coordinates( bbox , mag_start=bbox_mag , mag_end=mag )
        thumb    = cv2.rectangle( thumb                 , 
                                 ( bbox_new[1] , bbox_new[0] ) ,
                                 ( bbox_new[3] , bbox_new[2] ) , 
                                 color_dict[ labels[ num ] ]   , 
                                 thick                         )
    
    # Add legend
    if legend:
        font  = cv2.FONT_HERSHEY_SIMPLEX
        pos   = [ np.int( 0.55 * thumb.shape[0] ) , 
                  np.int( 0.05 * thumb.shape[1] ) ]
        scale = 3
        line  = 8

        for i in range( n_colors ): 
            cv2.putText( thumb , str( values[i] ).upper() , 
                         tuple( pos ) , font , scale, 
                         color_dict[ values[ i ] ] , line )
            pos[1] = np.int( 2 * pos[1] )

    thumb = cv2.cvtColor( thumb , cv2.COLOR_BGR2RGB )
    
    return thumb



def highlight_rectangle_on_thumbnail( thumb , bbox , mag=0.5 , bbox_mag=20 ):
    '''Draw rectangle on thumbnail of WSI at a given magnification

    Parameters
    ==========

    thumb   : RGB image
    bbox    : list of ints, bounding box verteces formed in order by
              the minimum row, the minimum column, the maximum row and the
              maximum column value
    mag     : magnification at which to extract the thumbnail
    bbox_mag: magnification of the bounding boxes

    Return
    ======

    thumb_tile: RGB image
    '''

    # Read slide
    thumb = cv2.cvtColor( thumb , cv2.COLOR_RGB2BGR )
    
    # For loop on the ROIs
    bbox_new   = convert_coordinates( bbox , mag_start=bbox_mag , mag_end=mag )
    thumb_tile = cv2.rectangle( thumb                 , 
                                ( bbox_new[1] , bbox_new[0] ) ,
                                ( bbox_new[3] , bbox_new[2] ) , 
                                ( 255 , 0 , 0 )               , 
                                 3                            )
    
    thumb_tile = cv2.cvtColor( thumb_tile , cv2.COLOR_BGR2RGB )

    return thumb_tile



def convert_coordinates( coords , mag_start=1.0 , mag_end=1.0 ):
    '''Convert coordinates

    Parameters
    ==========

    coords   : list of integer coordinates
    mag_start: float, magnification level corresponding to input coordinates
    mag_end  : float, target magnficiation level

    Return
    ======

    coords_new: list of integer coordinates corresponding to target magnification
    '''
    
    if mag_start != mag_end:
        coords_new = [ np.int( coord * mag_end / mag_start ) for coord in coords ]
    else:
        coords_new = coords[:]
    
    return coords_new



def create_hash_128():
    return random.getrandbits( 128 )



def get_vertices_from_xml( file_xml , mag_xml=40.0 , mag_out=1.0 ):
    '''Get vertices and associated labels
       from XML asnnotation file

    Parameters
    ==========

    file_xml: string, path to XML annotation file
    mag_xml : float, magnification corresponding to the XML coordinates
    mag_out : float, magnification of the output vertices 

    Return
    ======

    vertices: list of lists, each list contains the tuple
              of vertices related to a certain annotations;
              the list contains as many lists as the number
              of distinct annotations in the XML file
    types   : list of strings, it contains entries that can be
              either "polygon" or "rectangle"
    labels  : list of strings, it contains the label associated
              in order to each individual list of vertices
    '''

    tree = ET.parse( file_xml )
    root = tree.getroot()

    vertices = []
    types    = []
    labels   = []

    for child in root:
        try:
            label   = child.attrib[ 'Name' ].lower()
            regions = child.find( 'Regions' )
            form    = regions.find( 'Region' ).attrib[ 'Type' ].lower()
        
            if len( regions ) > 0:
                for instance in regions:
                    vertices_ann = []
                    verts = instance.find( 'Vertices' )
                
                    for ver in verts:
                        coords     = [ int( ver.get( 'X' ) ) , int( ver.get( 'Y' ) ) ]
                        coords_new = convert_coordinates( coords , mag_start=mag_xml , mag_end=mag_out )
                        vertices_ann.append( ( coords_new[0] , coords_new[1] ) )
 
                    vertices.append( vertices_ann )
                    types.append( form )
                    labels.append( label )

        except:
            pass

    return vertices , types , labels



def get_common_keys( df_a , df_b ):
    '''Get list of keys shared between 2 data frames 

    Parameters
    =========

    df_a: pandas data frame
    df_b: pandas data frame

    Return
    ======

    shaped_keys: list of strings, list of keys that are
                 common to the 2 data frames
    '''

    keys_a = np.array( df_a.keys() )
    keys_b = np.array( df_b.keys() )

    return list( np.intersect1d( keys_a , keys_b ) )
