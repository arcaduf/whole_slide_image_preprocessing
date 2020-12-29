''' Extract foreground of WSI at a selected resolution '''

import argparse
import time
import sys, os, glob
import pandas as pd
from skimage import io
from PIL import Image , ImageEnhance 
import numpy as np
from lib.utils.utils import get_image_from_slide , from_linux_to_windows_path , overlay_segmentation_to_image
from lib.segment_foreground.segment_foreground import extract_foreground 



# ==================================
# Get input arguments
# ==================================

def examples():
    print( '\nEXAMPLE:\n"""' )
    print( 'python extract_foreground.py -input test_data/master_index.csv -col-wsi filepath_wsi -mag 1.0 -pil-enhance 15 -output test_output/foreground_masks/\n"""\n' )


def get_args():
    parser = argparse.ArgumentParser( prog        = 'extract_foreground.py'           ,
                                      description = 'Foreground extraction at low magnification' ,
                                      add_help    = False                                        )

    parser.add_argument( '-input' , dest='csv_in' ,
                         help='Input CSV master index' )
    
    parser.add_argument( '-col-wsi' , dest='col_wsi' ,
                         help='Name of the column containing the slide filepaths' )
    
    parser.add_argument( '-output' , dest='path_out' , default='./' ,
                         help='Output folder' )
     
    parser.add_argument( '-mag' , dest='mag' , type=np.float32, default=1.0,
                         help='Magnification at which to extract the foreground mask' )
     
    parser.add_argument( '-thres-small-obj' , dest='thres_small_obj' , type=int, default=5000,
                         help='Select threshold for small objects' )
     
    parser.add_argument( '-pil-enhance' , dest='pil_enhance_factor' , type=np.float, default=15,
                         help='Enhance color to facilitate extraction of tissue mask' )
    
    parser.add_argument( '-h' , dest='help' , action='store_true' ,
                         help='Print help and examples' )

    args = parser.parse_args()

    if args.help is True:
        parser.print_help()
        examples()
        sys.exit()

    if args.csv_in is None:
        sys.exit( '\nERROR: input CSV must be specified!\n' )

    if os.path.isfile( args.csv_in ) is False:
        sys.exit( '\nERROR: input CSV ' + args.csv_in + ' does not exist!\n' )

    if args.col_wsi is None:
        sys.exit( '\nERROR: column containing the WSI filepaths must be specified!\n' )

    return args



# ==================================
# Class to extract foreground mask
# ==================================

class ExtractMaskTissue:
    def __init__( self , file_in , col_wsi , path_out , magnification=0.5 , 
                  thres_small_obj=500 , pil_enhance_factor=0.0 ):
        self.file_in            = file_in
        self.col_wsi            = col_wsi
        self.path_out           = path_out
        self.mag                = magnification
        self.thres_small_obj    = thres_small_obj
        self.pil_enhance_factor = pil_enhance_factor

        self.get_list_wsi()
        self.create_output_path()

    
    def get_list_wsi( self ):
        self.df       = pd.read_csv( self.file_in )
        self.list_wsi = self.df[ self.col_wsi ].values


    def create_output_path( self ):
        os.makedirs( self.path_out , exist_ok=True )

        self.path_out_mask  = os.path.join( self.path_out , 'wsi_masks' )
        self.path_out_thumb = os.path.join( self.path_out , 'wsi_thumbs' )
        self.path_out_over  = os.path.join( self.path_out , 'wsi_overlays' )

        os.makedirs( self.path_out_mask , exist_ok=True )
        os.makedirs( self.path_out_thumb , exist_ok=True )
        os.makedirs( self.path_out_over , exist_ok=True )


    def init_output_df( self ):
        self.df_out = pd.DataFrame( columns=[ self.col_wsi                      ,
                                             'filepath_thumbnail'               ,
                                             'filepath_mask_foreground'         ,
                                             'filepath_overlay'                 ,
                                             'filepath_thumbnail_windows'       ,
                                             'filepath_mask_foreground_windows' ,
                                             'filepath_overlay_windows'         ,
                                             'mag_mask_foreground'              ] )


    def create_output_filenames( self , file_in ):
        bname      = os.path.splitext( os.path.basename( file_in ) )[0]
        file_thumb = os.path.join( self.path_out_thumb , bname + '_thumbnail_mag' + str( self.mag ) + '.png' )
        file_mask = os.path.join( self.path_out_mask , bname + '_foreground_mask_mag' + str( self.mag ) + '.png' )
        file_over = os.path.join( self.path_out_over , bname + '_overlay_foreground_mask_mag' + str( self.mag ) + '.png' )

        return file_thumb , file_mask , file_over


    def worker( self , file_wsi , i ):
        print( '\nProcessing slide n.', i )
        print( 'Input WSI: ', file_wsi )

        thumb = get_image_from_slide( file_wsi , mag=self.mag )

        if self.pil_enhance_factor:
            enhancer    = ImageEnhance.Color( Image.fromarray( thumb ) )
            img_enhance = np.array( enhancer.enhance( self.pil_enhance_factor ) )
            thumb       = img_enhance[:]
            
        mask = extract_foreground( thumb                                  ,
                                   thres_small_obj = self.thres_small_obj ,
                                   fill_holes      = True                 ,
                                   remove_black    = True                 )
        overlay = overlay_segmentation_to_image( thumb , mask ) 

        file_thumb , file_mask , file_over = self.create_output_filenames( file_wsi )
            
        file_thumb_win = from_linux_to_windows_path( file_thumb )
        file_mask_win  = from_linux_to_windows_path( file_mask )
        file_over_win  = from_linux_to_windows_path( file_over )

        row = [ file_wsi , file_thumb , file_mask , 
                file_over , file_thumb_win , 
                file_mask_win , file_over_win , self.mag ]
        
        Image.fromarray( thumb ).save( file_thumb )
        Image.fromarray( mask.astype( np.uint8 ) ).save( file_mask )
        Image.fromarray( overlay.astype( np.uint8 ) ).save( file_over )

        print( 'Output WSI thumbnail at magnification ' + str( self.mag ) + ' saved in: ', file_thumb )
        print( 'Output foreground mask at magnification ' + str( self.mag ) + ' saved in: ', file_mask )
        print( 'Output overlay at  magnification ' + str( self.mag ) + ' saved in: ', file_over )

        return row


    def extract_tissue( self ):
        self.init_output_df()
       
        count = 0
        for i in range( len( self.list_wsi ) ):
            self.df_out.loc[ count ] = self.worker( self.list_wsi[i] , i )
            count += 1


    def write_df( self ):
        bname    = os.path.basename( self.file_in )
        file_out = os.path.join( self.path_out , bname )
        
        df_merge = pd.merge( self.df , self.df_out , on=[ self.col_wsi ] )

        df_merge.to_csv( file_out , index=False )
        
        print( '\nWritten output data frame to: ', file_out )



# ==================================
# Main
# ==================================

if __name__ == '__main__':
    # Starting print
    time1 = time.time()

    print( '\n' )
    print( '=========================================' )
    print( '=========================================' )
    print( '==                                     ==' )
    print( '==   EXTRACT FOREGROUND MASK FROM WSI  ==' )
    print( '==       AT SELECTED MAGNIFICATION     ==' )
    print( '==                                     ==' )
    print( '=========================================' )
    print( '=========================================' )

    
    # Get input arguments
    args = get_args()

    print( '\nInput CSV: ', args.csv_in )
    print( 'Column containing WSI filepaths: ', args.col_wsi )
    print( 'Output directory: ', args.path_out )
    print( 'Magnification at which to extract the foreground mask: ', args.mag )
    print( 'Threshold for small objects: ', args.thres_small_obj )
    print( 'PIL enhancing factor: ', args.pil_enhance_factor )

    # Init class
    emt = ExtractMaskTissue( args.csv_in                                  ,    
                             args.col_wsi                                 ,
                             args.path_out                                ,
                             magnification      = args.mag                ,
                             thres_small_obj    = args.thres_small_obj    ,
                             pil_enhance_factor = args.pil_enhance_factor )
    print( '\nNumber of WSI filepaths: ', len( emt.list_wsi ) )

    # For loop on the list of WSIs
    emt.extract_tissue()

    # Write output dictionary
    emt.write_df()

    time2 = time.time()
    
    print( '\nTotal elpased time: ', time2 - time1 )
    print( '\n\n' )
