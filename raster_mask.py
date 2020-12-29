''' Split input mask into multiple quasi-fitting rectangular boxes '''

import argparse
import time
import sys, os, glob
import pandas as pd
from skimage import io
from PIL import Image
import numpy as np
from lib.utils.utils import from_linux_to_windows_path , draw_rectangles_on_thumbnail
from lib.segment_foreground.segment_foreground import split_mask_in_bboxes 



# ==================================
# Get input arguments
# ==================================

def examples():
    print( '\nEXAMPLE:\n"""' )
    print( 'python raster_mask.py -input test_output/merge_foreground_and_annotations/master_index.csv -col-mask filepath_merged_mask_foreground_and_annotation -output test_output/rasterized_masks/\n"""\n' )


def get_args():
    parser = argparse.ArgumentParser( prog        = 'raster_mask.py'                                                 ,
                                      description = 'Split input mask into multiple quasi-fitting rectangular boxes' ,
                                      add_help    = False                                                            )

    parser.add_argument( '-input' , dest='csv_in' ,
                         help='Input CSV master index' )
    
    parser.add_argument( '-col-mask' , dest='col_mask' ,
                         help='Name of the column containing the filepaths to the masks to rasterize' )
    
    parser.add_argument( '-output' , dest='path_out' , default='./' ,
                         help='Output folder' )
     
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

    if args.col_mask is None:
        sys.exit( '\nERROR: column containing the mask filepaths must be specified!\n' )

    return args



# ==================================
# Class to rasterize mask
# ==================================

class RasterMask:
    def __init__( self , file_in , col_mask , path_out ):
        self.file_in  = file_in
        self.col_mask = col_mask
        self.path_out = path_out

        self.get_list_masks()
        self.create_output_path()

    
    def get_list_masks( self ):
        self.df         = pd.read_csv( self.file_in )
        self.list_masks = self.df[ self.col_mask ].values
        col_mag         = self.col_mask.replace( 'filepath' , 'mag' )
        self.list_mags  = self.df[ col_mag ].values
        self.mag        = np.int( np.sum( self.list_mags ) / self.df.shape[0] )


    def create_output_path( self ):
        self.path_out_over  = os.path.join( self.path_out , 'wsi_overlays' )
        os.makedirs( self.path_out_over , exist_ok=True )


    def create_output_filenames( self , file_in ):
        bname     = os.path.splitext( os.path.basename( file_in ) )[0]
        file_over = os.path.join( self.path_out_over , bname + '_overlay_rasterized_mask_mag' + str( self.mag ) + '.png' )

        return file_over

    
    def init_output_df( self ):
        self.df_out = pd.DataFrame( columns=[ self.col_mask                      ,
                                             'filepath_overlay_mask_raster'      ,
                                             'mag_bbox_mask_raster'              ,
                                             'bbox_mask_raster_x_top_left'       ,
                                             'bbox_mask_raster_y_top_left'       ,
                                             'bbox_mask_raster_x_bottom_right'   ,
                                             'bbox_mask_raster_y_bottom_right'   ,
                                             'bbox_mask_raster_label_int'        ] )


    def worker( self , i ):
        file_mask = self.list_masks[i]
        mag       = self.list_mags[i]

        print( '\nProcessing slide n.', i )
        print( 'Input mask: ', file_mask )
        print( 'Magnification: ', mag )

        mask = io.imread( file_mask )
        
        bboxes , labels = split_mask_in_bboxes( mask , height=100 , buffer=10 )
        overlay         = draw_rectangles_on_thumbnail( self.df[ 'filepath_thumbnail'].values[i] ,
                                                        bboxes , mag=mag , bbox_mag=mag , labels=labels ) 

        file_over = self.create_output_filenames( file_mask )
            
        row  = [ file_mask , file_over , mag ]
        rows = [ row + [ bboxes[ind][0] , bboxes[ind][1] , bboxes[ind][2] , bboxes[ind][3] , labels[ind] ] \
                 for ind in range( len( labels ) ) ]
        
        Image.fromarray( overlay.astype( np.uint8 ) ).save( file_over )
        print( 'Output overlay at  magnification ' + str( self.mag ) + ' saved in: ', file_over )

        return rows


    def run( self ):
        self.init_output_df()
       
        count = 0
        for i in range( len( self.list_masks ) ):
            rows = self.worker( i )

            for row in rows:
                self.df_out.loc[ count ] = row
                count += 1


    def write_df( self ):
        df_merge = pd.merge( self.df , self.df_out , on=[ self.col_mask ] )
        bname    = os.path.basename( self.file_in )
        file_out = os.path.join( self.path_out , bname )
        df_merge.to_csv( file_out , index=False )
        
        print( '\nWritten output data frame to: ', file_out )
        print( 'Merged data frame shape: ', df_merge.shape )



# ==================================
# Main
# ==================================

if __name__ == '__main__':
    # Starting print
    time1 = time.time()

    print( '\n' )
    print( '=============================================' )
    print( '=============================================' )
    print( '==                                         ==' )
    print( '==  RASTER MASKS PRIOR TO TILE EXTRACTION  ==' )
    print( '==                                         ==' )
    print( '=============================================' )
    print( '=============================================' )

    
    # Get input arguments
    args = get_args()

    print( '\nInput CSV: ', args.csv_in )
    print( 'Column containing filepaths to masks: ', args.col_mask )
    print( 'Output directory: ', args.path_out )

    # Init class
    raster = RasterMask( args.csv_in   ,    
                         args.col_mask ,
                         args.path_out )
    print( '\nNumber of mask filepaths: ', len( raster.list_masks ) )

    # For loop on the list of WSIs
    raster.run()

    # Write output dictionary
    raster.write_df()

    time2 = time.time()
    
    print( '\nTotal elpased time: ', time2 - time1 )
    print( '\n\n' )
