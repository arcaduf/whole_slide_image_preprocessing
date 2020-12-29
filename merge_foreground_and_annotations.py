''' Merge foreground and annotation masks '''

import argparse
import time
import sys, os, glob
import pandas as pd
from skimage import io
import numpy as np
from lib.utils.utils import get_common_keys , from_linux_to_windows_path 
from lib.segment_foreground.segment_foreground import merge_masks 



# ==================================
# Get input arguments
# ==================================

def examples():
    print( '\nEXAMPLE:\n"""' )
    print( 'python merge_foreground_and_annotations.py -input-fg test_output/foreground_masks/master_index.csv -input-ann test_output/from_xml_to_mask/master_index_xml_to_mask.csv -output test_output/merge_foreground_and_annotations/\n"""\n' )


def get_args():
    parser = argparse.ArgumentParser( prog        = 'merge_foreground_and_annotations.py'   ,
                                      description = 'Merge foreground and annotation masks' ,
                                      add_help    = False                                   )

    parser.add_argument( '-input-fg' , dest='csv_foreground' ,
                         help='Input CSV containing the filepath of the foreground masks' )

    parser.add_argument( '-input-ann' , dest='csv_annotation' ,
                         help='Input CSV containing the filepath of the annotation masks' )

    parser.add_argument( '-output' , dest='path_out' , default='./' ,
                         help='Output folder' )
     
    parser.add_argument( '-h' , dest='help' , action='store_true' ,
                         help='Print help and examples' )

    args = parser.parse_args()

    if args.help is True:
        parser.print_help()
        examples()
        sys.exit()

    if args.csv_foreground is None:
        sys.exit( '\nERROR: input CSV must be specified!\n' )

    if os.path.isfile( args.csv_foreground ) is False:
        sys.exit( '\nERROR: input foreground CSV ' + args.csv_foreground + ' does not exist!\n' )
    
    if args.csv_annotation is None:
        sys.exit( '\nERROR: input annotation CSV must be specified!\n' )

    if os.path.isfile( args.csv_annotation ) is False:
        sys.exit( '\nERROR: input annotation CSV ' + args.csv_annotation + ' does not exist!\n' )

    return args



# ==================================
# Class to merge foreground
# and annotation masks
# ==================================

class MergeForegroundAndAnnotations:
    def __init__( self , file_fg , file_ann , path_out ):
        self.file_fg  = file_fg
        self.file_ann = file_ann
        self.path_out = path_out

        self.merge_data_frames()
        self.create_output_path()

    
    def merge_data_frames( self ):
        self.df_fg  = pd.read_csv( self.file_fg )
        self.df_ann = pd.read_csv( self.file_ann )

        keys_merge = get_common_keys( self.df_fg , self.df_ann )
        self.df_in = pd.merge( self.df_fg , self.df_ann , on=keys_merge )
        self.mag   = self.df_in[ 'mag_mask_foreground' ].values[0]
        
        print( '\nShared keys: ', keys_merge )
        print( 'Merged data frame: ', self.df_in.shape )
        print( 'Reference magnification: ', self.mag )


    def create_output_path( self ):
        self.path_out_mask = os.path.join( self.path_out , 'wsi_merged_foreground_and_annotations' )
        os.makedirs( self.path_out_mask , exist_ok=True )


    def create_output_filename( self , file_in ):
        bname     = os.path.splitext( os.path.basename( file_in ) )[0]
        file_mask = os.path.join( self.path_out_mask , bname + '_foreground_mask_mag' + str( self.mag ) + '.png' )

        return file_mask


    def worker( self , row_index ):
        print( '\nProcessing slide n.', row_index )
        file_fg  = self.df_in[ 'filepath_mask_foreground' ].values[ row_index ]
        print( file_fg )
        mask_fg  = io.imread( file_fg )
        
        file_ann = self.df_in[ 'filepath_mask_annotation' ].values[ row_index ]
        mask_ann = io.imread( file_ann )
         
        mask_merge = merge_masks( mask_fg , mask_ann , content=mask_ann )

        file_mask     = self.create_output_filename( file_fg )
        file_mask_win = from_linux_to_windows_path( file_mask )
       
        io.imsave( file_mask , mask_merge.astype( np.uint8 ) )

        print( 'Output merged mask at  magnification ' + str( self.mag ) + ' saved in: ', file_mask )

        return file_mask , file_mask_win


    def run( self ):
        list_files_1 = [];  list_files_2 = []

        for i in range( self.df_in.shape[0] ):
            file_mask , file_mask_win = self.worker( i )
            list_files_1 += [ file_mask ]
            list_files_2 += [ file_mask ]

        self.df_in[ 'filepath_merged_mask_foreground_and_annotation' ]         = list_files_1
        self.df_in[ 'filepath_merged_mask_foreground_and_annotation_windows' ] = list_files_2
        self.df_in[ 'mag_merged_mask_foreground_and_annotation' ]             = [ self.mag ] * self.df_in.shape[0]


    def write_df( self ):
        file_out = os.path.join( self.path_out , os.path.basename( self.file_fg ) )
        self.df_in.to_csv( file_out , index=False )
        print( '\nWritten output data frame to: ', file_out )



# ==================================
# Main
# ==================================

if __name__ == '__main__':
    # Starting print
    time1 = time.time()

    print( '\n' )
    print( '==========================================' )
    print( '==========================================' )
    print( '==                                      ==' )
    print( '==  MERGE FOREGROUND & ANNOTATION MASK  ==' )
    print( '==                                      ==' )
    print( '==========================================' )
    print( '==========================================' )

    
    # Get input arguments
    args = get_args()

    print( '\nInput CSV with filepaths of foreground masks: ', args.csv_foreground )
    print( 'Input CSV with filepaths of annotation masks: ', args.csv_annotation )
    print( 'Output directory: ', args.path_out )

    # Init class
    merger = MergeForegroundAndAnnotations( args.csv_foreground ,
                                            args.csv_annotation ,
                                            args.path_out       )
    print( '\nNumber of masks to merge: ', merger.df_in.shape[0] )

    # For loop on the list of WSIs
    merger.run()

    # Write output dictionary
    merger.write_df()

    time2 = time.time()
    
    print( '\nTotal elpased time: ', time2 - time1 )
    print( '\n\n' )
