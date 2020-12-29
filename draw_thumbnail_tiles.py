''' Draw thumbnail with tiles for visual inspection '''

import argparse
import sys , os
import numpy as np
import pandas as pd
import multiprocessing 
import time
from PIL import Image
from lib.utils.utils import from_linux_to_windows_path , draw_rectangles_on_thumbnail 


# ==================================
# Get input arguments
# ==================================

def examples():
    print( '\nEXAMPLE:\n"""' )
    print( 'python draw_thumbnail_tiles.py -input test_output/tile_master_index/master_index_tiles_magref5.0.csv -col-wsi filepath_wsi -output test_output/tile_master_index/ -mag-tile 5 -mag-draw 1\n"""\n' )


def get_args():
    parser = argparse.ArgumentParser( prog        = 'draw_thumbnails_tiles.py' ,
                                      description = 'Draw thumbnail with tiles'     ,
                                      add_help    = False                           )

    parser.add_argument( '-input' , dest='file_in' ,
                         help='Input CSV master index' )
     
    parser.add_argument( '-output' , dest='path_out' ,
                         help='Specify output folder' )
     
    parser.add_argument( '-col-wsi' , dest='col_wsi' ,
                         help='Name of the column containing the slide filepaths' )
    
    parser.add_argument( '-mag-tile' , dest='mag_tile' , type=np.float32, default=5,
                         help='Select only tiles with the specified magnification' )
     
    parser.add_argument( '-mag-draw' , dest='mag_draw' , type=np.float32, default=1.0,
                         help='Select at which magnification to extract the thumbnail for drawing' )
     
    parser.add_argument( '-csv-out' , dest='csv_out' , action='store_true',
                         help='Enable saving of the output dataframe to CSV' )
     
    parser.add_argument( '-tile-label' , dest='tile_label' , action='store_true',
                         help='Enable use of the tile labels' )
     
    parser.add_argument( '-h' , dest='help' , action='store_true' ,
                         help='Print help and examples' )

    args = parser.parse_args()

    if args.help is True:
        parser.print_help()
        examples()
        sys.exit()

    if args.file_in is None:
        sys.exit( '\nERROR: input CSV must be specified!\n' )

    if os.path.isfile( args.file_in ) is False:
        sys.exit( '\nERROR: input CSV ' + args.csv_in + ' does not exist!\n' )

    if args.col_wsi is None:
        sys.exit( '\nERROR: column containing the WSI filepaths must be specified!\n' )
    
    return args



# ==================================
# Class DrawTiles
# ==================================

class DrawTiles:
    def __init__( self , file_in , col_wsi , 
                  path_out=None , mag_tile=2 , 
                  mag_draw=1 , enable_tile_label=False ):
        self.file_in           = file_in
        self.col_wsi           = col_wsi
        self.mag_tile          = mag_tile
        self.mag_draw          = mag_draw
        self.path_out          = path_out
        self.enable_tile_label = enable_tile_label

        self.get_data()
        self.create_output_path()

    
    def get_data( self ):
        if self.file_in.endswith( '.csv' ):
            df_in = pd.read_csv( self.file_in )
        else:
            df_in = pd.read_pickle( self.file_in )

        self.tile_minr  = 'tile_min_row'
        self.tile_minc  = 'tile_min_col'
        self.tile_maxr  = 'tile_max_row'
        self.tile_maxc  = 'tile_max_col'
        self.tile_mag   = 'tile_magnification'
        self.tile_label = 'tile_label'

        mags = np.unique( df_in[ self.tile_mag ].values )

        if self.mag_tile not in mags:
            str_mags = [ str( mag ) for mag in mags ]
            sys.exit( '\nERROR ( DrawTiles - get_data ): selected magnification ' + str( self.mag ) + \
                      ' is not available!\nChoose among ' + ','.join( str_mags ) )
        
        self.df          = df_in[ df_in[ self.tile_mag ] == self.mag_tile ]    
        self.n_rows      = self.df.shape[0]
        self.list_wsi_un = np.unique( self.df[ self.col_wsi ].values )
        self.n_wsi       = len( self.list_wsi_un )


    def create_output_path( self ):
        if self.path_out is None:
            self.path_out = os.path.abspath( os.path.dirname( self.file_in ) )
        self.path_out_qc = os.path.join( self.path_out , 'wsi_qc' )
        os.makedirs( self.path_out_qc , exist_ok=True )


    def worker( self , i ):
        file_wsi = self.list_wsi_un[i]
        df_sub   = self.df[ self.df[ self.col_wsi ] == file_wsi ]
        n_rows   = df_sub.shape[0]
        bboxes   = df_sub[ [ self.tile_minr , 
                             self.tile_minc ,
                             self.tile_maxr ,
                             self.tile_maxc ] ].values
        tile_mag = df_sub[ self.tile_mag ].values[0]
        labels   = df_sub[ self.tile_label ].values
 
        print( '\nProcessing WSI file n.', i,' out of ', self.n_wsi )
        print( 'Input file: ', file_wsi )
        print( 'Number of bounding boxes: ', df_sub.shape[0] )
        
        if self.enable_tile_label:
            thumb = draw_rectangles_on_thumbnail( file_wsi , bboxes        , 
                                                  mag      = self.mag_draw , 
                                                  bbox_mag = tile_mag      , 
                                                  labels   = labels        ,
                                                  legend   = True          )
        else:
            thumb = draw_rectangles_on_thumbnail( file_wsi , bboxes        , 
                                                  mag      = self.mag_draw , 
                                                  bbox_mag = tile_mag      , 
                                                  labels   = None          ,
                                                  legend   = False         )
 
        file_thumb = self.create_output_filename( file_wsi )
        Image.fromarray( thumb ).save( file_thumb )
        print( 'Output QC thumbnail saved to ', file_thumb )

        return [ file_thumb ] * n_rows

    
    def create_output_filename( self , file_wsi ):
        return os.path.join( self.path_out_qc , os.path.splitext( os.path.basename( file_wsi ) )[0] ) + \
                                                '_thumb_qc.png'

    
    def draw( self ):
        list_thumbs_qc  = [];  list_thumbs_qc_win = []
        
        for i in range( len( self.list_wsi_un ) ):
            results = self.worker( i )

            for result in results:
                list_thumbs_qc     += [ result ]
                list_thumbs_qc_win += [ from_linux_to_windows_path( result ) ]  

        self.df[ 'filepath_thumb_qc' ]         = list_thumbs_qc
        self.df[ 'filepath_thumb_qc_windows' ] = list_thumbs_qc_win


    def write_df( self ):
        bname    = os.path.basename( self.file_in )
        
        file_out = os.path.join( self.path_out , os.path.splitext( bname )[0] + '_qc.csv' )
        self.df.to_csv( file_out , index=False )
        
        file_out_un = os.path.join( self.path_out , os.path.splitext( bname )[0] + '_qc_unique.csv' )
        df_un       = self.df.drop_duplicates( subset=[ 'filepath_thumb_qc_windows' ] )
        df_un.to_csv( file_out_un , index=False )

        print( '\nWritten output data frames to:\n', file_out,'\n', file_out_un )



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
    print( '==      DRAW THUMBNAIL WITH TILES      ==' )
    print( '==                                     ==' )
    print( '=========================================' )
    print( '=========================================' )

    
    # Get input arguments
    args = get_args()

    print( '\nInput file: ', args.file_in )
    print( 'Column containing WSI filepaths: ', args.col_wsi )
    print( 'Thumbnail magnification level: ', args.mag_draw )
    print( 'Draw tiles at magnification: ', args.mag_tile )
    print( 'Enable tile labels: ', args.tile_label )
    
    # Init class
    et = DrawTiles( args.file_in                        ,  
                    args.col_wsi                        ,
                    path_out          = args.path_out   ,
                    mag_draw          = args.mag_draw   ,
                    mag_tile          = args.mag_tile   ,
                    enable_tile_label = args.tile_label )
    print( '\nNumber of rows in input dataframe: ', et.n_rows )
    print( 'Number of unique WSI filepaths: ', et.n_wsi )

    # Create thumbnail with the tile rectangles
    et.draw()

    # Write output dictionary
    if args.csv_out:
        et.write_df()
     
    time2 = time.time()
    
    print( '\nTotal elapsed time: ', time2 - time1 )
    print( '\n\n' )   
