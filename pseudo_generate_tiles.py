''' Pseudo-Generation of Tiles from WSIs '''

import argparse
import sys , os
import json
import numpy as np
import pandas as pd
import time
import large_image
from lib.extract_tiles.extract_tiles import extract_tiles
from lib.utils.utils import from_linux_to_windows_path


# ==================================
# Get input arguments
# ==================================

def examples():
    print( '\nExample of multi-head tile extraction:\n"""' )
    print( 'python pseudo_generate_tiles.py -input test_output/rasterized_masks/master_index.csv -col-wsi filepath_wsi -size 224 -mag 1.25,5,10,20 -mag-ref 5 -output test_output/tile_master_index/ -thres 5\n"""\n' )


def get_args():
    parser = argparse.ArgumentParser( prog        = 'pseudo_generate_tiles.py'                                    ,
                                      description = 'Pseudo-generate tiles from WSI to train multi-head approach' ,
                                      add_help    = False                                        )

    parser.add_argument( '-input' , dest='csv_in' ,
                         help='Input CSV master index' )

    parser.add_argument( '-output' , dest='path_out' , default='./',
                         help='Output path' )

    parser.add_argument( '-col-wsi' , dest='col_wsi' ,
                         help='Name of the column containing the slide filepaths' )

    parser.add_argument( '-mag' , dest='mag' , default='1.25,5.10,20',
                         help='Magnification at which to extract the tiles' )
    
    parser.add_argument( '-mag-ref' , dest='mag_ref' , type=np.float32 , default=5,
                         help='Reference magnification at which to set the grid' )

    parser.add_argument( '-size' , dest='tile_size' , type=np.int, default=224,
                         help='Size of the tiles to extract' )

    parser.add_argument( '-stride' , dest='stride' , type=np.int, default=0,
                         help='Tiling stride' )
    
    parser.add_argument( '-overlap' , dest='overlap' , type=np.float32, default=0.0,
                         help='Percentage of overlap' )

    parser.add_argument( '-thres' , dest='thres' , type=np.float, default=20,
                         help='Percentage of foreground per tile under which the tile gets discarded' )
    
    parser.add_argument( '-pil-enhance' , dest='pil_enhance_factor' , type=np.float, default=0.0,
                         help='Enhance color to facilitate extraction of tissue mask' )
     
    parser.add_argument( '-n-bad-tiles' , dest='n_bad_tiles' , type=np.int, default=10,
                         help='Number of tiles in a tile set allowed not to have enough foreground' )
     
    parser.add_argument( '-format' , dest='format' , default='.csv',
                         help='Specify output format for master index' )

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
# Tile Generation Class
# ==================================

class TileGenerator:

    def __init__( self , file_in , col_wsi , path_out , magnification='1.25,5.10,20' ,
                  mag_ref=5 , tile_size=224 , stride=1 , overlap=0.0, thres=60 , 
                  pil_enhance_factor=0.0 , format='.pkl' , n_bad_tiles=0 ):
        self.file_in            = file_in
        self.col_wsi            = col_wsi
        self.path_out           = path_out
        self.mag_ref            = mag_ref
        self.tile_size          = tile_size
        self.stride             = stride
        self.overlap            = overlap
        self.thres              = thres
        self.pil_enhance_factor = pil_enhance_factor
        self.format             = format
        self.n_bad_tiles        = n_bad_tiles

        self.get_magnification( magnification )
        self.get_data()
        self.create_output_path()


    def get_magnification( self , magnification ):
        if ',' in magnification:
            self.mags = np.array( magnification.split( ',' ) ).astype( np.float32 )
        else:
            self.mags = [ np.float( magnification ) ]


    def get_data( self ):
        self.df       = pd.read_csv( self.file_in )
        self.list_wsi = self.df[ self.col_wsi ].values

        # Get column names as defined by raster_mask.py 
        # and create_halo_annotation_mask.py
        self.bbox_minr_col   = 'bbox_mask_raster_x_top_left'
        self.bbox_minc_col   = 'bbox_mask_raster_y_top_left'
        self.bbox_maxr_col   = 'bbox_mask_raster_x_bottom_right'
        self.bbox_maxc_col   = 'bbox_mask_raster_y_bottom_right'
        self.bbox_mag_col    = 'mag_bbox_mask_raster'
        self.json_conver_col = 'filepath_label_conversion'
        self.bbox_label_col  = 'bbox_mask_raster_label_int'

        self.bbox_minr  = self.df[ self.bbox_minr_col ].values
        self.bbox_minc  = self.df[ self.bbox_minc_col ].values
        self.bbox_maxr  = self.df[ self.bbox_maxr_col ].values
        self.bbox_maxc  = self.df[ self.bbox_maxc_col ].values
        self.bbox_mag   = self.df[ self.bbox_mag_col ].values
        self.bbox_label = self.df[ self.bbox_label_col ].values

        if self.json_conver_col not in self.df.keys():
            self.json_conver = None
        else:
            self.json_conver = self.df[ self.json_conver_col ].values


    def create_output_path( self ):
        os.makedirs( self.path_out , exist_ok=True )


    def init_output_df( self ):
        str_mag        = 'at_' + str( self.mag_ref ) + 'x'
        self.col_x_ctr = 'tile_centroid_x'
        self.col_y_ctr = 'tile_centroid_y'
        self.col_mag   = 'tile_magnification'
        
        cols = [ self.col_wsi , self.bbox_minr_col ,
                 self.bbox_minc_col , self.bbox_maxr_col ,
                 self.bbox_maxc_col , self.bbox_mag_col ,
                 'mag_grid_tile_extraction' ,
                 'tile_min_row' , 'tile_min_col' , 
                 'tile_max_row' , 'tile_max_col' , self.col_mag ,
                 'tile_size' , self.col_x_ctr , self.col_y_ctr ,
                 'tile_label' ]

        self.df_out = pd.DataFrame( columns=cols )


    def worker( self , i ):
        file_wsi   = self.list_wsi[i]
        bbox       = [ self.bbox_minr[i] , self.bbox_minc[i] ,
                      self.bbox_maxr[i] , self.bbox_maxc[i] ]
        bbox_mag   = self.bbox_mag[i]
        label      = self.get_label( i )
        print( '\nProcessing ROI n.', i,' out of ', len( self.list_wsi ) )
        print( 'Input WSI: ', file_wsi )
        print( 'Bounding box: ', bbox,' at ', bbox_mag,' magnification' )
        print( 'Label: ', label )
        
        info_tiles = extract_tiles( file_wsi                                     ,
                                    bbox                                         ,
                                    mags               = self.mags               ,
                                    mag_ref            = self.mag_ref            ,
                                    tile_size          = self.tile_size          ,
                                    bbox_mag           = bbox_mag                ,
                                    stride             = self.stride             ,
                                    overlap            = self.overlap            ,
                                    thres              = self.thres              ,
                                    pil_enhance_factor = self.pil_enhance_factor ,
                                    n_bad_tiles        = self.n_bad_tiles        )

        rows = [ [ file_wsi ] + bbox + [ bbox_mag ] + [ self.mag_ref ] + info_tiles[i] + [ label ] \
                 for i in range( len( info_tiles ) ) ]

        return rows


    def get_label( self , i ):
        if self.json_conver is None:
            return None
        else:
            label_str = str( self.bbox_label[i] )
            
            with open( self.json_conver[i] ) as fp:
                  label_dict = json.load( fp )
            
            return label_dict[ label_str ]


    def create_tiles( self ):
        self.init_output_df()

        count = 0
        for i in range( len( self.list_wsi ) ):
            rows = self.worker( i )

            for row in rows:
                self.df_out.loc[ count ] = row
                count += 1


    def write_df( self ):
        if self.format == '.csv':
            ending = '.csv'
        else:
            ending = '.pkl'

        bname    = os.path.splitext( os.path.basename( self.file_in ) )[0] + '_tiles_magref' + str( self.mag_ref ) + ending 
        file_out = os.path.join( self.path_out , bname )

        self.df_out = self.df_out.drop_duplicates( subset=[ self.col_x_ctr , self.col_y_ctr , self.col_mag ] )
        cols_merge  = [ self.col_wsi ,  self.bbox_minr_col ,
                        self.bbox_minc_col , self.bbox_maxr_col ,
                        self.bbox_maxc_col , self.bbox_mag_col  ]
        df_merge    = pd.merge( self.df , self.df_out , on=cols_merge )

        if self.format == '.csv':
            df_merge.to_csv( file_out , index=False )
        elif self.format == '.pkl':
            df_merge.to_pickle( file_out )

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
    print( '==      TILE GENERATION FROM WSI       ==' )
    print( '==                                     ==' )
    print( '=========================================' )
    print( '=========================================' )

    # Get input arguments
    args = get_args()

    print( '\nInput CSV: ', args.csv_in )
    print( 'Column containing WSI filepaths: ', args.col_wsi )
    print( 'Output directory: ', args.path_out )
    print( '\nTile magnification levels: ', args.mag )
    print( 'Reference magnification: ', args.mag_ref )
    print( 'Tile size: ', args.tile_size )
    print( 'Tiling stride: ', args.stride )
    print( 'Tiling overlap: ', args.overlap )
    print( 'Foreground threshold: ', args.thres )
    print( 'PIL enhancement factor: ', args.pil_enhance_factor )
    print( 'Select number of bad tiles allowed per set: ', args.n_bad_tiles )
    print( 'Output format: ', args.format )

    # Init class
    et = TileGenerator( args.csv_in                                  ,
                        args.col_wsi                                 ,
                        args.path_out                                ,
                        magnification      = args.mag                ,
                        mag_ref            = args.mag_ref            ,
                        tile_size          = args.tile_size          ,
                        stride             = args.stride             ,
                        overlap            = args.overlap            ,
                        thres              = args.thres              ,
                        pil_enhance_factor = args.pil_enhance_factor ,
                        format             = args.format             ,
                        n_bad_tiles        = args.n_bad_tiles        )

    print( '\nNumber of WSI filepaths: ', len( et.list_wsi ) )

    # For loop on the list of WSIs
    et.create_tiles()

    # Write output dictionary
    et.write_df()

    time2 = time.time()

    print( '\nTotal elapsed time: ', time2 - time1 )
    print( '\n\n' )
