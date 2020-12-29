''' From XML annotations to mask '''

import argparse
import time
import sys, os, glob
import pandas as pd
from skimage import io
import numpy as np
import json
from lib.utils import utils
from PIL import Image , ImageDraw



# ==================================
# Get input arguments
# ==================================

def examples():
    print( '\nEXAMPLE:\n"""' )
    print( 'python from_xml_to_mask.py -input test_data/master_index.csv -col-wsi filepath_wsi -col-ann filepath_xml -output test_output/from_xml_to_mask/\n"""\n' )


def get_args():
    parser = argparse.ArgumentParser( prog        = 'from_xml_to_mask.py'          ,
                                      description = 'Convert XML annotations into masks' ,
                                      add_help    = False                                     )

    parser.add_argument( '-input' , dest='csv_in' ,
                         help='Input CSV master index' )
    
    parser.add_argument( '-col-wsi' , dest='col_wsi' ,
                         help='Name of the column containing the slide filepaths' )
     
    parser.add_argument( '-col-ann' , dest='col_ann' ,
                         help='Name of the column containing the XML annotation filepaths' )
    
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

    if args.col_wsi is None:
        sys.exit( '\nERROR: column containing the WSI filepaths must be specified!\n' )
    
    if args.col_ann is None:
        sys.exit( '\nERROR: column containing the XML annotation filepaths must be specified!\n' )

    return args



# ==================================
# Map XML annotation to mask
# ==================================

class XmlToMask:
    def __init__( self , file_in , col_wsi , col_ann , path_out ):
        self.file_in  = file_in
        self.col_wsi  = col_wsi
        self.col_ann  = col_ann
        self.path_out = path_out

        self.get_data()
        self.create_output_path()
        self.init_output_df()

    
    def get_data( self ):
        self.df       = pd.read_csv( self.file_in )
        self.list_wsi = self.df[ self.col_wsi ].values
        self.list_xml = self.df[ self.col_ann ].values


    def create_output_path( self ):
        self.path_out_mask = os.path.join( self.path_out , 'annotation_masks' )
        self.path_out_json = os.path.join( self.path_out , 'label_conversion' )
        os.makedirs( self.path_out_mask , exist_ok=True )
        os.makedirs( self.path_out_json , exist_ok=True )


    def init_output_df( self ):
        self.df_out   = pd.DataFrame( columns=[ self.col_wsi                       ,
                                                self.col_ann                       ,
                                                'filepath_mask_annotation'         ,
                                                'filepath_mask_annotation_windows' ,
                                                'filepath_label_conversion'        ,
                                                'mag_mask_annotation'              ] )
        bname         = os.path.splitext( os.path.basename( self.file_in ) )[0] + '_xml_to_mask.csv'
        self.file_out = os.path.join( self.path_out , bname )


    def run( self ):
        for i in range( len( self.list_wsi ) ):
            print( '\nProcessing XML file n.', i,' out of ', len( self.list_wsi ) )
            print( 'Input WSI: ', self.list_wsi[i] )
            print( 'Input XML: ', self.list_xml[i] )
            
            row = self.xml_to_mask( self.list_wsi[i] , self.list_xml[i] )
            
            print( 'Output mask: ', row[-3] )
            print( 'Output label dictionary: ', row[-2] )

            self.df_out.loc[ i ] = row

    
    def xml_to_mask( self , file_wsi , file_xml , mag_ref=1.0 ):
        # Get shape of the WSI thumbnail at base magnification
        mag_base = utils.get_base_magnification( file_wsi )
        shape    = utils.get_image_from_slide( file_wsi , mag=mag_ref ).shape[:2]

        # Get vertices from XML
        vertices , types , labels = utils.get_vertices_from_xml( file_xml           , 
                                                                 mag_xml = mag_base ,
                                                                 mag_out = mag_ref  )

        # For loop on the list of lists of vertex tuples to draw each annotation
        # (polygon/rectangle); if 2 annotations intersect, set the intersection area to 0
        labels_dict = {}
        
        n_ann = len( vertices )
        mask  = np.zeros( shape , dtype=np.uint8 )
        
        for num , annotation in enumerate( vertices ):
            value    = np.int( ( num + 1 )/n_ann * 255.0 )
            check    = Image.new( 'L' , ( shape[1] , shape[0] ) , 0 )
            intersec = np.zeros( shape , dtype=np.uint8 ) 
            
            if types[ num ] == 'polygon':
                ImageDraw.Draw( check ).polygon( annotation , outline=0 , fill=value )
            elif types[ num ] == 'rectangle':
                ImageDraw.Draw( check ).rectangle( annotation , outline=0 , fill=value )

            check = np.array( check )
            
            intersec[ ( mask != 0 ) & ( check == value ) ] = 1
            mask[ ( intersec == 1 ) ]  = 0
            check[ ( intersec == 1 ) ] = 0
            
            fill = ( mask == 0 ) & ( check == value ) 
            mask[ fill ] = check[ fill ]
            
            labels_dict.update( { value: labels[num] } )

        # Save output mask
        file_mask , file_mask_win = self.save_mask( file_wsi , mask )

        # Save output JSON conversion file
        file_json = self.save_label_dict( file_wsi , labels_dict )

        # Create row for output data frame
        row = [ file_wsi , file_xml , file_mask , file_mask_win , file_json , mag_ref ]

        return row


    def save_mask( self , file_wsi , img ):
        bname     = os.path.splitext( os.path.basename( file_wsi ) )[0]
        file_mask = os.path.join( self.path_out_mask , bname + '_annotation_mask.png' )
        file_mask_win = utils.from_linux_to_windows_path( file_mask )
        io.imsave( file_mask , img )
        return file_mask , file_mask_win

    
    def save_label_dict( self , file_wsi , labels_dict ):
        bname     = os.path.splitext( os.path.basename( file_wsi ) )[0]
        file_json = os.path.join( self.path_out_json , bname + '_label_conversion.json' )
        with open( file_json , 'w' ) as fp:
                json.dump( labels_dict , fp )
        return file_json


    def save_master_index( self ):
        df_merge = pd.merge( self.df , self.df_out , on=[ self.col_wsi , self.col_ann ] )
        df_merge.to_csv( self.file_out , index=False )
        print( '\nWritten output data frame to: ', self.file_out )



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
    print( '==     FROM XML ANNOTATIONS TO MASK    ==' )
    print( '==                                     ==' )
    print( '=========================================' )
    print( '=========================================' )
    print( '\n' )

    # Get input arguments
    args = get_args()

    print( '\nInput CSV: ', args.csv_in )
    print( 'Column containing WSI filepaths: ', args.col_wsi )
    print( 'Column containing XML filepaths: ', args.col_ann )
    print( 'Output directory: ', args.path_out )

    # Init class
    transf = XmlToMask( args.csv_in   ,  
                        args.col_wsi  ,
                        args.col_ann  ,      
                        args.path_out )
    print( '\nNumber of XML to process: ', len( transf.list_xml ) )

    # COnvert XMLs to masks
    transf.run()

    # Write output master index
    transf.save_master_index()

    time2 = time.time()
    
    print( '\nTotal elpased time: ', time2 - time1 )
    print( '\n\n' )
