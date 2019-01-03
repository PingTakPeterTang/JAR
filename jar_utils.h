
#include <assert.h>
#include "jar_type.h"

#ifndef JAR_UTILS

#define JAR_UTILS

void show_UniJAR( char description[], UniJAR x );
void show_val_LogPS80( char description[], UniJAR x );
void show_val_LinFP32( char description[], UniJAR x );

UniJAR two_2_k( int k );
UniJAR rnd_2_L_frac( UniJAR x, int L );
UniJAR rnd_2_PS80( UniJAR x );
float  LogPS80_2_Lin_val( UniJAR x );


extern UniJAR Big_tbl[256]; 

void gen_exp2_tbl( );
void gen_log2_tbl( );
void gen_Big_tbl();
void gen_Mask_tbl();


#endif



