
#include <assert.h>
#include <stdio.h>
#include <math.h>

#include "jar_type.h"
#include "jar_utils.h"

#define DEBUG_utils 0

void show_UniJAR( char description[], UniJAR x ) {
   printf("%s (I,F) = (%08X,%10.6e) \n", description, x.I, x.F);
   return;
}

void show_val_LogPS80( char description[], UniJAR x ) {
/* 
A LogPS80 number is of the form ((-1)^s, m+f) but encoded
here as IEEE FP32 representation of the value (-1)^s * 2^m * (1+f).
This function shows the numeric values of (-1)^s and  m+f
*/
   int    m;
   UniJAR sign, f;

   sign.I = (x.I & SIGN_MASK) | 0X3F800000;
   f.I    = (x.I & FRAC_MASK) | 0x3F800000;
   f.F   -= 1.0;
   
   m      = ((x.I & BEXP_MASK) >> 23) - 127;
   printf("%s  sign is %f, m is %d, f is %f \n", description, sign.F, m, f.F);
}

void show_val_LinFP32( char description[], UniJAR x ) {
/* 
A LinFP32 number is a linear domain number encoded in IEEE FP32.
So in fact the value is the numerical val of the bit string of x
interpreted as an IEEE encoding. This function is here just for
the sake of symmetry. To add a little bit of value, we show the
bit string in hex.
*/

   printf("%s  value is %10.6e and hex is %08X \n", description, x.F, x.I);
}




UniJAR two_2_k( int k ) {
   UniJAR x;
   assert( k <= 50 & k >= -50 );
   if (k >= 0) {
      x.F = 1.0; x.I += (k << 23); 
   }
   else {
      x.F = 1.0; x.I -= ((-k) << 23);
   }
 
   return x;
}


UniJAR rnd_2_L_frac( UniJAR x, int L ) {
   UniJAR Big, y;
   
   assert (L >= 0 & L <= 10);
   y = x; y.I &= CLEAR_FRAC;
   Big = two_2_k( 23-L );

   if (DEBUG_utils) {
      show_UniJAR("..input ---> ",x);   printf("\n"); 
      show_UniJAR("..BIG   ---> ",Big); printf("\n"); 
   }

   Big.F *= y.F;
   x.F += Big.F;
   x.F -= Big.F;
   return x;
}

UniJAR rnd_2_PS80( UniJAR x ) {
   UniJAR Big, sign_x, y;
   unsigned long opMask;
   int    expo, ind;

   sign_x.I = x.I & SIGN_MASK;
   ind = (x.I & BEXP_MASK) >> 23;
   expo = ind - 127;
   if ( expo <= -7 || expo >= 6 ) {
      opMask = 0x00000000;
   }
   else {
      opMask = 0x7FFFFFFF;
   }
   /* (Big + (x & opMask)) - (Big & opMask) */
   Big = Big_tbl[ ind ];
   
   if (DEBUG_utils) {
      printf("....inside rnd_2_PS80 \n"); 
      show_UniJAR("      input  ",x);
      show_UniJAR("      Big    ",Big);
   }
   
   y.I = x.I & opMask;
   y.F += Big.F;
   if (DEBUG_utils) show_UniJAR("    x+Big    ",y); 
   Big.I &= opMask;
   y.F -= Big.F;
   if (DEBUG_utils) show_UniJAR("  undo Big   ",y); 
   y.I |= sign_x.I;
   return y;
}

float LogPS80_2_Lin_val( UniJAR x ) {
/*
Compute the accurate exp2 value of a LogPS80 input number
*/
   UniJAR y;
   float a, b;

   /* input logarithmic value is ( (-1)^s, m + f ) */
   y.I = (x.I & FRAC_MASK) | 0X3F800000;    
   /* y.F is now 1 + f */
   a = y.F - 1.0; y.F = exp2(a); 
   x.I &= CLEAR_FRAC; y.I &= FRAC_MASK;
   x.I |= y.I;
   b = x.F;
   return b;
}
   
  

void gen_exp2_tbl( ) {
/* Generates and print table */
   int num_entries, num_entries_per_line, num_lines;
   int i, j;
   UniJAR x, y, z, delta;
 
 
   assert( EXP2_IND_BITS >= 3 );

   num_entries = (int) pow(2.0, EXP2_IND_BITS);
   num_entries_per_line = 4;
   num_lines = num_entries / num_entries_per_line;

   delta = two_2_k( -EXP2_IND_BITS );

   printf("UniJAR exp2_tbl[%d] = {\n", num_entries);

   for ( i=0; i<num_lines; i++ ){
       for ( j=0; j<num_entries_per_line; j++ ){
           x.F = (float)(i*num_entries_per_line + j) * delta.F;
           y.F = exp2( x.F );
           y   = rnd_2_L_frac( y, EXP2_FRAC_BITS );
           /* For a logarithmic value of m + f, we compute the fractional bits
              of 2^f  (which is 2^f - 1 but in fixed-pt rep)                */
           y.I &= FRAC_MASK;
           if (j < num_entries_per_line-1) {
               printf("0X%08X,", y.I);
           }
           else {
               if (i < num_lines-1) {
                   printf("0X%08X,\n",y.I);
              }
              else {
                   printf("0X%08X\n",y.I);
              }
           }
       }
   } 
   printf("}\n");
}


void gen_log2_tbl( ) {
/* Generates and print table */
   int num_entries, num_entries_per_line, num_lines;
   int i, j;
   UniJAR one, x, y, z, delta;
 
 
   assert( LOG2_IND_BITS >= 3 );

   num_entries = (int) pow(2.0, LOG2_IND_BITS);
   num_entries_per_line = 4;
   num_lines = num_entries / num_entries_per_line;

   delta = two_2_k( -LOG2_IND_BITS );
   one.F = 1.0;

   printf("UniJAR log2_tbl[%d] = {\n", num_entries);

   for ( i=0; i<num_lines; i++ ){
       for ( j=0; j<num_entries_per_line; j++ ){
           x.F = one.F + (float)(i*num_entries_per_line + j) * delta.F;
           y.F = one.F + log2( x.F );
           y   = rnd_2_L_frac( y, LOG2_FRAC_BITS );
           y.I &= FRAC_MASK;
           if (j < num_entries_per_line-1) {
               printf("0X%08X,", y.I);
           }
           else {
               if (i < num_lines-1) {
                   printf("0X%08X,\n",y.I);
              }
              else {
                   printf("0X%08X\n",y.I);
              }
           }
       }
   } 
   printf("}\n");
}
   
void gen_Big_tbl( ) {
/* Generates and print table of the "Big" values used in variable precision rounding */
/* This version uses the entire 8-bit biased exponent field of a FP32 as index       */

   int num_entries, num_entries_per_line, num_lines;
   int biased_expo, expo, m, i, j, n_frac;
   UniJAR one, x, y, z, delta;
 
   num_entries = (int) pow(2.0, 8);
   num_entries_per_line = 4;
   num_lines = num_entries / num_entries_per_line;

   printf("UniJAR Big_tbl[%d] = {\n", num_entries);

   for ( i=0; i<num_lines; i++ ){
       for ( j=0; j<num_entries_per_line; j++ ){
           biased_expo = i*num_entries_per_line + j;
           expo = biased_expo - 127;
           if (expo >= 6) {
              /* saturate to largest number: Big = (m,0) m=6 */ 
              y = two_2_k( 6 );
           }
           if (0 <= expo && expo <= 5) {
              /* Need to round to 5 - expo fractional bits */
              y = two_2_k( (23+expo)-(5-expo) );
           }
           if (-6 <= expo && expo <= -1) {
              /* Need to round to 6 + expo fractional bits */
              y = two_2_k( (23+expo)-(6+expo) );
           }
           if (-28 <= expo && expo <= -7) {
              /* Saturates to non-zero number closest to 0 */
              y = two_2_k( -6 );
           }
           if (expo <= -29) {
              /* Our emulation considers this to be the logarithm value of 0 */
              /* Instead of using a special IsZero flag, we use a very small number */
              /* The idea is that accumulation with this number will be rounded off completely */
              y.I = 0X20000000;
           }
           /* now prints out y */
           if (j < num_entries_per_line-1) {
               printf("0X%08X,", y.I);
           }
           else {
               if (i < num_lines-1) {
                   printf("0X%08X,\n",y.I);
              }
              else {
                   printf("0X%08X\n",y.I);
              }
           }
       }
   } 
   printf("}\n");
}

void gen_Mask_tbl( ) {
/* Generates and print table of the "Mask" values used in variable precision rounding */
/* The formula is basically ( Big +  x & Mask ) - (Mask & Big)                        */
/* Normally Mask = 0x7FFFFFFF; but for x that are out of range, Mask is 0x00000000    */
/* making the formula a simple replacement of x by Big                                */

   int num_entries, num_entries_per_line, num_lines;
   int biased_expo, expo, m, i, j, n_frac;
   UniJAR one, x, y, z, delta;
 
   num_entries = (int) pow(2.0, 8);
   num_entries_per_line = 4;
   num_lines = num_entries / num_entries_per_line;

   printf("UniJAR Mask_tbl[%d] = {\n", num_entries);

   for ( i=0; i<num_lines; i++ ){
       for ( j=0; j<num_entries_per_line; j++ ){
           biased_expo = i*num_entries_per_line + j;
           expo = biased_expo - 127;
           if (expo <= -7 || expo >= 6) {
              y.I = 0x00000000;
           }
           else {
              y.I = 0x7FFFFFFF;
           }
           /* now prints out y */
           if (j < num_entries_per_line-1) {
               printf("0X%08X,", y.I);
           }
           else {
               if (i < num_lines-1) {
                   printf("0X%08X,\n",y.I);
              }
              else {
                   printf("0X%08X\n",y.I);
              }
           }
       }
   } 
   printf("}\n");
}

UniJAR Big_tbl[256] = {
0X20000000,0X20000000,0X20000000,0X20000000,
0X20000000,0X20000000,0X20000000,0X20000000,
0X20000000,0X20000000,0X20000000,0X20000000,
0X20000000,0X20000000,0X20000000,0X20000000,
0X20000000,0X20000000,0X20000000,0X20000000,
0X20000000,0X20000000,0X20000000,0X20000000,
0X20000000,0X20000000,0X20000000,0X20000000,
0X20000000,0X20000000,0X20000000,0X20000000,
0X20000000,0X20000000,0X20000000,0X20000000,
0X20000000,0X20000000,0X20000000,0X20000000,
0X20000000,0X20000000,0X20000000,0X20000000,
0X20000000,0X20000000,0X20000000,0X20000000,
0X20000000,0X20000000,0X20000000,0X20000000,
0X20000000,0X20000000,0X20000000,0X20000000,
0X20000000,0X20000000,0X20000000,0X20000000,
0X20000000,0X20000000,0X20000000,0X20000000,
0X20000000,0X20000000,0X20000000,0X20000000,
0X20000000,0X20000000,0X20000000,0X20000000,
0X20000000,0X20000000,0X20000000,0X20000000,
0X20000000,0X20000000,0X20000000,0X20000000,
0X20000000,0X20000000,0X20000000,0X20000000,
0X20000000,0X20000000,0X20000000,0X20000000,
0X20000000,0X20000000,0X20000000,0X20000000,
0X20000000,0X20000000,0X20000000,0X20000000,
0X20000000,0X20000000,0X20000000,0X3C800000,
0X3C800000,0X3C800000,0X3C800000,0X3C800000,
0X3C800000,0X3C800000,0X3C800000,0X3C800000,
0X3C800000,0X3C800000,0X3C800000,0X3C800000,
0X3C800000,0X3C800000,0X3C800000,0X3C800000,
0X3C800000,0X3C800000,0X3C800000,0X3C800000,
0X3C800000,0X48000000,0X48000000,0X48000000,
0X48000000,0X48000000,0X48000000,0X48800000,
0X49800000,0X4A800000,0X4B800000,0X4C800000,
0X4D800000,0X42800000,0X42800000,0X42800000,
0X42800000,0X42800000,0X42800000,0X42800000,
0X42800000,0X42800000,0X42800000,0X42800000,
0X42800000,0X42800000,0X42800000,0X42800000,
0X42800000,0X42800000,0X42800000,0X42800000,
0X42800000,0X42800000,0X42800000,0X42800000,
0X42800000,0X42800000,0X42800000,0X42800000,
0X42800000,0X42800000,0X42800000,0X42800000,
0X42800000,0X42800000,0X42800000,0X42800000,
0X42800000,0X42800000,0X42800000,0X42800000,
0X42800000,0X42800000,0X42800000,0X42800000,
0X42800000,0X42800000,0X42800000,0X42800000,
0X42800000,0X42800000,0X42800000,0X42800000,
0X42800000,0X42800000,0X42800000,0X42800000,
0X42800000,0X42800000,0X42800000,0X42800000,
0X42800000,0X42800000,0X42800000,0X42800000,
0X42800000,0X42800000,0X42800000,0X42800000,
0X42800000,0X42800000,0X42800000,0X42800000,
0X42800000,0X42800000,0X42800000,0X42800000,
0X42800000,0X42800000,0X42800000,0X42800000,
0X42800000,0X42800000,0X42800000,0X42800000,
0X42800000,0X42800000,0X42800000,0X42800000,
0X42800000,0X42800000,0X42800000,0X42800000,
0X42800000,0X42800000,0X42800000,0X42800000,
0X42800000,0X42800000,0X42800000,0X42800000,
0X42800000,0X42800000,0X42800000,0X42800000,
0X42800000,0X42800000,0X42800000,0X42800000,
0X42800000,0X42800000,0X42800000,0X42800000,
0X42800000,0X42800000,0X42800000,0X42800000,
0X42800000,0X42800000,0X42800000,0X42800000,
0X42800000,0X42800000,0X42800000,0X42800000
};

