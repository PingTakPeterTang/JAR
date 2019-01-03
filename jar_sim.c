
#include <stdio.h>
#include <math.h>
#include "jar_sim.h"

#define  DEBUG_sim  0

UniJAR LinFP32_2_LogPS80( UniJAR x ) {
/*
Input is in the linear (real) domain and encoded in IEEE FP32. 
The input usually corresponds to either 
(1) parameters already trained in a normal way and we wish to perform 
inference in JAR, or
(2) the linear-domain value in a Kulisch type of accumulator. In our JAR emulation,
this value is encoded in IEEE FP32. 
In either case, the value is of the form (-1)^s * 2^m * (1 + f).
The logarithmic value is  ( (-1)^s,  m + log2(1+f_rnd)_rnd ). In Johnson's
paper, a log-domain value of ((-1)^s, m + g) -- m integer, g fraction -- is encoded in 
Posit: the information (-1)^s is encoded in Posit's sign bit, and  m is treated as an exponent 
and thus encoded using the combination of regime/expoent; g is encoded in the fractional part. 
Thus the number of bits for g varys depending on the value of m. 
In our simulation, we encode this value using IEEE FP32: (-1)^s is encoded in the sign bit,
the value m is biased by 127 and the resulting 8-bit (non-negative) value is encoded in the
exponent field, and g is simply put into the explicit mantissa field.

The conversion LinFP32_2_LogPS80 works as follows:
(1) The input value is first rounded to "b" fractional bits corresponds to 
the log2 lookup table size of 2^b. This is the "b" in "(N, s, a, b, c) log" using Johnson's
notation. So the input value is rounded to (-1)^s * 2^m * (1 + f_rnd). (Note that 
occasionally this "m" can be different from the "m" of the original input by 1 due to
the rounding.)
(2) The "b" fractional bits in f_rnd is used to fetch from a table log2(1+f_rnd), which
is of the form 0.g where g has "c" bits -- the "c" in "(N, s, a, b, c) log". In our
emulation, we simply replace the explicit mantissa bits of the input number with "g"
(3) Finally, the "g" fraction of ((-1)^s, m + g) is rounded according to Posit80 based
on the value of m. This is accomplished by the utility function rnd_2_LogPS80.

In Johnson's software implementation, a special tag IsZero is raised if the linear domain
value is 0. In our emulation, for the sake of speed, we encode zero as 2^(-63) or 0x20000000
if FP32. Any product involving this number will be rounded off completely with any
non-zero within range sum. 
*/

   UniJAR y, z;
   int    i, j, k;
   
   if (DEBUG_sim) printf("....inside LinFP32_2_LogPS80 \n");
   
   /* round the fraction to the number of bits used to fetch log2(1+f) */
   y    = rnd_2_L_frac( x, LOG2_IND_BITS );
   if (DEBUG_sim) show_UniJAR("     input LinFP32 rounded to LOG2_IND_BITS ", y);

   /* get index and fetch from table */
   i = (y.I & FRAC_MASK) >> LOG2_IND_SHIFT; 
   z    = log2_tbl[i];
   if (DEBUG_sim) show_UniJAR("     fraction bits of log2(1+f)             ", z);
   
   /* replace fraction of y by the log2(1+f) */
   y.I &= CLEAR_FRAC; y.I |= z.I;
   if (DEBUG_sim) show_UniJAR("     fraction replaced by log2(1+f)         ", y);



   /* Must round y to a number of fractional bits under Posit (8,0) */
   z = rnd_2_PS80( y );
   if (DEBUG_sim) show_UniJAR("     rounded to PS80 precision              ", z);
   if (DEBUG_sim) printf("----exiting LinFP32_2_LogPS80 \n");

   return z;

}

UniJAR LogPS80_2_LinFP32( UniJAR x ) {
/*
Input is in the logarithmic domain with range and precision 
that accmmodates the sum of two LogPS80 values. Note however
that the fractional bits never exceed 5.
The input is usually the sum of two LogPS80 values; the sum
thus corresponds to the product of two linear domain values.
The logarithmic domain value is of the form ((-1)^s, m + f )
where the sign (-1)^s (s is 0 or 1) is the sign of the linear 
domain value, m is an integer (can be positive or negative) and
f is a fraction, 0 <= f < 1.

The linear domain value is therefore (-1)^s * 2^m * 2^(1+f) 
and in FP32 encoding, s is the sign bit, the biased exponent
is 127+m, represented in 8 bits, and the explicit mantissa bits
g is such that 2^(1+f) = 1 + g.

In our simulation the logarithm value ((-1)^s, m+f) is 
encoded as the IEEE FP32 representation of the value 
(-1)^s * 2^m * (1+f). Hence to get the LinFP32 encoding,
one merely needs to replace the explicit mantissa bits f
with g, which is obtained by table lookup indexed by f.

*/

   UniJAR y, z;
   int    i;
   
   if (DEBUG_sim) printf("....inside LogPS80_2_LinFP32 \n");
   
   /* get index and fetch from table */
   i = (x.I & FRAC_MASK) >> EXP2_IND_SHIFT; 
   z    = exp2_tbl[i];
   if (DEBUG_sim) show_UniJAR("     fraction bits of exp2(1+f)             ", z);
   
   /* replace fraction of x by the exp2(1+f) */
   y.I = x.I & CLEAR_FRAC; y.I |= z.I;
   if (DEBUG_sim) show_UniJAR("     fraction replaced by exp2(1+f)         ", y);

   return y;
}

UniJAR sum2_LogPS80( UniJAR x, UniJAR y ) {
/* 
This function sums two LogPS80 values exactly. The two
logarithmic value ((-1)^s_x , m_x + f_x ) and
((-1)^s_y, m_y + f_y). The sum is simply
(-1)^(s_x+s_y), m_z +  f_z where m_z and f_z are the integer 
and fractions of m_x+f_x + m_y+f_y.

Since the encoding we use for the logarithm value ((-1)^s, m+f) here is the 
IEEE FP32 representation of the value (-1)^s * 2^m * (1+f),
the sum can be computed as follows

1) take the encodings of x and y and perform a simple unsigned integer
addition.
2) At this point, the bottom 23 bits corresponds to the fraction of
f_z and the 8-bit exponent field is 127+127+m_z mod 128,
which is m_z - 2 mod 128. Thus, adding 127+1 to this will yield 127+m_z
3) Replace the msb (sign bit) with s_x + s_y mod 2 (which is s_x XOR s_y
if using logical operation)
*/

   UniJAR sign_z, z;

   sign_z.I = (x.I & SIGN_MASK) + (y.I & SIGN_MASK);
   z.I = x.I + y.I + 0X40800000; /* adding 128+1 to exponent field of int sum */
   z.I = (z.I & CLEAR_SIGN) | sign_z.I;

   return z;
}

UniJAR jar_dotprod( int n, UniJAR x[], UniJAR y[] ) {
/* 
compute n-length dotprod in JAR. In particular, inputs x[], y[] and output are LogPS80 
but accumulation of products are done in linear domain. The additions of LogPS80 quantities
and also accumulation of LinFP32 numbers are exact; but conversion between the two domains
are not necessarily exact.
*/
   UniJAR w, z;
   int    i;

   assert (n >= 0);
   z.I = JAR_ZERO;
   for (i=0; i<n; i++) {
       w = LogPS80_2_LinFP32(
                sum2_LogPS80( x[i], y[i] ) );
       z.F += w.F;
   }
   z = LinFP32_2_LogPS80( z );
   return z;
}





UniJAR exp2_tbl[64] = {
0X00000000,0X00000000,0X00040000,0X00040000,
0X00040000,0X00080000,0X00080000,0X000C0000,
0X000C0000,0X000C0000,0X00100000,0X00100000,
0X00100000,0X00140000,0X00140000,0X00180000,
0X00180000,0X00180000,0X001C0000,0X001C0000,
0X00200000,0X00200000,0X00240000,0X00240000,
0X00240000,0X00280000,0X00280000,0X002C0000,
0X002C0000,0X00300000,0X00300000,0X00340000,
0X00340000,0X00380000,0X00380000,0X003C0000,
0X003C0000,0X00400000,0X00400000,0X00440000,
0X00440000,0X00480000,0X00480000,0X004C0000,
0X00500000,0X00500000,0X00540000,0X00540000,
0X00580000,0X00580000,0X005C0000,0X00600000,
0X00600000,0X00640000,0X00640000,0X00680000,
0X006C0000,0X006C0000,0X00700000,0X00740000,
0X00740000,0X00780000,0X007C0000,0X007C0000
};

UniJAR log2_tbl[32] = {
0X00000000,0X00060000,0X000B0000,0X00110000,
0X00160000,0X001B0000,0X00200000,0X00250000,
0X00290000,0X002E0000,0X00320000,0X00370000,
0X003B0000,0X003F0000,0X00430000,0X00470000,
0X004B0000,0X004F0000,0X00520000,0X00560000,
0X005A0000,0X005D0000,0X00610000,0X00640000,
0X00670000,0X006B0000,0X006E0000,0X00710000,
0X00740000,0X00770000,0X007A0000,0X007D0000
};

