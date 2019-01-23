
/****************************************************************************************
 *  JAR is Johnson ARithmetic. There are two domains in question:
 *  Following Johnson's terminology, "Linear" is the usual real domain,
 *  and "Logarithmic", which is taking the based-2 logarithm of a real
 *  numbers. More specifically,
 *        (real domain) x |--> ( sign(x), log2(|x|) )
 *  For x == 0, a special IsZero flag is tagged along.
 *
 *  Computing an inner product sum_j x_j*y_j for real numbers involves
 *    0) x_j and y_j are already represented in the logarithmic domain
 *    1) adding the logarithmic domain values of x_j and y_j
 *    2) immediately converting the logarithmic sum back to the real domain
 *       which involves taking the exp2 (that is 2**(z) function)
 *    3) summation of the resulting values is done exactly inside an accumulator
 *       which sufficient range and precision to rid of rounding errors
 *       (note however that neither of the conversions back and forth between
 *       real and logarithmic domains are exact)
 *    4) At the end of all summations, the accumulated value, which is in 
 *       the real domain is converted back to logarithmic domain
 *
 *  JAR is emulated here in jar.h and jar.c. These are the key components
 *    1) The type JAR is the main container of the numerical quantities.
 *       UniJAR is a simple union type that overlays an uint32 with FP32.
 *       If Posit(8,0) is used for the the logarithmic domain, FP32 can 
 *       encode the corresponding numerical values without error 
 *       (so is BigFloat16 for that matter). Moreover, accumulation of the 
 *       summands of the products x_j*y_j can be done without error in FP32. 
 *       Overlays of uint32 with FP32 facilitates mainpulation as in JAR, (1) one
 *       often needs to replace the fractional bits during conversion between
 *       the two domains, (2) table lookup based on certains bits of the encoding
 *       are often needed, and (3) our method here in adding two logarithmic values
 *       in the range of Posit (8,0) but encoded using FP32 requries some explicit
 *       manipulation of bit patterns. UniJAR facilitates all these quite well.
 *       The "Uni" in UniJAR can be thought of as "union" of "universal".
 *    2) There are two conversion functions 
 *           1) LinFP32_2_LogPS80: Takes a normal IEEE FP32 number whose value
 *              is (-1)^s 2^m *(1+f) and converts to the form ( s, m + log2(1+f)_rnd )
 *              the number of significant bits in log2(1+f) is rounded according to
 *              the value of m (following Posit (8,0)). Finally, this value is
 *              encoded as the FP32 number (-1)^s * 2^m * (1 + log2(1+f)_rnd).
 *              LinFP32_2_LogPS80 is used to (a) convert a normal FP32 neuralnet model
 *              to be used in JAR and (b) to convert the accumulator value back to the
 *              logarithmic domain.
 *           2) LogPS80_2_LinFP32: The reverse of the LinFP32_2_Log80. This is used 
 *              primarily after sum of two numbers in the logarithmic domain is computed. 
 *              A logarithmic number is of the form (s, m + f). The exp2 value is 
 *              (-1)^s * 2^m * (1 + (2^f - 1)). The fraction 2^f - 1 is obtained by 
 *              a table lookup from the most significant bits of f.
 *    3) add_LogPS80_2_LinFP32: Sums two LogPS80 numbers and immediately converts to LinFP32
 *    4) JAR_dotprod 
 *
 ****************************************************************************************/

#ifndef JAR_SIM

#define JAR_SIM
#include "jar_type.h"
#include "jar_utils.h"

UniJAR LinFP32_2_LogPS80( UniJAR x );
UniJAR LogPS80_2_LinFP32( UniJAR x );
UniJAR sum2_LogPS80( UniJAR x, UniJAR y );
UniJAR jar_dotprod( const int n, const UniJAR* x, const UniJAR* y );
void jar_matvecmul( const int M, const int K, const UniJAR* A, const UniJAR* b, UniJAR* c );
void jar_matmul( const int M, const int N, const int K, const UniJAR* A, const UniJAR* B, UniJAR* C );

extern UniJAR exp2_tbl[64];
extern UniJAR log2_tbl[32];

#if defined(__AVX512F__)
#include <immintrin.h>
__m512i jar_fma_avx512( const __m512i a, const __m512i b, const __m512i c );
#endif

#endif


