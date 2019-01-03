#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "jar_sim.h"


#define N_dot  500
#define N_1    20
#define N_2    20
#define VAL_lo  -2.0
#define VAL_hi   2.0


int main(){

   UniJAR  a[N_dot], b[N_dot], x, y, z;
   float   alpha[N_dot], beta[N_dot], width;
   float   z_jar, z_accurate;
   int     i, j, k;

   assert( N_dot >= N_1 );
   assert( N_dot >= N_2 );

   width = (float)VAL_hi - (float)VAL_lo;
   for (i = 0; i < N_dot; i++) {
       alpha[i] = (float)VAL_lo + width * (float)rand()/((float) RAND_MAX);
        beta[i] = (float)VAL_lo + width * (float)rand()/((float) RAND_MAX);

           x.F  = alpha[i];
           a[i] = LinFP32_2_LogPS80( x );
           x.F  = LogPS80_2_Lin_val( a[i] );
       alpha[i] = x.F;

           x.F  = beta[i];
           b[i] = LinFP32_2_LogPS80( x );
           x.F  = LogPS80_2_Lin_val( b[i] );
        beta[i] = x.F;
   }
   /* 
   at this point, we have two random vectors x, y. x and y are in logarithmic domains and
   the corresponding float vectores alpha and beta have identical numerical values
   */

   
   printf("...Test 1 examine the round-trip behavior LogPS80 --> LinFP32 --> LogPS80 \n");
   for (i=0; i<N_1; i++) {
       x = a[i]; 
       y = LinFP32_2_LogPS80(  LogPS80_2_LinFP32( x ) );
       show_UniJAR("   start with LogPS80  ", x );
       show_UniJAR("   after round trip    ", y );
       printf("\n");
   }


   printf("...Test 2 examine the accuracy of LogPS80 --> LinFP32 \n");
   printf("   The error in this conversion is a characteristic of JAR (8,0,5,5,7) Log \n");
   printf("   The conversion is not exact as the exp2 value is from a table of entries with 5 bits \n");
   printf("   We show the accurate version with a FP32 accurate exp2 value \n");
   for (i=0; i<N_2; i++) {
       x = a[i]; 
       y = LogPS80_2_LinFP32( x );
       show_UniJAR("    the LogPS80 content          is ", x);
       printf(     "    LogPS80 --> LinFP32          is %10.6e \n", y.F);
       printf(     "    LogPS80 --> accurate LinFP32 is %10.6e \n", alpha[i] );
       printf("\n");
   }

   printf("...Test 3, we perform an inner product using JAR and compare it with  \n");
   printf("   the inner product of the accurate linear domain value of the input vectors \n");
   x = jar_dotprod( N_dot, a, b );
   z_jar = LogPS80_2_Lin_val( x );
   z_accurate = 0;
   for (i=0; i<N_dot; i++) {
       z_accurate += alpha[i]*beta[i];
   }
   printf("Accurate LinFP32 of the resulting logarithmic domain value in JAR dotprod is %10.6e\n",  z_jar);
   printf("Dot product in FP32 arithmetic                                            is %10.6e\n",  z_accurate);  

   return 0;
}

