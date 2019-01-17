#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "jar_sim.h"

#define VAL_lo  -2.0
#define VAL_hi   2.0

float float_dotprod( const int K, float* a, float* b ) {
  float res = 0.0f;
  int i;

  for ( i=0; i<K; ++i ) {
    res += a[i] * b[i];
  }

  return res;
}

void init_float( float* f, const int size, const float val_lo, const float width ) {
  int i;

  for ( i=0; i<size; ++i ) {
    f[i] = (float)val_lo + width * (float)rand()/((float) RAND_MAX);
  }
}

void init_JAR_update_float( UniJAR* j, float* f, const int size ) {
  int i;
  UniJAR x;

  /* 
   at this point, we have two random vectors j, f. j is in logarithmic domains and
   the corresponding float vector f has identical numerical values
   */
  for ( i=0; i<size; ++i ) {
    x.F  = f[i];
    j[i] = LinFP32_2_LogPS80( x );
    x.F  = LogPS80_2_Lin_val( j[i] );
    f[i] = x.F;
  }
} 

void test_rt( const int size ) {
  UniJAR* a = (UniJAR*) malloc( size*sizeof(UniJAR) );
  float* f_a = (float*) malloc( size*sizeof(float) );
  UniJAR x, y;
  float width = (float)VAL_hi - (float)VAL_lo;
  int i;
 
  printf("Test: examine the round-trip behavior LogPS80 --> LinFP32 --> LogPS80 \n");
  init_float( f_a, size, (float)VAL_lo, width );

  init_JAR_update_float( a, f_a, size );
  
  for (i=0; i<size; i++) {
    x = a[i]; 
    y = LinFP32_2_LogPS80(  LogPS80_2_LinFP32( x ) );
    show_UniJAR("   start with LogPS80  ", x );
    show_UniJAR("   after round trip    ", y );
    printf("\n");
  }

  free( f_a );
  free( a );
}

void test_LogPS80_to_LinFP32( const int size ) {
  UniJAR* a = (UniJAR*) malloc( size*sizeof(UniJAR) );
  float* f_a = (float*) malloc( size*sizeof(float) );
  UniJAR x, y;
  float width = (float)VAL_hi - (float)VAL_lo;
  int i;

  printf("Test: examine the accuracy of LogPS80 --> LinFP32 \n");
  printf("   The error in this conversion is a characteristic of JAR (8,0,5,5,7) Log \n");
  printf("   The conversion is not exact as the exp2 value is from a table of entries with 5 bits \n");
  printf("   We show the accurate version with a FP32 accurate exp2 value \n");

  init_float( f_a, size, (float)VAL_lo, width );

  init_JAR_update_float( a, f_a, size );
  
  for (i=0; i<size; i++) {
    x = a[i]; 
    y = LogPS80_2_LinFP32( x );
    show_UniJAR("    the LogPS80 content          is ", x);
    printf(     "    LogPS80 --> LinFP32          is %10.6e \n", y.F);
    printf(     "    LogPS80 --> accurate LinFP32 is %10.6e \n", f_a[i] );
    printf("\n");
  }

  free( f_a );
  free( a );
}

void test_dotprod( const int size ) {
  UniJAR* a = (UniJAR*) malloc( size*sizeof(UniJAR) );
  UniJAR* b = (UniJAR*) malloc( size*sizeof(UniJAR) );
  UniJAR c;
  float* f_a = (float*) malloc( size*sizeof(float) );
  float* f_b = (float*) malloc( size*sizeof(float) );
  float f_c, jar_c;
  float width = (float)VAL_hi - (float)VAL_lo;

  printf("Test: we perform an inner product using JAR and compare it with  \n");
  printf("   the inner product of the accurate linear domain value of the input vectors \n");

  init_float( f_a, size, (float)VAL_lo, width );
  init_float( f_b, size, (float)VAL_lo, width );

  init_JAR_update_float( a, f_a, size );
  init_JAR_update_float( b, f_b, size );
  
  /* running JAR dotproduct */
  jar_c = LogPS80_2_Lin_val( jar_dotprod( size, a, b ) );

  /* running fp32 dotproduct */
  f_c = float_dotprod( size, f_a, f_b );

  printf("Accurate LinFP32 of the resulting logarithmic domain value in JAR dotprod is %10.6e\n",  jar_c);
  printf("Dot product in FP32 arithmetic                                            is %10.6e\n",  f_c);

  free( f_b );
  free( f_a );
  free( b );
  free( a );
}

void test_matvecmul( const int M, const int K ) {

}

void test_matmul( const int M, const int N, const int K ) {

}

void print_help() {
  printf("\n");
  printf("This tester can run multiple tests, which one is determined by the first integer arugments\n");
  printf("  0 : examine the round-trip behavior LogPS80 --> LinFP32 --> LogPS80\n");
  printf("  1 : examine the accuracy of LogPS80 --> LinFP32\n");
  printf("  2 : inner product using LogPS80\n");
  printf("  3 : matrix vector multiplication using LogPS80\n");
  printf("  4 : matrix matrix multiplication using LogPS80\n");
  printf("\n");
  printf("each of them require additional integer paramters:\n");
  printf("  0,1,2 : one additional integer specifying N (length of array to test)\n");
  printf("  3     : two additional integers specifying M, K\n");
  printf("  4     : three additional integers specifying M, N, K\n");
  printf("\n");
  printf("Examples:\n");
  printf("   ./demo 0 20\n");
  printf("   ./demo 1 30\n");
  printf("   ./demo 2 50\n");
  printf("   ./demo 3 16 50\n");
  printf("   ./demo 4 16 24 50\n");
  printf("\n");
}

int main(int argc, char* argv[]) {
  if ( argc == 3 ) {
    int test = atoi(argv[1]);
    int size = atoi(argv[2]);

    if ( test == 0 ) {
      test_rt( size );
    } else if ( test == 1 ) {
      test_LogPS80_to_LinFP32( size );
    } else if ( test == 2 ) {
      test_dotprod( size );
    } else {
      print_help();
    }
  } else if ( argc == 4 ) {
    int test = atoi(argv[1]);
    int M    = atoi(argv[2]);
    int K    = atoi(argv[3]);

    if ( test == 3 ) {
      test_matvecmul( M, K );
    } else {
      print_help();
    }
  } else if ( argc == 5 ) {
    int test = atoi(argv[1]);
    int M    = atoi(argv[2]);
    int N    = atoi(argv[3]);
    int K    = atoi(argv[4]);

    if ( test == 4 ) {
      test_matmul( M, N, K );
    } else {
      print_help();
    }

  } else {
    print_help();
  } 
  
  return 0;
}

