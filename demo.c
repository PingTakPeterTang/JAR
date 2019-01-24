#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include <sys/time.h>

#include "jar_sim.h"

#define VAL_lo  -2.0
#define VAL_hi   2.0

inline double time_in_sec(struct timeval start, struct timeval end) {
  return ((double)(((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)))) / 1.0e6;
}

float float_dotprod( const int K, float* a, float* b ) {
  float res = 0.0f;
  int i;

  for ( i=0; i<K; ++i ) {
    res += a[i] * b[i];
  }

  return res;
}

void float_matvecmul( const int M, const int K, float* A, float* b, float* c ) {
  int m, k;

  for ( m=0 ; m<M; ++m ) {
    c[m] = 0.0f;
  }

  for ( k=0; k<K; ++k ) {
    for ( m=0; m<M; ++m ) {
      c[m] += A[(k*M)+m] * b[k];
    }
  }
}

void float_matmul( const int M, const int N, const int K, float* A, float* B, float* C ) {
  int m, n, k;

  for ( m=0 ; m<M*N; ++m ) {
    C[m] = 0.0f;
  }

  for ( k=0; k<K; ++k ) {
    for ( n=0; n<N; ++n ) {
      for ( m=0; m<M; ++m ) {
        C[(n*M)+m] += A[(k*M)+m] * B[(n*K)+k];
      }
    }
  }   
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

void compute_norms( const int size, UniJAR* j, float* f, float* l1_jar, float* l1_f, float* lmax ) {
  int m;

  *l1_jar = 0.0f;
  *l1_f = 0.0f;
  *lmax = 0.0f;

  for ( m=0 ; m<size; ++m ) {
    float j_f = LogPS80_2_Lin_val( j[m] );
    *l1_jar += j_f;
    *l1_f   += f[m];
    *lmax = ( *lmax > fabs( j_f - f[m] ) ) ? *lmax : fabs( j_f - f[m] );
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
  UniJAR* A = (UniJAR*) malloc( M*K*sizeof(UniJAR) );
  UniJAR* b = (UniJAR*) malloc( K*sizeof(UniJAR) );
  UniJAR* c1 = (UniJAR*) malloc( M*sizeof(UniJAR) );
  UniJAR* c2 = (UniJAR*) malloc( M*sizeof(UniJAR) );
  float* f_A = (float*) malloc( M*K*sizeof(float) );
  float* f_b = (float*) malloc( K*sizeof(float) );
  float* f_c = (float*) malloc( M*sizeof(float) );
  float width = (float)VAL_hi - (float)VAL_lo;
  float lmax = 0.0f;
  float l1_f = 0.0f;
  float l1_jar = 0.0f; 

  printf("Test: we perform a matrix vector product using JAR and compare it with  \n");
  printf("   the matrix vector product of the accurate linear domain value of the input data \n");

  init_float( f_A, M*K, (float)VAL_lo, width );
  init_float( f_b, K, (float)VAL_lo, width );
  init_float( f_c, M, (float)VAL_lo, width );

  init_JAR_update_float( A, f_A, M*K );
  init_JAR_update_float( b, f_b, K );
  init_JAR_update_float( c1, f_c, M );
  init_JAR_update_float( c2, f_c, M );
  
  /* running JAR matvecmul */
  jar_matvecmul( M, K, A, b, c1 );
  jar_matvecmul_avx512( M, K, A, b, c2 );

  /* running fp32 matvecmul */
  float_matvecmul( M, K, f_A, f_b, f_c );

  /* computing norms */
  compute_norms( M, c1, f_c, &l1_jar, &l1_f, &lmax );
  printf("scalar code\n"); 
  printf("Accurate LinFP32 of the resulting logarithmic domain 1-norm in JAR vecmatmul is %10.6e\n", l1_jar);
  printf("matvecmul in FP32 arithmetic 1-norm                                          is %10.6e\n", l1_f);
  printf("Max norm of error                                                            is %10.6e\n", lmax);

  compute_norms( M, c2, f_c, &l1_jar, &l1_f, &lmax ); 
  printf("vector code\n");
  printf("Accurate LinFP32 of the resulting logarithmic domain 1-norm in JAR vecmatmul is %10.6e\n", l1_jar);
  printf("matvecmul in FP32 arithmetic 1-norm                                          is %10.6e\n", l1_f);
  printf("Max norm of error                                                            is %10.6e\n", lmax);

  free( f_c );
  free( f_b );
  free( f_A );
  free( c1 );
  free( c2 );
  free( b );
  free( A );
}

void test_matmul( const int M, const int N, const int K ) {
  UniJAR* A = (UniJAR*) malloc( M*K*sizeof(UniJAR) );
  UniJAR* B = (UniJAR*) malloc( K*N*sizeof(UniJAR) );
  UniJAR* C1 = (UniJAR*) malloc( M*N*sizeof(UniJAR) );
  UniJAR* C2 = (UniJAR*) malloc( M*N*sizeof(UniJAR) );
  float* f_A = (float*) malloc( M*K*sizeof(float) );
  float* f_B = (float*) malloc( K*N*sizeof(float) );
  float* f_C = (float*) malloc( M*N*sizeof(float) );
  float width = (float)VAL_hi - (float)VAL_lo;
  float lmax = 0.0f;
  float l1_f = 0.0f;
  float l1_jar = 0.0f;
  int i, reps;
  struct timeval start;
  struct timeval stop;
  double time;
  double flops = 2.0*(double)M*(double)N*(double)K;

  printf("Test: we perform a matrix matrix product using JAR and compare it with  \n");
  printf("   the matrix matrix product of the accurate linear domain value of the input data \n");

  init_float( f_A, M*K, (float)VAL_lo, width );
  init_float( f_B, K*N, (float)VAL_lo, width );
  init_float( f_C, M*N, (float)VAL_lo, width );

  init_JAR_update_float( A, f_A, M*K );
  init_JAR_update_float( B, f_B, K*N );
  init_JAR_update_float( C1, f_C, M*N );
  init_JAR_update_float( C2, f_C, M*N );
  
  /* running JAR matmul */
  jar_matmul( M, N, K, A, B, C1 );
  jar_matmul_avx512( M, N, K, A, B, C2 );

  /* running fp32 matmul */
  float_matmul( M, N, K, f_A, f_B, f_C );

  /* computing norms */
  compute_norms( M*N, C1, f_C, &l1_jar, &l1_f, &lmax ); 
  printf("scalar code\n");
  printf("Accurate LinFP32 of the resulting logarithmic domain 1-norm in JAR matmul is %10.6e\n", l1_jar);
  printf("matmul in FP32 arithmetic 1-norm                                          is %10.6e\n", l1_f);
  printf("Max norm of error                                                         is %10.6e\n", lmax);

  compute_norms( M*N, C2, f_C, &l1_jar, &l1_f, &lmax ); 
  printf("vector code\n");
  printf("Accurate LinFP32 of the resulting logarithmic domain 1-norm in JAR matmul is %10.6e\n", l1_jar);
  printf("matmul in FP32 arithmetic 1-norm                                          is %10.6e\n", l1_f);
  printf("Max norm of error                                                         is %10.6e\n", lmax);

  /* let's do some performance test */
  reps = 10000;
  gettimeofday(&start, NULL);
  for ( i = 0; i < reps; ++i ) {
    jar_matmul_avx512( M, N, K, A, B, C2 );
  }
  gettimeofday(&stop, NULL);
  time = time_in_sec( start, stop )/(double)reps;
  printf("time for GEMM M=%i, N=%i, K=%i is %f seconds, GFLOPS=%f\n", M, N, K, time, (flops/time)/1.0e9);
  
  free( f_C );
  free( f_B );
  free( f_A );
  free( C1 );
  free( C2 );
  free( B );
  free( A );
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

