
/* note that unsigned long takes precedence;         */
/* initializing a table such as UniJAR my_tbl[8]     */
/* with literals are interpreted as unit32.          */

#ifndef JAR_TYPE

#define JAR_TYPE
typedef union{
   unsigned long  I;
   float          F;
} UniJAR;

/* Various configuration parameters pertaining to JAR */

#define JAR_ZERO   0x20000000

#define SIGN_MASK  0x80000000
#define FRAC_MASK  0x007FFFFF
#define BEXP_MASK  0x7F800000
#define CLEAR_SIGN 0x7FFFFFFF
#define CLEAR_FRAC 0xFF800000

#define EXP2_IND_BITS    6 
#define EXP2_IND_SHIFT   (23-EXP2_IND_BITS)
#define LOG2_IND_BITS    5
#define LOG2_IND_SHIFT   (23-LOG2_IND_BITS)

#define EXP2_FRAC_BITS   5
#define LOG2_FRAC_BITS   7

#endif



