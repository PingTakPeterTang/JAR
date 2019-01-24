/******************************************************************************
** Copyright (c) 2019, Intel Corporation                                     **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
/* Ping Tak Peter Tang, Alexander Heinecke (Intel Corp.)
******************************************************************************/

/* note that unsigned long takes precedence;         */
/* initializing a table such as UniJAR my_tbl[8]     */
/* with literals are interpreted as unit32.          */

#ifndef JAR_TYPE

#define JAR_TYPE
typedef union{
   unsigned int   I;
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



