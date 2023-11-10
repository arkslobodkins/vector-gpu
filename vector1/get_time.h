#pragma once

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TIME(a)                                                      \
do {                                                                 \
   double start = get_cur_time();                                    \
   a;                                                                \
   printf("%s took: %.4e seconds\n", #a, get_cur_time() - start);    \
   } while (0)

double get_cur_time();

#ifdef __cplusplus
}
#endif

