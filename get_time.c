#include "get_time.h"
#include <stdlib.h>
#include <sys/time.h>

double get_cur_time()
{
   struct timeval time;
   if(gettimeofday(&time, NULL)) return 0;
   return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
