#pragma once

#include <chrono>

#define TIME_EXPR(a)                                                         \
do {                                                                         \
   timer t;                                                                  \
   a;                                                                        \
   std::printf("%s took: %.4e seconds\n\n", (#a), t.wall_time());            \
   } while (0)


struct timer
{
   explicit timer() {
      start = std::chrono::high_resolution_clock::now();
   }

   void restart() {
      start = std::chrono::high_resolution_clock::now();
   }

   double wall_time() {
      return double(std::chrono::duration_cast<std::chrono::nanoseconds>
            (std::chrono::high_resolution_clock::now() - start).count()) / 1.e9;
   }

private:
   std::chrono::system_clock::time_point start;
};


