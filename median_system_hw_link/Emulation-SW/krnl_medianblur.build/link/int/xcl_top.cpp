#include "libspir_types.h"
#include "hls_stream.h"
#include "xcl_top_defines.h"
#include "ap_axi_sdata.h"
#define EXPORT_PIPE_SYMBOLS 1
#include "cpu_pipes.h"
#undef EXPORT_PIPE_SYMBOLS
#include "xcl_half.h"
#include <cstddef>
#include <vector>
#include <complex>
#include <pthread.h>
using namespace std;

extern "C" {

void sobel_accel(size_t img_inp, size_t img_out1, size_t img_out2, unsigned int rows, unsigned int cols);

static pthread_mutex_t __xlnx_cl_sobel_accel_mutex = PTHREAD_MUTEX_INITIALIZER;
void __stub____xlnx_cl_sobel_accel(char **argv) {
  void **args = (void **)argv;
  size_t img_inp = *((size_t*)args[0+1]);
  size_t img_out1 = *((size_t*)args[1+1]);
  size_t img_out2 = *((size_t*)args[2+1]);
  unsigned int rows = *((unsigned int*)args[3+1]);
  unsigned int cols = *((unsigned int*)args[4+1]);
 pthread_mutex_lock(&__xlnx_cl_sobel_accel_mutex);
  sobel_accel(img_inp, img_out1, img_out2, rows, cols);
  pthread_mutex_unlock(&__xlnx_cl_sobel_accel_mutex);
}
void medianblur_accel(size_t img_in, unsigned int rows, unsigned int cols, size_t img_out);

static pthread_mutex_t __xlnx_cl_medianblur_accel_mutex = PTHREAD_MUTEX_INITIALIZER;
void __stub____xlnx_cl_medianblur_accel(char **argv) {
  void **args = (void **)argv;
  size_t img_in = *((size_t*)args[0+1]);
  unsigned int rows = *((unsigned int*)args[1+1]);
  unsigned int cols = *((unsigned int*)args[2+1]);
  size_t img_out = *((size_t*)args[3+1]);
 pthread_mutex_lock(&__xlnx_cl_medianblur_accel_mutex);
  medianblur_accel(img_in, rows, cols, img_out);
  pthread_mutex_unlock(&__xlnx_cl_medianblur_accel_mutex);
}
}
