#include <iostream>
#include <cstring>

#include "fft.hh"

#define CHECK(MSG)                                          \
    if (err != CL_SUCCESS) {                                \
      std::cerr << __FILE__ << ":" << __LINE__ << " Unexpected result for " << MSG << " (" << err << ")" << std::endl;  \
      return;                                               \
    }

FftData::FftData(Fft& fft)
  : _fft(fft),
    _real(0),
    _imag(0),
    _wait_list{0}
{
    cl_int err = 0;

    // allocate local memory
    _local_buffers[REAL] = clCreateBuffer(fft.get_context(), CL_MEM_READ_WRITE,
                                       buffer_size(), NULL, &err);
    CHECK("clCreateBuffer REAL");

    // Prepare OpenCL memory objects : create buffer for input. 
    _local_buffers[IMAG] = clCreateBuffer(fft.get_context(), CL_MEM_READ_WRITE,
                                       buffer_size(), NULL, &err);
    CHECK("clCreateBuffer IMAG");
}

FftData::~FftData() {
    if (NULL != _local_buffers[REAL])
        clReleaseMemObject(_local_buffers[REAL]);
    if (NULL != _local_buffers[IMAG])
        clReleaseMemObject(_local_buffers[IMAG]);
}

void FftData::get(cl_float* real, cl_float* imag) {
    memcpy(_real, _local_buffers[REAL], buffer_size());
    memcpy(_imag, _local_buffers[IMAG], buffer_size());
}

void FftData::set(cl_float* real, cl_float* imag) {
    _real = real;
    _imag = imag;
}

void FftData::wait() {
    
    cl_int err = clWaitForEvents(2, _wait_list);
    CHECK("clWaitForEvents");
}

inline size_t FftData::get_fft_size() { 
    return _fft.get_size();
}

inline size_t FftData::buffer_size() {
    return _fft.get_size() * sizeof(cl_float);
}