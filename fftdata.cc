#include <iostream>
#include <cstring>

#include "fft.hh"

#define CHECK(MSG)                              \
    if (err != CL_SUCCESS) {                    \
      std::cerr << __FILE__ << ":" << __LINE__  \
          << " Unexpected result for " << MSG   \
          << " (" << err << ")" << std::endl;   \
      return;                                   \
    }

FftData::FftData(Fft& fft)
  : _fft(fft),
    _real(0),
    _imag(0),
    _temp_buf(0),
    _wait_list{0}
{
    cl_int err = 0;

    // allocate local input memory
    _in_buf[REAL] = clCreateBuffer(fft.get_context(), CL_MEM_READ_WRITE,
                                    buffer_size(), NULL, &err);
    CHECK("clCreateBuffer in REAL");

    _in_buf[IMAG] = clCreateBuffer(fft.get_context(), CL_MEM_READ_WRITE,
                                    buffer_size(), NULL, &err);
    CHECK("clCreateBuffer in IMAG");
    
    // allocate local output memory
    _out_buf[REAL] = clCreateBuffer(fft.get_context(), CL_MEM_READ_WRITE,
                                    buffer_size(), NULL, &err);
    CHECK("clCreateBuffer out REAL");

    _out_buf[IMAG] = clCreateBuffer(fft.get_context(), CL_MEM_READ_WRITE,
                                    buffer_size(), NULL, &err);
    CHECK("clCreateBuffer out IMAG");

    // allocate temp buffer
    size_t temp_size = fft.getTempBufferSize();
    if (0 != temp_size) {
        _temp_buf = clCreateBuffer(fft.get_context(), CL_MEM_READ_WRITE, temp_size, 0, &err);
        CHECK("clCreateBuffer temp");
    }
}

FftData::~FftData() {
    if (NULL != _in_buf[REAL])
        clReleaseMemObject(_in_buf[REAL]);
    if (NULL != _in_buf[IMAG])
        clReleaseMemObject(_in_buf[IMAG]);
    if (NULL != _out_buf[REAL])
        clReleaseMemObject(_out_buf[REAL]);
    if (NULL != _out_buf[IMAG])
        clReleaseMemObject(_out_buf[IMAG]);
    if (NULL != _temp_buf)
        clReleaseMemObject(_temp_buf);
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