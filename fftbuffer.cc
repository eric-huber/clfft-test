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

FftBuffer::FftBuffer(Fft& fft)
  : _fft(fft),
    _job(NULL),
    _temp_buf(0),
    _wait{0},
    _in_use(false)
{
    cl_int err = 0;

    // allocate device local memory
    _data_buf = clCreateBuffer(fft.get_context(), CL_MEM_READ_WRITE, size(), NULL, &err);
    CHECK("clCreateBuffer data");

    // allocate temp buffer
    size_t temp_size = fft.get_temp_buffer_size();
    if (0 != temp_size) {
        _temp_buf = clCreateBuffer(fft.get_context(), CL_MEM_READ_WRITE, temp_size, 0, &err);
        CHECK("clCreateBuffer temp");
    }
}

FftBuffer::~FftBuffer() {
    release();
}

void FftBuffer::release() {

    if (NULL != _data_buf) {
        clReleaseMemObject(_data_buf);
        _data_buf = NULL;
    }
    if (NULL != _temp_buf) {
        clReleaseMemObject(_temp_buf);
        _temp_buf = NULL;
    }
}

void FftBuffer::wait() {
    cl_int err = clWaitForEvents(1, &_wait);
    CHECK("clWaitForEvents");
    _in_use = false;
}

void dump_status(cl_int status) {
    switch (status) {
    case CL_COMPLETE:    std::cout << "CL_COMPLETE"    << std::endl; break;
    case CL_SUBMITTED:   std::cout << "CL_SUBMITTED"   << std::endl; break;
    case CL_QUEUED:      std::cout << "CL_QUEUED"      << std::endl; break;
    case CL_RUNNING:     std::cout << "CL_RUNNING"     << std::endl; break;
    }
}

bool FftBuffer::is_finished() {   
    cl_int ret = 0;
    cl_int real_info = 0;
    
    ret = clGetEventInfo(_wait, CL_EVENT_COMMAND_EXECUTION_STATUS,
                        sizeof(cl_int), (void*) &real_info, NULL);
    dump_status(real_info);
    return CL_COMPLETE == real_info;
}

inline size_t FftBuffer::get_fft_size() { 
    return _fft.get_size();
}

inline size_t FftBuffer::size() {
    return _fft.get_size() * sizeof(cl_float);
}