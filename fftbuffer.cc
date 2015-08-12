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

FftBuffer::~FftBuffer() {
    release();
}

void FftBuffer::release() {

    if (NULL != _in_buf[REAL]) {
        clReleaseMemObject(_in_buf[REAL]);
        _in_buf[REAL] = NULL;
    }
    if (NULL != _in_buf[IMAG]) {
        clReleaseMemObject(_in_buf[IMAG]);
        _in_buf[IMAG] = NULL;
    }
    if (NULL != _out_buf[REAL]) {
        clReleaseMemObject(_out_buf[REAL]);
        _out_buf[REAL] = NULL;
    }
    if (NULL != _out_buf[IMAG]) {
        clReleaseMemObject(_out_buf[IMAG]);
        _out_buf[IMAG] = NULL;
    }
    if (NULL != _temp_buf) {
        clReleaseMemObject(_temp_buf);
        _temp_buf = NULL;
    }
}

void FftBuffer::wait() {
    cl_int err = clWaitForEvents(2, _wait_list);
    CHECK("clWaitForEvents");
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
    cl_int imag_info = 0;
    
    ret = clGetEventInfo(_wait_list[REAL], CL_EVENT_COMMAND_EXECUTION_STATUS,
                        sizeof(cl_int), (void*) &real_info, NULL);
    ret = clGetEventInfo(_wait_list[IMAG], CL_EVENT_COMMAND_EXECUTION_STATUS,
                        sizeof(cl_int), (void*) &imag_info, NULL);
    dump_status(real_info);
    dump_status(imag_info);
    return CL_COMPLETE == real_info && CL_COMPLETE == imag_info;
}

inline size_t FftBuffer::get_fft_size() { 
    return _fft.get_size();
}

inline size_t FftBuffer::buffer_size() {
    return _fft.get_size() * sizeof(cl_float);
}