#include <iostream>
#include <cstring>

#include "fft.hh"

#define CHECK(MSG)                              \
    if (err != CL_SUCCESS) {                    \
      std::cerr << __FILE__ << ":" << __LINE__  \
          << " Unexpected result for " << MSG   \
          << " (" << err << ")" << std::endl;   \
      return false;                             \
    }


Fft::Fft(size_t fft_size)
  : _fft_size(fft_size)
{
}

bool Fft::init() {

    if (setupCl() && setupClFft())
        return true;
    return false;
}

void Fft::shutdown() {
    
    // Release clFFT library. 
    clfftTeardown();
    
    // Release OpenCL working objects. 
    clReleaseCommandQueue(_queue);
    clReleaseContext(_context);
}

bool Fft::add(FftData& data) {
    cl_int err = 0;
   
    cl_event writes[2]  = {0};
    cl_event reads[2] = {0};
    cl_event transform = 0;

    // Enqueue write tab array into _local_buffers[0]. 
    err = clEnqueueWriteBuffer(_queue, data.in_real(), CL_TRUE, 0, 
                                data.buffer_size(), data._real, 0, NULL, &writes[0]);
    CHECK("clEnqueueWriteBuffer real");
    
    err = clEnqueueWriteBuffer(_queue, data.in_imag(), CL_TRUE, 0,
                                data.buffer_size(), data._imag, 0, NULL, &writes[1]);
    CHECK("clEnqueueWriteBuffer imag");

    // Enqueue the FFT
    err = clfftEnqueueTransform(_planHandle, CLFFT_FORWARD, 1, &_queue, 2, writes, &transform,
                                 data.in_buffers(), data.out_buffers(), data.temp_buffer());
    CHECK("clEnqueueTransform");
    
    // Copy result to input array
    err = clEnqueueReadBuffer(_queue, data.out_real(), CL_TRUE, 0,
                               data.buffer_size(), data._real, 1, &transform, &reads[0]);
    CHECK("clEnqueueReadBuffer real");
    
    err = clEnqueueReadBuffer(_queue, data.out_imag(), CL_TRUE, 0, 
                               data.buffer_size(), data._imag, 1, &transform, &reads[1]);
    CHECK("clEnqueueReadBuffer imag");

    data.set_wait(reads);

    return true;
}

size_t Fft::getTempBufferSize() {
    size_t size = 0;
    int status = clfftGetTmpBufSize(_planHandle, &size);
    return 0 == status ? size : 0;
}

bool Fft::setupCl() {
    cl_int err = 0;

    // Setup platform 
    err = clGetPlatformIDs(1, &_platform, NULL);
    CHECK("clGetPlatformIds");

    // Setup devices
    err = clGetDeviceIDs(_platform, CL_DEVICE_TYPE_GPU, 1, &_device, NULL);
    CHECK("clGetDeviceIds GPU");

    // Setup context
    cl_context_properties props[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties) _platform, 0};
    _context = clCreateContext(props, 1, &_device, NULL, NULL, &err);
    CHECK("clCreateContext");

    // Setup queues
    _queue = clCreateCommandQueue(_context, _device, 0 /* IN-ORDER */, &err);
    CHECK("clCreateCommandQueue CPU");

    return true;
}

bool Fft::setupClFft() {
    cl_int err = 0;

    // Setup clFFT. 
    clfftSetupData fftSetup;
    err = clfftInitSetupData(&fftSetup);
    CHECK("clfftInitSetupData");
    err = clfftSetup(&fftSetup);
    CHECK("clfftSetup");    
    
    // Size of FFT. 
    size_t clLengths = _fft_size;
    clfftDim dim = CLFFT_1D;
    
    // Create a default plan for a complex FFT. 
    err = clfftCreateDefaultPlan(&_planHandle, _context, dim, &clLengths);
    CHECK("clfftCreateDefaultPlan");

    // Set plan parameters. 
    err = clfftSetPlanPrecision(_planHandle, CLFFT_SINGLE);
    CHECK("clfftSetPlanPrecision");
    err = clfftSetLayout(_planHandle, CLFFT_COMPLEX_PLANAR, CLFFT_COMPLEX_PLANAR);
    CHECK("clfftSetLayout");
    err = clfftSetResultLocation(_planHandle, CLFFT_OUTOFPLACE);
    CHECK("clfftSetResultLocation");

    // Bake the plan
    err = clfftBakePlan(_planHandle, 1, &_queue, NULL, NULL);
    CHECK("clfftBakePlan");

    return true;
}