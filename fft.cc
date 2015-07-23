#include <iostream>

#include "fft.hh"

#define CHECK(MSG)                                          \
    if (err != CL_SUCCESS) {                                \
      std::cerr << __FILE__ << ":" << __LINE__ << " Unexpected result for " << MSG << " (" << err << ")" << std::endl;  \
      return false;                                         \
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
    clReleaseCommandQueue(_copy_queue);
    clReleaseCommandQueue(_fft_queue);
    clReleaseContext(_context);
}

bool Fft::add(FftData& data) {
    cl_int err = 0;
   
    size_t   size       = _fft_size * sizeof(cl_float);
    cl_event writes[2]  = {0};
    cl_event transform;

    // Enqueue write tab array into _local_buffers[0]. 
    err = clEnqueueWriteBuffer(_copy_queue, data._local_buffers[FftData::REAL], CL_FALSE, 0, 
                                data.buffer_size(), data._real, 0, NULL, &writes[0]);
    CHECK("clEnqueueWriteBuffer real");
    
    err = clEnqueueWriteBuffer(_copy_queue, data._local_buffers[FftData::IMAG], CL_TRUE, 0,
                                data.buffer_size(), data._imag, 0, NULL, &writes[1]);
    CHECK("clEnqueueWriteBuffer imag");

    // Enqueue the FFT
    err = clfftEnqueueTransform(_planHandle, CLFFT_FORWARD, 1, &_fft_queue, 2, writes, &transform,
                                 data._local_buffers, NULL, NULL);//_tmpBuffer);
    CHECK("clEnqueueTransform");
    
    // Copy result to input array
    err = clEnqueueReadBuffer(_copy_queue, data._local_buffers[FftData::REAL], CL_FALSE, 0,
                               data.buffer_size(), data._real, 1, &transform, &data._wait_list[FftData::REAL]);
    CHECK("clEnqueueReadBuffer real");
    
    err = clEnqueueReadBuffer(_copy_queue, data._local_buffers[FftData::IMAG], CL_FALSE, 0, 
                               data.buffer_size(), data._imag, 1, &transform, &data._wait_list[FftData::IMAG]);
    CHECK("clEnqueueReadBuffer imag");

    return true;
}

bool Fft::setupCl() {
    cl_int err = 0;

    // Setup platform 
    err = clGetPlatformIDs(1, &_platform, NULL);
    CHECK("clGetPlatformIds");

    cl_uint num_devices = 0;
    
    // Setup devices
    err = clGetDeviceIDs(_platform, CL_DEVICE_TYPE_GPU, 1, &_device, &num_devices);
    CHECK("clGetDeviceIds GPU");
    std::cout << "Num GPU devices: " << num_devices << std::endl;

    // Setup context
    _context = clCreateContext(0, 1, &_device, NULL, NULL, &err);
    CHECK("clCreateContext");

    // Setup queues
    _copy_queue = clCreateCommandQueue(_context, _device, 0 /* IN-ORDER */, &err);
    CHECK("clCreateCommandQueue CPU");
       
    _fft_queue = clCreateCommandQueue(_context, _device, 0 /* IN-ORDER */, &err);
    CHECK("clCreateCommandQueue GPU");

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
    err = clfftSetResultLocation(_planHandle, CLFFT_INPLACE);
    CHECK("clfftSetResultLocation");

    // Bake the plan. 
    err = clfftBakePlan(_planHandle, 1, &_fft_queue, NULL, NULL);
    CHECK("clfftBakePlan");

    return true;
}