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


Fft::Fft(size_t fft_size, bool use_cpu, int parallel)
  : _fft_size(fft_size),
    _use_cpu(use_cpu),
    _parallel(parallel)
{
}

bool Fft::init() {

    if (select_platform() && setup_cl() && setup_clFft() && setup_buffers())
        return true;
    return false;
}

void Fft::shutdown() {
    
    if (_buffers.empty()) {
        for (int i = _parallel-1; 0 <= i; --i) {
            _buffers.at(i)->release();
            _buffers.pop_back();
        }
        
    }
    
    // Release clFFT library. 
    clfftTeardown();
    
    // Release OpenCL working objects. 
    clReleaseCommandQueue(_queue);
    clReleaseContext(_context);
}

bool Fft::forward(FftJob& job) {
    cl_int err = 0;
   
    cl_event writes[2]  = {0};
    cl_event reads[2] = {0};
    cl_event transform = 0;

    // get buffer (may block)
    FftBuffer* buffer = get_buffer();
    // temp - no buffer? return false
    if (NULL == buffer)
        return NULL;
    buffer->set_job(&job);

    // Enqueue write tab array into _local_buffers[0]. 
    err = clEnqueueWriteBuffer(_queue, buffer->in_real(), CL_FALSE, 0, 
                                buffer->buffer_size(), buffer->data_real(), 0, NULL, &writes[0]);
    CHECK("clEnqueueWriteBuffer real");

    err = clEnqueueWriteBuffer(_queue, buffer->in_imag(), CL_FALSE, 0,
                                buffer->buffer_size(), buffer->data_imag(), 0, NULL, &writes[1]);
    CHECK("clEnqueueWriteBuffer imag");

    // Enqueue the FFT
    err = clfftEnqueueTransform(_planHandle, CLFFT_FORWARD, 1, &_queue, 2, writes, &transform,
                                 buffer->in_buffers(), buffer->out_buffers(), buffer->temp_buffer());
    CHECK("clEnqueueTransform");

    // Copy result to input array
    err = clEnqueueReadBuffer(_queue, buffer->out_real(), CL_FALSE, 0,
                               buffer->buffer_size(), buffer->data_real(), 1, &transform, &reads[0]);
    CHECK("clEnqueueReadBuffer real");

    err = clEnqueueReadBuffer(_queue, buffer->out_imag(), CL_FALSE, 0, 
                               buffer->buffer_size(), buffer->data_imag(), 1, &transform, &reads[1]);
    CHECK("clEnqueueReadBuffer imag");

    buffer->set_wait(reads);

    return true;
}


bool Fft::backward(FftJob& job) {
    cl_int err = 0;
   
    cl_event writes[2]  = {0};
    cl_event reads[2] = {0};
    cl_event transform = 0;

    // get buffer (may block)
    FftBuffer* buffer = get_buffer();
    // temp - no buffer? return false
    if (NULL == buffer)
        return NULL;
    buffer->set_job(&job);

    // Enqueue write tab array into _local_buffers[0]. 
    err = clEnqueueWriteBuffer(_queue, buffer->in_real(), CL_FALSE, 0, 
                                buffer->buffer_size(), buffer->data_real(), 0, NULL, &writes[0]);
    CHECK("clEnqueueWriteBuffer real");

    err = clEnqueueWriteBuffer(_queue, buffer->in_imag(), CL_FALSE, 0,
                                buffer->buffer_size(), buffer->data_imag(), 0, NULL, &writes[1]);
    CHECK("clEnqueueWriteBuffer imag");

    // Enqueue the FFT
    err = clfftEnqueueTransform(_planHandle, CLFFT_BACKWARD, 1, &_queue, 2, writes, &transform,
                                 buffer->in_buffers(), buffer->out_buffers(), buffer->temp_buffer());
    CHECK("clEnqueueTransform");

    // Copy result to input array
    err = clEnqueueReadBuffer(_queue, buffer->out_real(), CL_FALSE, 0,
                               buffer->buffer_size(), buffer->data_real(), 1, &transform, &reads[0]);
    CHECK("clEnqueueReadBuffer real");

    err = clEnqueueReadBuffer(_queue, buffer->out_imag(), CL_FALSE, 0, 
                               buffer->buffer_size(), buffer->data_imag(), 1, &transform, &reads[1]);
    CHECK("clEnqueueReadBuffer imag");

    buffer->set_wait(reads);

    return true;
}

void Fft::wait_all() {
    for (auto buffer : _buffers) {
        if (buffer->in_use())
            buffer->wait();
    }
}

size_t Fft::get_temp_buffer_size() {
    size_t size = 0;
    int status = clfftGetTmpBufSize(_planHandle, &size);
    return 0 == status ? size : 0;
}

bool Fft::select_platform() {
    cl_int          err = 0;
    cl_uint         platform_count = 0;
    cl_platform_id  platform[5];
    cl_device_type  type = _use_cpu ? CL_DEVICE_TYPE_CPU : CL_DEVICE_TYPE_GPU;

    // get list of platforms
    err = clGetPlatformIDs(0, NULL, &platform_count);
    CHECK("clGetPlatformIds - platform count");
    
    err = clGetPlatformIDs(5, platform, NULL);
    CHECK("clGetPlatformIds - list of platforms");
    
    // find a platform supporting our device type
    for (uint i = 0; i < platform_count; ++i) {
        err = clGetDeviceIDs(platform[i], type, 1, &_device, NULL);
        if (err == CL_SUCCESS) {
            _platform = platform[i];
            return true;
        }
    }
    
    return false;
}

bool Fft::setup_cl() {
    cl_int err = 0;

    // Setup context
    cl_context_properties props[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties) _platform, 0};
    _context = clCreateContext(props, 1, &_device, NULL, NULL, &err);
    CHECK("clCreateContext");

    // Setup queues
    _queue = clCreateCommandQueue(_context, _device, 0 /* IN-ORDER */, &err);
    CHECK("clCreateCommandQueue CPU");

    return true;
}

bool Fft::setup_clFft() {
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

bool Fft::setup_buffers() {
    for (int i = 0; i < _parallel; ++i) {
        _buffers.push_back(new FftBuffer(*this));
    }
}

FftBuffer* Fft::get_buffer() {
    
    for (auto buffer : _buffers) {
        if (buffer->in_use())
            continue;
        buffer->set_in_use(true);
        return buffer;
    }
    
    return NULL;
}