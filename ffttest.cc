#include <boost/program_options.hpp>

#include <clFFT/clFFT.h>

#include <cstdlib>
#include <chrono>
#include <iostream>
 
using namespace std;
using namespace chrono;

namespace po = boost::program_options;

#define CHECK(MSG)                        \
    if (_err != CL_SUCCESS) {             \
      cerr << MSG << " (" << _err << ")"; \
      return;                             \
    }

size_t  _fft_size       = 8192;
int     _count          = 1000;
int     _count_per_loop = 1000;
double  _range          = 25.0;
double  _min            = 0.0;
double  _invert         = false;

// OpenCL variables
cl_int                  _err;
cl_platform_id          _platform = 0;
cl_device_id            _device = 0;
cl_context_properties   _props[3] = {CL_CONTEXT_PLATFORM, 0, 0};
cl_context              _ctx = 0;
cl_command_queue        _queue = 0;

// Input and Output buffer.
cl_mem                  _buffersIn[2]  = {0, 0};

// Temporary buffer.
cl_mem                  _tmpBuffer = 0;

// Plan handle
clfftPlanHandle         _planHandle = 0;

// Data buffer
cl_double*               _inReal  = 0;
cl_double*               _inImag  = 0;

void allocate_buffers() {

    _inReal  = (cl_double*) malloc (_fft_size * sizeof (cl_double));
    _inImag  = (cl_double*) malloc (_fft_size * sizeof (cl_double));

    memset(_inReal, 0, sizeof(_inReal));
    memset(_inImag, 0, sizeof(_inImag));

    // Size of temp buffer. 
    size_t tmpBufferSize = 0;
    int status = 0;

    // Create temporary buffer. 
    status = clfftGetTmpBufSize(_planHandle, &tmpBufferSize);

    if ((status == 0) && (tmpBufferSize > 0)) {
        _tmpBuffer = clCreateBuffer(_ctx, CL_MEM_READ_WRITE, tmpBufferSize, 0, &_err);
        if (_err != CL_SUCCESS)
            printf("Error with _tmpBuffer clCreateBuffer\n");
    }

    // Prepare OpenCL memory objects : create buffer for input. 
    _buffersIn[0] = clCreateBuffer(_ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  _fft_size * sizeof(cl_double), _inReal, &_err);
    if (_err != CL_SUCCESS)
        printf("Error with _buffersIn[0] clCreateBuffer\n");

    // Prepare OpenCL memory objects : create buffer for input. 
    _buffersIn[1] = clCreateBuffer(_ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  _fft_size * sizeof(cl_double), _inImag, &_err);
    if (_err != CL_SUCCESS)
        printf("Error with _buffersIn[1] clCreateBuffer\n");
}

void populate() {
    
    srand(time(NULL));
    
    // Initialization of _inReal, _inImag, _outReal and _outImag. 
    for(int i = 0; i < _fft_size; i++) {
        _inReal[i]  = (double) rand() / RAND_MAX * _range + _min;
        _inImag[i]  = 0.0f;
    }
}

void setup_clFFT(size_t N) {

    // Setup OpenCL environment. 
    _err = clGetPlatformIDs(1, &_platform, NULL);
    _err = clGetDeviceIDs(_platform, CL_DEVICE_TYPE_GPU, 1, &_device, NULL);
    
    _props[1] = (cl_context_properties) _platform;
    _ctx = clCreateContext(_props, 1, &_device, NULL, NULL, &_err);
    _queue = clCreateCommandQueue(_ctx, _device, 0, &_err);
    
    // Setup clFFT. 
    clfftSetupData fftSetup;
    _err = clfftInitSetupData(&fftSetup);
    _err = clfftSetup(&fftSetup);    
    
    // Size of FFT. 
    size_t clLengths[1] = {N};
    clfftDim dim = CLFFT_1D;
    
    // Create a default plan for a complex FFT. 
    _err = clfftCreateDefaultPlan(&_planHandle, _ctx, dim, clLengths);
    
    // Set plan parameters. 
    _err = clfftSetPlanPrecision(_planHandle, CLFFT_DOUBLE);
    _err = clfftSetLayout(_planHandle, CLFFT_COMPLEX_PLANAR, CLFFT_COMPLEX_PLANAR);
    _err = clfftSetResultLocation(_planHandle, CLFFT_INPLACE);
    
    // Bake the plan. 
    _err = clfftBakePlan(_planHandle, 1, &_queue, NULL, NULL);
}

void dump_fft(string label) {
    
    cout << label << endl;
    cout.precision(2);
    for (int i = 0; i < 8; ++i) {
        cout << i << "    \t" << _inReal[i] << "    \t" << _inImag[i] << endl;
    }
    cout << endl;
}

void release_clFFT() {
    // Release OpenCL memory objects. 
    clReleaseMemObject(_buffersIn[0]);
    clReleaseMemObject(_buffersIn[1]);
    clReleaseMemObject(_tmpBuffer);
    
    // Release the plan. 
    _err = clfftDestroyPlan(&_planHandle);
    
    // Release clFFT library. 
    clfftTeardown();
    
    // Release OpenCL working objects. 
    clReleaseCommandQueue(_queue);
    clReleaseContext(_ctx);
}

inline void perform_fft() {

    cl_event writes[2] = {0};

    // Enqueue write tab array into _buffersIn[0]. 
    _err = clEnqueueWriteBuffer(_queue, _buffersIn[0], CL_FALSE, 0, 
                                _fft_size * sizeof(double), _inReal, 0, NULL, &writes[0]);
    CHECK("clEnqueueWriteBuffer real");
    
    _err = clEnqueueWriteBuffer(_queue, _buffersIn[1], CL_TRUE, 0,
                                _fft_size *sizeof(double), _inImag, 0, NULL, &writes[1]);
    CHECK("clEnqueueWriteBuffer imag");

    // Execute the plan.
    cl_event transform;
    _err = clfftEnqueueTransform(_planHandle, CLFFT_FORWARD, 1, &_queue, 2, writes, &transform,
                                 _buffersIn, NULL, _tmpBuffer);
    CHECK("clEnqueueTransform");
    
    // Copy result to input array
    _err = clEnqueueReadBuffer(_queue, _buffersIn[0], CL_FALSE, 0,
                               _fft_size * sizeof(double), _inReal, 1, &transform, NULL);
    CHECK("clEnqueueReadBuffer real");
    
    _err = clEnqueueReadBuffer(_queue, _buffersIn[1], CL_FALSE, 0, 
                               _fft_size * sizeof(double), _inImag, 1, &transform, NULL);
    CHECK("clEnqueueReadBuffer imag");
    
    _err = clFinish(_queue);
    CHECK("clFinish");
}

void summarize(nanoseconds total_duration) {

    double count = _count * _count_per_loop * (_invert ? 2 : 1);
    double ave = total_duration.count() / count;

    cout.precision(8);
    cerr << "\r100 %     " << endl;
    cout << endl;
    cout << "Iterations: " << _count << endl;
    cout << "Per loop:   " << _count_per_loop << endl;
    cout << "Data size:  " << _fft_size << endl;
    cout << "Range:      " << _range << endl;
    cout << "Min:        " << _min << endl;
    cout << endl;
    cout << "Time:       " << total_duration.count() << " ns" << endl;
    cout << "Average:    " << ave << " ns (" << (ave / 1000.0) << " μs)" << endl;    
}

void time_fft() {

    setup_clFFT(_fft_size);

    allocate_buffers();

    nanoseconds total_duration(0);

    for (int i = 0; i < _count; ++i) {
        populate();

        // start time
        high_resolution_clock::time_point start = high_resolution_clock::now();

        // fft loop
        for (int j = 0; j < _count_per_loop; ++j) {
            perform_fft();    
        }
    
        // compute duration
        high_resolution_clock::time_point finish = high_resolution_clock::now();

        auto duration = finish - start;
        total_duration += duration_cast<nanoseconds>(duration);
        
        // tell user
        double percent = 100.0 * ((i+1) * _count_per_loop) / (_count * _count_per_loop);
        cerr.precision(2);
        cerr << "\r" << percent << " %    ";
        cerr.flush();
    }
    
    // example
    populate();
    dump_fft("Before...");
    perform_fft();
    dump_fft("After...");
    
    release_clFFT();    

    summarize(total_duration);
}

int main(int ac, char* av[]) {

    try {
        
        po::options_description desc("Allowed options");
    
        desc.add_options()
        ("help,h",   "produce help message")
        ("count,c",  po::value<int>(), "set the number of timed loops to perform")
        ("loop,l",   po::value<int>(), "Set the number of FFT interations per loop")
        ("size,s",   po::value<int>(), "Set the size of the data buffer [8192]")
        ("range,r",  po::value<double>(), "Set the range of the random data [25.0]")
        ("min,m",    po::value<double>(), "Set the minimum value of the random data [0.0]")
        ("invert,i", "Perform an FFT, then an inverse FFT on the same data");

        po::variables_map vm;
        po::store(po::parse_command_line(ac, av, desc), vm);
        po::notify(vm);    

        if (vm.count("help")) {
            cout << desc << "\n";
            return 1;
        }

        if (vm.count("count")) {
            _count = vm["count"].as<int>();
        }
    
        if (vm.count("loop")) {
            _count_per_loop = vm["loop"].as<int>();
        }

        if (vm.count("size")) {
            _fft_size = vm["size"].as<int>();
        }
        
        if (vm.count("range")) {
            _range = vm["range"].as<double>();
        }
        
        if (vm.count("min")) {
            _min = vm["min"].as<double>();
        }
        
        if (vm.count("invert")) {
            _invert = true;
        }

        time_fft();

    } catch (exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    } catch (...) {
        cerr << "Unknown error" << endl;
        return 1;
    }

    return 0;
}
