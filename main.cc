#include <boost/program_options.hpp>

#include <clFFT.h>

#include <cstdlib>
#include <iostream>
#include <iomanip>
 
#include "fft.hh"

using namespace std;

namespace po = boost::program_options;

void dump(std::string label, size_t size, cl_float* real, cl_float* imag) {

    std::cout << label << std::endl;
    
    for (int i = 0; i < 32; i = i + 4) {
        for (int j = 0; j < 4; ++j) { 
            std::cout << std::setw(5) << std::setprecision(1) << std::fixed << std::setfill(' ') << real[i + j] << "  ";
            std::cout << std::setw(5) << std::setprecision(1) << std::fixed << std::setfill(' ') << imag[i + j];
            if (3 != j)
                std::cout << "    ";
        }
        std::cout << std::endl;
    }
}

void test_fft(size_t size, int count, double range, double min) {

    Fft fft(size);
    if (!fft.init()) {
        fft.shutdown();
        return;
    }

    // allocate buffers    
    cl_float* real  = (cl_float*) malloc (size * sizeof (cl_float));
    cl_float* imag  = (cl_float*) malloc (size * sizeof (cl_float));

    srand(time(NULL));
    
    // Initialization of _real, _imag, _outReal and _outImag. 
    for(int i = 0; i < size; i++) {
        real[i]  = (float) rand() / RAND_MAX * range + min;
        imag[i]  = 0.0f;
    }

    dump("Init:", size, real, imag);

    // init data object
    FftData data(fft);
    data.set(real, imag);

    // perform fft
    fft.add(data);

    // wait for completion
    data.wait();

    dump("Final:", size, real, imag);
    
    fft.shutdown();    
}

int main(int ac, char* av[]) {

    size_t  fft_size    = 8192;
    int     count       = 1000;
    double  range       = 25.0;
    double  min         = 0.0;

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
            count = vm["count"].as<int>();
        }

        if (vm.count("size")) {
            fft_size = vm["size"].as<int>();
        }
        
        if (vm.count("range")) {
            range = vm["range"].as<double>();
        }
        
        if (vm.count("min")) {
            min = vm["min"].as<double>();
        }

    } catch (exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    } catch (...) {
        cerr << "Unknown error" << endl;
        return 1;
    }

    test_fft(fft_size, count, range, min);
    
    return 0;
}
