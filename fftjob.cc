#include "fftjob.hh"

#include <ctime>
#include <cstdlib>
#include <iostream>
#include <iomanip>

FftJob::FftJob(size_t size) 
 : _size(size)
{
    _real  = new cl_float[_size];
    _imag  = new cl_float[_size];
}


FftJob::~FftJob() {
    release();
}

void FftJob::randomize(double range, double min) {
    
    srand(time(NULL));

    for(int i = 0; i < _size; i++) {
        _real[i]  = (float) rand() / RAND_MAX * range + min;
        _imag[i]  = 0.0f;
    }
}

void FftJob::dump(std::string label) {

    std::cout << label << std::endl;
    
    for (int i = 0; i < 32; i = i + 4) {
        for (int j = 0; j < 4; ++j) { 
            std::cout << std::setw(5) << std::setprecision(1) << std::fixed << std::setfill(' ') << _real[i + j] << "  ";
            std::cout << std::setw(5) << std::setprecision(1) << std::fixed << std::setfill(' ') << _imag[i + j];
            if (3 != j)
                std::cout << "    ";
        }
        std::cout << std::endl;
    }
}

void FftJob::release() {
    if (NULL != _real) {
        delete[] _real;
        _real = NULL;
    }
    if (NULL != _imag) {
        delete[] _imag;
        _imag = NULL;
    }
}