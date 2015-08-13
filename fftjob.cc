#include "fftjob.hh"

#include <ctime>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <math.h>

FftJob::FftJob(size_t size) 
 : _size(size)
{
    _real  = new cl_float[_size];
    _imag  = new cl_float[_size];
}


FftJob::~FftJob() {
    release();
}

void FftJob::copy(FftJob& other) {
    for (int i = 0; i < _size; ++i) {
        _real[i] = other._real[i];
        _imag[i] = other._imag[i];
    }    
}

void FftJob::compare(FftJob& other) {
    
    double diff = 0;
    
    for (int i = 0; i < _size; ++i) {
        diff += abs(abs(_real[i]) - abs(other._real[i]));
        diff += abs(abs(_imag[i]) - abs(other._imag[i])); 
    }
    diff /= 2 * _size;
    std::cout << "Ave diff " << std::setprecision(10) << diff << std::endl;
}

void FftJob::randomize(double range, double min) {
    
    srand(time(NULL));

    for(int i = 0; i < _size; i++) {
        _real[i]  = (float) rand() / RAND_MAX * range + min;
        _imag[i]  = 0.0f;
    }
}

void FftJob::periodic() {
    for (int i = 0; i < _size; ++i) {
        double t = i * .002;
        _real[i] = sin(2 * M_PI * t);
        _imag[i] = 0;
    }
}

void FftJob::invert() {
    for (int i = 0; i < _size; ++i) {
        cl_float swap = _real[i];
        _real[i] = _imag[i];
        _imag[i] = swap;
    }   
}

void FftJob::scale(double factor) {
    for (int i = 0; i < _size; ++i) {
        _real[i] *= factor;
        _imag[i] *= factor;
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

void FftJob::write(std::string filename) {
    std::ofstream ofs;
    ofs.open(filename);
    
    for (int i = 0; i < _size; ++i) {
        ofs << _real[i] << ", " << _imag[i] << std::endl;
    }
    
    ofs.close();   
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