#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <boost/program_options.hpp>

#include <cstdlib>
#include <chrono>
#include <iostream>
 
using namespace std;
using namespace cv;
using namespace chrono;

namespace po = boost::program_options;

int     _fft_size       = 8192;
int     _count          = 1000;
int     _count_per_loop = 1000;
double  _range          = 25.0;
double  _min            = 0.0;
double  _invert         = false;

void populate(vector<Point2d>& data) {
    
    srand(time(NULL));

    data.clear();
    
    for (int i = 0; i < _fft_size; ++i) {
        double x = i * 0.20;
        double y = (double) rand() / RAND_MAX * _range + _min;
        data.push_back(Point2d(x, y));
    }
}

void dump_fft(String label, vector<Point2d>& data) {
    
    cout << label << " size " << data.size() << endl;
    for (int i = 0; i < 48 ; ++i) {
        cout << data[i].x << ",\t " << data[i].y << endl;
        if (i % 8 == 0 && i != 0)
            cout << endl;
    }
    cout << endl;
}

void time_fft() {

    vector<Point2d> data;
    vector<Point2d> output;

    cout.precision(2);
    cout << "0 %";
    cout.flush();

    nanoseconds total_duration(0);

    for (int i = 0; i < _count; ++i) {

        populate(data);
        if (0 == i)
            populate(output); // allocate space in output

        int j = 0;
        
        if (_invert) {
            high_resolution_clock::time_point start = high_resolution_clock::now();
            for (; j < _count_per_loop; ++j) {
                dft(data, data, 0, data.size());
                dft(data, data, DFT_INVERSE | DFT_SCALE, data.size());
            }        
            high_resolution_clock::time_point finish = high_resolution_clock::now();

            auto duration = finish - start;
            total_duration += duration_cast<nanoseconds>(duration);
            
        } else {
            high_resolution_clock::time_point start = high_resolution_clock::now();
            for (; j < _count_per_loop; ++j) {
                dft(data, output, 0, data.size());
            }        
            high_resolution_clock::time_point finish = high_resolution_clock::now();
         
            auto duration = finish - start;
            total_duration += duration_cast<nanoseconds>(duration);
        }        
       
        if (i % 10 == 0) {
            double percent = ((double) i / (double) _count * 100.0);
            cout << "\r" << percent << " %    ";
            cout.flush();
        }
    }
    
    double count = _count * _count_per_loop * (_invert ? 2 : 1);
    double ave = total_duration.count() / count;

    cout.precision(8);
    cout << "\r100 % " << endl;
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
