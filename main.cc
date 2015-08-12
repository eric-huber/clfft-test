#include <boost/program_options.hpp>

#include <clFFT.h>

#include <cstdlib>
#include <chrono>
#include <iostream>
#include <iomanip>
 
#include "fft.hh"

using namespace std;
using namespace chrono;

namespace po = boost::program_options;

void test_fft(size_t size, int count, int loop, double range, double min) {

    Fft fft(size);
    if (!fft.init()) {
        fft.shutdown();
        return;
    }
    
    FftJob job(size);
    job.randomize(range, min);

    job.dump("Initial random buffer");

    // perform fft
    fft.add(job);

    // wait for completion
    fft.wait_all();

    //buffer.is_finished();

    job.dump("FFT");

    // cleanup
    fft.shutdown();
}

void time_fft(size_t size, int count, int loop, double range, double min) {

    Fft fft(size);
    if (!fft.init()) {
        fft.shutdown();
        return;
    }

    for (int outer = 0; outer < count; ++outer) {
        
        // start timer
        high_resolution_clock::time_point start = high_resolution_clock::now();
    
        for (auto datum : buffer) {
            // perform fft
            //fft.add(*datum);
        }
        cout << "buffer added" << endl;
    
        // wait for completion
        // ???

        // end timer            
        high_resolution_clock::time_point finish = high_resolution_clock::now();

        auto duration = finish - start;
        total_duration += duration_cast<nanoseconds>(duration);
                
        if (outer % 10 == 0) {
            double percent = ((double) outer / (double) count * loop * 100.0);
            cerr << "\r" << percent << " %    ";
            cerr.flush();
        }

    }
    
    fft.shutdown();
    
    // report time
    double ave = total_duration.count() / (count * loop);

    cout.precision(8);
    cerr << "\r100 % " << endl;
    cout << endl;
    cout << "Iterations: " << count << endl;
    cout << "Per loop:   " << loop << endl;
    cout << "Data size:  " << size << endl;
    cout << "Range:      " << range << endl;
    cout << "Min:        " << min << endl;
    cout << endl;
    cout << "Time:       " << total_duration.count() << " ns" << endl;
    cout << "Average:    " << ave << " ns (" << (ave / 1000.0) << " μs)" << endl;  
}

int main(int ac, char* av[]) {

    size_t  fft_size    = 8192;
    int     count       = 1000;
    int     loop        = 1000;
    double  range       = 25.0;
    double  min         = 0.0;

    try {
        
        po::options_description desc("Allowed options");
    
        desc.add_options()
        ("help,h",   "produce help message")
        ("count,c",  po::value<int>(), "set the number of timed loops to perform")
        ("loop,l",   po::value<int>(), "Set the number of FFT interations per loop")
        ("size,s",   po::value<int>(), "Set the size of the buffer buffer [8192]")
        ("range,r",  po::value<double>(), "Set the range of the random buffer [25.0]")
        ("min,m",    po::value<double>(), "Set the minimum value of the random buffer [0.0]")
        ("invert,i", "Perform an FFT, then an inverse FFT on the same buffer");

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
        
        if (vm.count("loop")) {
            loop = vm["loop"].as<int>();
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

    test_fft(fft_size, count, loop, range, min);
    //time_fft(fft_size, count, loop, range, min);
    
    return 0;
}
