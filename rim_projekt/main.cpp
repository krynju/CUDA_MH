#include "cuda_header.h"

#include <string>
#include <fstream>
#include <iterator>

#define NN 10000000 // 0.5 *10^9 is a limit for 1060 3gb , float32
#define BURN_N 100000



#define save_result true

#define use_matlab false

#if use_matlab
    #include "MatlabEngineWrapper.h"
#endif


int main(int argc, char** argv) {

    int N = NN;
    int burn_N = BURN_N;
    int save = 0;


    if (argc == 4) {
        burn_N = std::stol(argv[1]);
        N = std::stol(argv[2]);
        save = std::stol(argv[3]);
    }
    if (argc ==3) {
        burn_N = std::stol(argv[1]);
        N = std::stol(argv[2]);
    }

	float *x = cuda_main(N, burn_N);

#if use_matlab

    float test_float_array[] = { 1, 2, 3, 4, 5 };
    MatlabEngineWrapper::instance().setArray("test", test_float_array, 5);

    MatlabEngineWrapper::instance().setArray("result", x, N);

#endif

    if (save || save_result ) {
        std::ofstream output_file("result.csv");
        std::ostream_iterator<float> output_iterator(output_file, "\n");
        std::copy(x, x+N, output_iterator);
    }

	return 0;
}