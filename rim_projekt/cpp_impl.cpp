
#include <iostream>
#include"common.hpp"
#include<chrono>
#include "Density.hpp"
#include<string>
#include <fstream>
#include <iterator>
#include<algorithm>

using namespace std;
using namespace std::chrono;

void saveVectorToFile (const std::vector<float> &results,
                       const string &filePath) {
  std::ofstream output_file (filePath);
  std::ostream_iterator<float> output_iterator (output_file, "\n");
  std::copy (results.begin (), results.end (), output_iterator);
}

void parseArgs (int argc, char **argv, long &sigmaGenerationCount,
                long &iterationCount, string &filepath) {
  const long DEFAULT_SIGMA_ITERATIONS = 10000;
  const long DEFAULT_ITERATIONS = 100000; //10^5
  const string DEFAULT_FILE_PATH = "/tmp/results-seq.txt";

  sigmaGenerationCount = DEFAULT_SIGMA_ITERATIONS;
  iterationCount = DEFAULT_ITERATIONS;
  filepath = DEFAULT_FILE_PATH;

  if (argc >= 4) {
    try {
      sigmaGenerationCount = std::stol (argv[1]);
      iterationCount = std::stol (argv[2]);
      filepath = argv[3];
    }
    catch (const std::exception &e) {

      sigmaGenerationCount = DEFAULT_SIGMA_ITERATIONS;
      iterationCount = DEFAULT_ITERATIONS;
      filepath = DEFAULT_FILE_PATH;

    }
  }
  else {
  }

}

int main_sequential (int argc, char **argv) {
  steady_clock::time_point begin = steady_clock::now();

  long sigmaGenerationCount = 0;
  long iterationCount = 0;
  string filepath;
  parseArgs (argc, argv, sigmaGenerationCount, iterationCount, filepath);

  unsigned int seed = duration_cast < milliseconds
      > (system_clock::now ().time_since_epoch ()).count ();
  Common sampler (seed);

  auto params = sampler.countSigma (0, sigmaGenerationCount);
  float x_init = params.first;
  float sigma = params.second;

  std::vector<float> results = sampler.generateRandomSample (x_init, sigma,
                                                             iterationCount);
//  saveVectorToFile (results, filepath);

  steady_clock::time_point end = steady_clock::now();
  auto duration = duration_cast<milliseconds>(end - begin).count();
  cout<<"Execution time: "<< duration <<" ms"<<endl;
  return 0;
}

int main_parallel(int argc, char** argv) {
    steady_clock::time_point begin = steady_clock::now();

    long sigmaGenerationCount = 0;
    long iterationCount = 0;
    long PARALLEL_COUNT = 5;

    string filepath;
    parseArgs(argc, argv, sigmaGenerationCount, iterationCount, filepath);

    unsigned int seed = duration_cast <milliseconds
    > (system_clock::now().time_since_epoch()).count();
    Common sampler(seed);

    auto params = sampler.countSigma(0, sigmaGenerationCount);
    float x_init = params.first;
    float sigma = params.second;

    const long ITERATIONS_PER_PARALLEL = iterationCount / PARALLEL_COUNT;

    std::shared_ptr<std::vector<float>> results = std::make_shared<std::vector<float>>();
    results->reserve(iterationCount);

#pragma omp parallel for
    for (int threadNumber = 0; threadNumber < PARALLEL_COUNT; ++threadNumber)
    {
        unsigned int nowTime = duration_cast <milliseconds> (system_clock::now().time_since_epoch()).count();
        Common sampler(nowTime);
        std::vector<float> resultsLocal = sampler.generateRandomSample(x_init, sigma, ITERATIONS_PER_PARALLEL);

#pragma omp critical
        {
            results->insert(results->end(), resultsLocal.begin(), resultsLocal.end());
        }
    }

    //  saveVectorToFile (*results, filepath);

    steady_clock::time_point end = steady_clock::now();
    auto duration = duration_cast<milliseconds>(end - begin).count();
    cout << "Execution time: " << duration << " ms" << endl;
    return 0;
}
