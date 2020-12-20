
#ifndef COMMON_HPP_
#define COMMON_HPP_

#include<stdexcept>
#include<random>
#include<cmath>
#include <iostream>
#include<limits>

#include "Density.hpp"
#include<vector>
class Common {

  private:
    int _seed;
    std::default_random_engine _generator;
    Density _densityFunction;

    float generateCandidate (float previousSample, float sigma);

    float getCandidateScore (const float candidate, const float PreviousSampleDensity, float &candidateDensity) const;

    float getSigmaDifference (int acceptedCount, int totalCount,
                              float proportionalConstans,
                              float idealAcceptanceLevel) const;
  public:

    Common (int seed);

    std::pair<float, float> countSigma (float initialSample,
                                        long sigmaGenerationIterations);

    std::vector<float> generateRandomSample (float initialSample, float sigma,
                                             long iterationCount);

};

#endif /* COMMON_HPP_ */
