

#include"common.hpp"
using namespace std;

Common::Common (int seed) : _seed (seed), _generator (seed) {
}

float Common::generateCandidate (float previousSample, float sigma) {
  std::normal_distribution<float> normalDistribution (previousSample, sigma);
  return normalDistribution (_generator);
}

float Common::getCandidateScore (const float candidate, const float PreviousSampleDensity, float &candidateDensity) const {
  candidateDensity = _densityFunction (candidate);
  return candidateDensity / PreviousSampleDensity;
}

float Common::getSigmaDifference (int acceptedCount, int totalCount,
                                  float proportionalConstans,
                                  float idealAcceptanceLevel) const {
  float acceptanceLevel = (float) acceptedCount / (float) totalCount;
  return proportionalConstans * (acceptanceLevel - idealAcceptanceLevel);
}

std::pair<float, float> Common::countSigma (float initialSample,
                                            long sigmaGenerationIterations) {
  float sigma = 1.0;
  float previousSample = initialSample;
  float candidate = 0.0;
  float score = 0;
  float acceptationLevel = 1;
  const float IDEAL_ACCEPTED_PERCENT = 0.3;
  float P = 0.3;
  long accepted = 0;
  float previousSampleDensity = _densityFunction(previousSample);
  float candidateSampleDensity = 0;

  std::uniform_real_distribution<float> uniformDistribution (0, 1);

  for (int iter = 1; iter <= sigmaGenerationIterations; ++iter) {
    candidate = generateCandidate (previousSample, sigma);
    score = getCandidateScore (candidate, previousSampleDensity, candidateSampleDensity);
    acceptationLevel = uniformDistribution (_generator);

    if (score >= acceptationLevel) {
      ++accepted;
      previousSample = candidate;
      previousSampleDensity = candidateSampleDensity;
    }

    P = 1000 / (sigma);
    sigma += getSigmaDifference (accepted, iter, P, IDEAL_ACCEPTED_PERCENT);

    if (sigma <= 0) {
      sigma = std::numeric_limits<float>::epsilon ();
    }
//    std::cout << "acceptation level: " << (float) accepted / (float) iter
//        << std::endl;
  }
//
//  std::cout << "acceptation level: "
//      << (float) accepted / (float) sigmaGenerationIterations << std::endl;
  return std::pair<float, float> (previousSample, sigma);
}

std::vector<float> Common::generateRandomSample (float initialSample,
                                                 float sigma,
                                                 long iterationCount) {

  float previousSample = initialSample;
  float candidate = 0.0;
  float score = 0;
  float acceptationLevel = 1;
  float previousSampleDensity = _densityFunction(previousSample);
  float candidateSampleDensity = 0;

  std::vector<float> results(iterationCount);

  std::uniform_real_distribution<float> uniformDistribution (0, 1);

  for (int iterationNumber = 1; iterationNumber <= iterationCount; ++iterationNumber) {
    candidate = generateCandidate (previousSample, sigma);
    score = getCandidateScore (candidate, previousSampleDensity, candidateSampleDensity);
    acceptationLevel = uniformDistribution (_generator);

    if (score >= acceptationLevel) {
      previousSample = candidate;
      previousSampleDensity = candidateSampleDensity;
    }
    results[iterationNumber - 1] = previousSample;
  }
//  cout << "result counted after iteration: " << iterationNumber << endl;
  return results;
}

