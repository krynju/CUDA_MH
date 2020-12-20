

#include "Density.hpp"

Density::Density () {
}

float Density::operator() (float x) const {
  x /= 100;
  return normalPdf (x, 0, 1) + normalPdf (x, 3, 1) + normalPdf (x, 6, 1);

}

Density::~Density () {

}

float Density::normalPdf (float x, float m, float s) const {
  static const float inv_sqrt_2pi = 0.3989422804014327;
  float a = (x - m) / s;

  return inv_sqrt_2pi / s * std::exp (-0.5f * a * a);
}
