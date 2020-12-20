

#ifndef DENSITY_HPP_
#define DENSITY_HPP_

#include<cmath>
#include<random>

class Density {
  public:
    Density ();

    float operator()(float x)const;

    virtual ~Density ();
  private:
    float normalPdf(float x, float m, float s)const;

};

#endif /* DENSITY_HPP_ */
