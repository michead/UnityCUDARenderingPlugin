#include "particleSystem.h"
#include <vector>
#include "vecmath\vecmath.h"

class TimeStepper
{
public:
	virtual void takeStep(ParticleSystem* ps, float step) = 0;
};

class RK4 : public TimeStepper
{
	void takeStep(ParticleSystem* ps, float step);
};