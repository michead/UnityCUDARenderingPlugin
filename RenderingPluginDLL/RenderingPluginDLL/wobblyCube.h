#include "springSystem.h"
#include "vector_types.h"
#include "spring.h"
#include <vector>

using namespace std;

#define SPACING 0.075f

#define CUBE_LENGTH 12
#define CUBE_HEIGHT 12
#define CUBE_WIDTH 12

#define MASS 0.0005f
#define DAMPING 2
#define SPRING_COSTANT 100
#define REST_LENGTH 0.075f

class WobblyCube : public SpringSystem
{
public:
	WobblyCube();
	~WobblyCube();

	float3* evalF(float3* state);

	void populateSprings();
	void computeStructuralSprings();
	void computeShearSprings();
	void computeFlexSprings();

	inline unsigned indexOf(unsigned l, unsigned h, unsigned w);

	void computeFaces();
};