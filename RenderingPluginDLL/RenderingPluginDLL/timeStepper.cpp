#include "timeStepper.h"

void RK4::takeStep(ParticleSystem* ps, float step)
{
	float3* k1 = (float3*)malloc(ps->stateCount * sizeof(float3));
	float3* k2 = (float3*)malloc(ps->stateCount * sizeof(float3));
	float3* k3 = (float3*)malloc(ps->stateCount * sizeof(float3));
	float3* k4 = (float3*)malloc(ps->stateCount * sizeof(float3));
	float3* f_2 = (float3*)malloc(ps->stateCount * sizeof(float3));
	float3* f_3 = (float3*)malloc(ps->stateCount * sizeof(float3));
	float3* f_4 = (float3*)malloc(ps->stateCount * sizeof(float3));

	float3* state = ps->state;
	unsigned stateCount = ps->stateCount;

	float3* f1 = ps->evalF(state);
	for (int i = 0; i < stateCount; i++) k1[i] = { step * f1[i].x, step * f1[i].y, step * f1[i].z };

	for (int i = 0; i < stateCount; i++) f_2[i] = { state[i].x + k1[i].x / 2, state[i].y + k1[i].y / 2, state[i].z + k1[i].z / 2 };
	float3* f2 = ps->evalF(f_2);
	for (int i = 0; i < stateCount; i++) k2[i] = { step * f2[i].x, step * f2[i].y, step * f2[i].z };

	for (int i = 0; i < stateCount; i++) f_3[i] = { state[i].x + k2[i].x / 2, state[i].y + k2[i].y / 2, state[i].z + k2[i].z / 2 };
	float3* f3 = ps->evalF(f_3);
	for (int i = 0; i < stateCount; i++) k3[i] = { step * f3[i].x, step * f3[i].y, step * f3[i].z };

	for (int i = 0; i < stateCount; i++) f_4[i] = { state[i].x + k3[i].x, state[i].y + k3[i].y, state[i].z + k3[i].z };
	float3* f4 = ps->evalF(f_4);
	for (int i = 0; i < stateCount; i++) k4[i] = { step * f4[i].x, step * f4[i].y, step * f4[i].z };

	for (int i = 0; i < stateCount; i++) ps->state[i] = { 
		state[i].x + (k1[i].x + 2 * k2[i].x + 2 * k3[i].x + k4[i].x) / 6,
		state[i].y + (k1[i].y + 2 * k2[i].y + 2 * k3[i].y + k4[i].y) / 6,
		state[i].z + (k1[i].z + 2 * k2[i].z + 2 * k3[i].z + k4[i].z) / 6 
	};

	free(f1);
	free(f2);
	free(f3);
	free(f4);
	free(f_2);
	free(f_3);
	free(f_4);
	free(k1);
	free(k2);
	free(k3);
	free(k4);
}