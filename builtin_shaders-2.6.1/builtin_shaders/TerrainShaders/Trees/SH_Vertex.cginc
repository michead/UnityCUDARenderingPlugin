#include "UnityCG.cginc"
#include "TerrainEngine.cginc"


uniform float _Occlusion, _AO, _BaseLight;
uniform float4 _Color;
uniform float3[4] _TerrainTreeLightDirections;

struct v2f {
	float4 pos : POSITION;
	float fog : FOGC;
	float4 uv : TEXCOORD0;
	float4 color : COLOR0;
};

uniform float4x4 _CameraToWorld;

// original: 56 instructions
// point lights: 93 instructions
v2f leaves(appdata_tree v)
{
	v2f o;
	
	TerrainAnimateTree(v.vertex, v.color.w);
	
	float3 viewpos = mul(glstate.matrix.modelview[0], v.vertex).xyz;
	o.pos = mul(glstate.matrix.mvp, v.vertex);
	o.fog = o.pos.z;
	o.uv = v.texcoord;
	
	float4 lightDir;
	lightDir.w = _AO;

	float4 lightColor = glstate.lightmodel.ambient;
	for (int i = 0; i < 4; i++) {
		#ifdef USE_CUSTOM_LIGHT_DIR
		lightDir.xyz = _TerrainTreeLightDirections[i];
		float atten = 1.0;
		#else
		float3 toLight = glstate.light[i].position.xyz - viewpos.xyz * glstate.light[i].position.w;
		toLight.yz *= -1.0;
		lightDir.xyz = mul( (float3x3)_CameraToWorld, normalize(toLight) );
		float lengthSq = dot(toLight, toLight);
		float atten = 1.0 / (1.0 + lengthSq * glstate.light[i].attenuation.z);		
		#endif

		lightDir.xyz *= _Occlusion;
		float occ =  dot (v.tangent, lightDir);
		occ = max(0, occ);
		occ += _BaseLight;
		occ *= atten;
		lightColor += glstate.light[i].diffuse * occ;
	}

	lightColor.a = 1;
//	lightColor = saturate(lightColor);
	
	o.color = lightColor * _Color;
	#ifdef WRITE_ALPHA_1
	o.color.a = 1;
	#endif
	return o; 
}


// original: 50 instructions
// point lights: 87 instructions
v2f bark(appdata_tree v) {
	v2f o;
	
	TerrainAnimateTree(v.vertex, v.color.w);
	
	float3 viewpos = mul(glstate.matrix.modelview[0], v.vertex).xyz;
	o.pos = mul(glstate.matrix.mvp, v.vertex);
	o.fog = o.pos.z;
	o.uv = v.texcoord;
	
	float4 lightDir;
	lightDir.w = _AO;

	float4 lightColor = glstate.lightmodel.ambient;
	for (int i = 0; i < 4; i++) {
		#ifdef USE_CUSTOM_LIGHT_DIR
		lightDir.xyz = _TerrainTreeLightDirections[i];
		float atten = 1.0;
		#else
		float3 toLight = glstate.light[i].position.xyz - viewpos.xyz * glstate.light[i].position.w;
		toLight.yz *= -1.0;
		lightDir.xyz = mul( (float3x3)_CameraToWorld, normalize(toLight) );
		float lengthSq = dot(toLight, toLight);
		float atten = 1.0 / (1.0 + lengthSq * glstate.light[i].attenuation.z);		
		#endif
		float occ = dot (lightDir.xyz, v.normal);
		occ = max(0, occ);
		occ *=  _AO * v.tangent.w + _BaseLight;
		occ *= atten;
		lightColor += glstate.light[i].diffuse * occ;
	}
	
	lightColor.a = 1;
	o.color = lightColor * _Color;	
	
	#ifdef WRITE_ALPHA_1
	o.color.a = 1;
	#endif
	return o; 
}
