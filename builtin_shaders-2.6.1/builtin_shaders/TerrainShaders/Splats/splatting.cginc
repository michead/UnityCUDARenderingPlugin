#include "UnityCG.cginc"

struct v2f_ambient {
	V2F_POS_FOG;
	float4 uv[3] : TEXCOORD0;
	float fade : TEXCOORD3;
};
struct v2f_vertex {
	V2F_POS_FOG;
	float4 uv[3] : TEXCOORD0;
	float4 color : COLOR;
	#ifdef USE_LIGHTMAP
	float fade : TEXCOORD3;
	#endif
};

#ifdef INCLUDE_PIXEL
struct v2f_pixel {
	V2F_POS_FOG;
	LIGHTING_COORDS
	float3 normal;
	float4 lightDirFade;
	float4 uv[3];
};
#endif


uniform sampler2D _Control;
uniform sampler2D _Splat0,_Splat1,_Splat2,_Splat3;
uniform float4 _Splat0_ST,_Splat1_ST,_Splat2_ST,_Splat3_ST,_Splat4_ST;

uniform sampler2D _LightMap;

uniform float4 _RealtimeFade;

float4 CalculateVertexLights (float3 viewPos, float3 objSpaceNormal)
{
	float3 normal = mul (objSpaceNormal, (float3x3)glstate.matrix.transpose.modelview[0]);
	
	// Do vertex light calculation: up to four lights,
	// treat spot lights as point lights.
	// TODO: optimize/vectorize me!
	float4 lightColor = glstate.lightmodel.ambient;
	for (int i = 0; i < 4; i++) {
		float3 toLight = glstate.light[i].position.xyz - viewPos.xyz * glstate.light[i].position.w;
		float lengthSq = dot(toLight, toLight);
		float atten = 1.0 / (1.0 + lengthSq * glstate.light[i].attenuation.z);
		float lightAmt = max( 0, dot (normal, normalize(toLight)) * atten );
		lightColor += glstate.light[i].diffuse * lightAmt;
	}

	return lightColor;
}

#define CALC_SPLAT_UV(baseUV) \
	o.uv[0].xy = baseUV; \
	o.uv[1].xy = TRANSFORM_TEX (baseUV, _Splat0); \
	o.uv[1].zw = TRANSFORM_TEX (baseUV, _Splat1); \
	o.uv[2].xy = TRANSFORM_TEX (baseUV, _Splat2); \
	o.uv[2].zw = TRANSFORM_TEX (baseUV, _Splat3);

#define SAMPLE_SPLAT(i,splat_color) \
	half4 splat_control = tex2D (_Control, i.uv[0].xy); \
	splat_color = splat_control.r * tex2D (_Splat0, i.uv[1].xy); \
	splat_color += splat_control.g * tex2D (_Splat1, i.uv[1].zw); \
	splat_color += splat_control.b * tex2D (_Splat2, i.uv[2].xy); \
	splat_color += splat_control.a * tex2D (_Splat3, i.uv[2].zw);

float4 VertexlitSplatFragment (v2f_vertex i) : COLOR {
	half4 splat;
	SAMPLE_SPLAT(i,splat);
	#ifdef USE_LIGHTMAP
	half4 lightmap = tex2D (_LightMap, i.uv[0].xy);
	i.color = lerp( lightmap, i.color, saturate(i.fade) );
	#endif
	half4 col = splat * i.color;
	col *= float4 (2,2,2,0);
	return col;
}

v2f_vertex VertexlitSplatVertex (appdata_base v) {
	v2f_vertex o;
	PositionFog( v.vertex, o.pos, o.fog );
	
	float3 viewpos = mul(glstate.matrix.modelview[0], v.vertex).xyz;
	o.color = CalculateVertexLights (viewpos, v.normal);
	CALC_SPLAT_UV(v.texcoord.xy);
	
	#ifdef USE_LIGHTMAP
	o.fade = 1.0 - (o.fog * _RealtimeFade.z + _RealtimeFade.w);
	#endif	

	return o;
}


float4 AmbientSplatFragment (v2f_ambient i) : COLOR {
	half4 splat;
	SAMPLE_SPLAT(i,splat);
	
	half4 lightcolor = tex2D (_LightMap, i.uv[0].xy);	
	lightcolor = lerp( _PPLAmbient, lightcolor, saturate(i.fade) );

	half4 col = splat * lightcolor;	
	col *= float4 (2,2,2,0);
		
	return col;
}

v2f_ambient AmbientSplatVertex (appdata_base v) {
	v2f_ambient o;
	PositionFog( v.vertex, o.pos, o.fog );
	
	CALC_SPLAT_UV(v.texcoord.xy);
	
	o.fade = o.fog * _RealtimeFade.z + _RealtimeFade.w;

	return o;
}


#ifdef INCLUDE_PIXEL

float4 PixellitSplatFragment (v2f_pixel i) : COLOR {
	half4 splat;
	SAMPLE_SPLAT(i,splat);
	float atten = LIGHT_ATTENUATION(i);
	return DiffuseLight( i.lightDirFade.xyz, i.normal, splat, atten * saturate(i.lightDirFade.w) );
}

v2f_pixel PixellitSplatVertex (appdata_base v) {
	v2f_pixel o;
	PositionFog( v.vertex, o.pos, o.fog );
	CALC_SPLAT_UV(v.texcoord.xy);
	o.normal = v.normal;
	o.lightDirFade.xyz = ObjSpaceLightDir( v.vertex );
	o.lightDirFade.w = 1.0 - (o.fog * _RealtimeFade.z + _RealtimeFade.w);
	TRANSFER_VERTEX_TO_FRAGMENT(o);
	return o;
}

#endif


float4 LightmapSplatFragment (v2f_vertex i) : COLOR {
	half4 splat;
	SAMPLE_SPLAT(i,splat);
	half4 col = splat * tex2D (_LightMap, i.uv[0].xy);
	col *= float4 (2,2,2,0);
	return col;
}

v2f_vertex LightmapSplatVertex (appdata_base v) {
	v2f_vertex o;
	PositionFog( v.vertex, o.pos, o.fog );
	
	CALC_SPLAT_UV(v.texcoord.xy);

	return o;
}

