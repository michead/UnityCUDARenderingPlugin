Shader "Transparent/Cutout/Bumped Specular" {
Properties {
	_Color ("Main Color", Color) = (1,1,1,1)
	_SpecColor ("Specular Color", Color) = (0.5, 0.5, 0.5, 0)
	_Shininess ("Shininess", Range (0.01, 1)) = 0.078125
	_MainTex ("Base (RGB) TransGloss (A)", 2D) = "white" {}
	_BumpMap ("Bumpmap", 2D) = "bump" {}
	_Cutoff ("Alpha cutoff", Range(0,1)) = 0.5
}

Category {
	Tags {"IgnoreProjector"="True" "RenderType"="TransparentCutout"}
	LOD 400
	Alphatest Greater [_Cutoff]
	AlphaToMask True
	ColorMask RGB
	Fog { Color [_AddFog] }
	Blend AppSrcAdd AppDstAdd
	
	// ------------------------------------------------------------------
	// ARB fragment program
	
	SubShader {
		UsePass "Transparent/Cutout/Specular/BASE"
		
		// Pixel lights
		Pass {
			Name "PPL"
			Tags { "LightMode" = "Pixel" }
CGPROGRAM
#pragma vertex vert
#pragma fragment frag
#pragma multi_compile_builtin
#pragma fragmentoption ARB_fog_exp2
#pragma fragmentoption ARB_precision_hint_fastest

#include "UnityCG.cginc"
#include "AutoLight.cginc" 

struct v2f {
	V2F_POS_FOG;
	LIGHTING_COORDS
	float3	uvK; // xy = UV, z = specular K
	float2	uv2;
	float3	viewDirT;
	float3	lightDirT;
}; 

uniform float4 _MainTex_ST, _BumpMap_ST;
uniform float _Shininess;


v2f vert (appdata_tan v)
{	
	v2f o;
	PositionFog( v.vertex, o.pos, o.fog );
	o.uvK.xy = TRANSFORM_TEX(v.texcoord,_MainTex);
	o.uvK.z = _Shininess * 128;
	o.uv2 = TRANSFORM_TEX(v.texcoord,_BumpMap);

	TANGENT_SPACE_ROTATION;
	o.lightDirT = mul( rotation, ObjSpaceLightDir( v.vertex ) );	
	o.viewDirT = mul( rotation, ObjSpaceViewDir( v.vertex ) );	
	
	TRANSFER_VERTEX_TO_FRAGMENT(o);
	return o;
}

uniform sampler2D _BumpMap;
uniform sampler2D _MainTex;
uniform float4 _Color;

float4 frag (v2f i) : COLOR
{		
	float4 texcol = tex2D( _MainTex, i.uvK.xy );
	
	// get normal from the normal map
	float3 normal = tex2D(_BumpMap, i.uv2).xyz * 2 - 1;
	
	half4 c = SpecularLight( i.lightDirT, i.viewDirT, normal, texcol, i.uvK.z, LIGHT_ATTENUATION(i) );
	c.a = texcol.a * _Color.a;
	return c;
}
ENDCG  
		}
	}
	
	// ------------------------------------------------------------------
	// Radeon 9000
	
	SubShader {
		UsePass "Transparent/Cutout/Specular/BASE"
		
		// Pixel lights with 0 light textures
		Pass {
			Name "PPL"
			Tags {
				"LightMode" = "Pixel"
				"LightTexCount" = "0"
			}

CGPROGRAM
#pragma vertex vert
#include "UnityCG.cginc"

struct v2f {
	V2F_POS_FOG;
	float2 uv		: TEXCOORD0;
	float2 uv2		: TEXCOORD3;
	float3 lightDirT: TEXCOORD2;
	float3 halfDirT	: TEXCOORD1; 
};

uniform float4 _MainTex_ST, _BumpMap_ST;

v2f vert(appdata_tan v)
{
	v2f o;
	PositionFog( v.vertex, o.pos, o.fog );
	o.uv = TRANSFORM_TEX(v.texcoord,_MainTex);
	o.uv2 = TRANSFORM_TEX(v.texcoord,_BumpMap);
	
	TANGENT_SPACE_ROTATION;
	o.lightDirT = mul( rotation, ObjSpaceLightDir( v.vertex ) );
	float3 viewDirT = mul( rotation, ObjSpaceViewDir( v.vertex ) );
	o.halfDirT = normalize( normalize(o.lightDirT) + normalize(viewDirT) );
	
	return o;
}
ENDCG
			Program "" {
				SubProgram {
					Local 0, [_SpecularLightColor0]
					Local 1, [_ModelLightColor0]
					Local 2, (0,[_Shininess],0,1)
					Local 3, [_Color]
"!!ATIfs1.0
StartConstants;
	CONSTANT c0 = program.local[0];
	CONSTANT c1 = program.local[1];
	CONSTANT c2 = program.local[2];
	CONSTANT c3 = program.local[3];
EndConstants;

StartPrelimPass;
	SampleMap r3, t3.str;			# normal
	SampleMap r2, t2.str;			# normalized light dir
	PassTexCoord r4, t1.str;		# half dir
	
	DOT3 r5.sat, r3.bias.2x, r2.bias.2x;	# diffuse (N.L)
	
	# Compute lookup UVs into specular falloff texture.
	# Normally it would be: r=sat(N.H), g=_Shininess*0.5
	# However, we'll use projective read on this to automatically
	# normalize H. Gives better precision in highlight.
	DOT3 r1.sat, r3.bias.2x, r4;	# N.H
	MUL  r1, r1, r1;				# (N.H)^2
	DOT3 r1.b.sat, r4, r4;         	# .b = |H|^2
	MUL  r1.g, r1.b, c2.g; 			# .g = |H|^2 * k
EndPass;

StartOutputPass;
	SampleMap r0, t0.str;			# main texture
	SampleMap r1, r1.str_dr;		# .a = specular (projective to normalize H)
	PassTexCoord r5, r5.str;		# diffuse
	
	MUL r1, r1.a, r5.b;
	MUL r5.rgb, r5, c1; 			# modelLightColor.rgb * diffuse
	MUL r5.rgb, r5, r0;				# * texture
	MUL r1, r1, r0.a;				# spec *= gloss
	MUL r2, r1.a, c0;				# specColor * spec
	ADD r0.rgb.2x, r5, r2;			# (diff+spec)*2
	MUL r0.a, r0, c3;
EndPass;
"
				}
			}
			SetTexture[_MainTex] {combine texture}
			SetTexture[_SpecFalloff] {combine texture}
			SetTexture[_CubeNormalize] {combine texture}
			SetTexture[_BumpMap] {combine texture}
		}

		// Pixel lights with 1 light texture
		Pass {
			Name "PPL"
			Tags {
				"LightMode" = "Pixel"
				"LightTexCount" = "1"
			}

CGPROGRAM
#pragma vertex vert
#include "UnityCG.cginc"

uniform float4 _MainTex_ST, _BumpMap_ST;
uniform float4x4 _SpotlightProjectionMatrix0;

struct v2f {
	V2F_POS_FOG;
	float2 uv		: TEXCOORD0;
	float2 uv2		: TEXCOORD4;
	float3 lightDirT: TEXCOORD3;
	float3 halfDirT	: TEXCOORD2;
	float4 LightCoord0 : TEXCOORD1;
};

v2f vert(appdata_tan v)
{
	v2f o;
	PositionFog( v.vertex, o.pos, o.fog );
	o.uv = TRANSFORM_TEX(v.texcoord,_MainTex);
	o.uv2 = TRANSFORM_TEX(v.texcoord,_BumpMap);
	
	TANGENT_SPACE_ROTATION;
	o.lightDirT = mul( rotation, ObjSpaceLightDir( v.vertex ) );
	float3 viewDirT = mul( rotation, ObjSpaceViewDir( v.vertex ) );
	o.halfDirT = normalize( normalize(o.lightDirT) + normalize(viewDirT) );
	
	o.LightCoord0 = mul(_SpotlightProjectionMatrix0, v.vertex);
	
	return o;
}
ENDCG
		Program "" {
			SubProgram {
				Local 0, [_SpecularLightColor0]
				Local 1, [_ModelLightColor0]
				Local 2, (0,[_Shininess],0,1)
				Local 3, [_Color]

"!!ATIfs1.0
StartConstants;
	CONSTANT c0 = program.local[0];
	CONSTANT c1 = program.local[1];
	CONSTANT c2 = program.local[2];
	CONSTANT c3 = program.local[3];
EndConstants;

StartPrelimPass;
	SampleMap r4, t4.str;			# normal
	SampleMap r3, t3.str;			# normalized light dir
	PassTexCoord r1, t2.str;		# half dir
	
	DOT3 r5.sat, r4.bias.2x, r3.bias.2x;	# diffuse (N.L)
	
	# Compute lookup UVs into specular falloff texture.
	# Normally it would be: r=sat(N.H), g=_Shininess*0.5
	# However, we'll use projective read on this to automatically
	# normalize H. Gives better precision in highlight.
	DOT3 r2.sat, r4.bias.2x, r1;	# N.H
	MUL  r2, r2, r2;				# (N.H)^2
	DOT3 r2.b.sat, r1, r1;         	# .b = |H|^2
	MUL  r2.g, r2.b, c2.g; 			# .g = |H|^2 * k
EndPass;

StartOutputPass;
	SampleMap r0, t0.str;			# main texture
	SampleMap r1, t1.str;			# .a = attenuation
	SampleMap r2, r2.str_dr;		# .a = specular (projective to normalize H)
	PassTexCoord r5, r5.str;		# diffuse
	
	MUL r2, r2.a, r5.b;
	MUL r5.rgb, r5, c1; 			# modelLightColor.rgb * diffuse
	MUL r5.rgb, r5, r0;				# * texture
	MUL r2, r2, r0.a;				# spec *= gloss
	MUL r3, r2.a, c0;				# specColor * spec
	ADD r0.rgb.2x, r5, r3;			# (diff+spec)*2
	MUL r0.rgb, r0, r1.a;
	MUL r0.a, r0, c3;
EndPass;
"
				}
			}
			SetTexture[_MainTex] {combine texture}
			SetTexture[_LightTexture0] {combine texture}
			SetTexture[_SpecFalloff] {combine texture}
			SetTexture[_CubeNormalize] {combine texture}
			SetTexture[_BumpMap] {combine texture}
		}
		
		// Pixel lights with 2 light textures
		Pass {
			Name "PPL"
			Tags {
				"LightMode" = "Pixel"
				"LightTexCount" = "2"
			}

CGPROGRAM
#pragma vertex vert
#include "UnityCG.cginc"

uniform float4 _MainTex_ST, _BumpMap_ST;
uniform float4x4 _SpotlightProjectionMatrix0;
uniform float4x4 _SpotlightProjectionMatrixB0;

struct v2f {
	V2F_POS_FOG;
	float2 uv		: TEXCOORD0;
	float2 uv2		: TEXCOORD5;
	float3 lightDirT: TEXCOORD3;
	float3 halfDirT	: TEXCOORD2;
	float4 LightCoord0 : TEXCOORD1;
	float4 LightCoordB0 : TEXCOORD4;
};

v2f vert(appdata_tan v)
{
	v2f o;
	PositionFog( v.vertex, o.pos, o.fog );
	o.uv = TRANSFORM_TEX(v.texcoord,_MainTex);
	o.uv2 = TRANSFORM_TEX(v.texcoord,_BumpMap);
	
	TANGENT_SPACE_ROTATION;
	o.lightDirT = mul( rotation, ObjSpaceLightDir( v.vertex ) );
	float3 viewDirT = mul( rotation, ObjSpaceViewDir( v.vertex ) );
	o.halfDirT = normalize( normalize(o.lightDirT) + normalize(viewDirT) );
	
	o.LightCoord0 = mul(_SpotlightProjectionMatrix0, v.vertex);
	o.LightCoordB0 = mul(_SpotlightProjectionMatrixB0, v.vertex);
	
	return o;
}
ENDCG

			Program "" {
				SubProgram {
					Local 0, [_SpecularLightColor0]
					Local 1, [_ModelLightColor0]
					Local 2, (0,[_Shininess],0,1)
					Local 3, [_Color]

"!!ATIfs1.0
StartConstants;
	CONSTANT c0 = program.local[0];
	CONSTANT c1 = program.local[1];
	CONSTANT c2 = program.local[2];
	CONSTANT c3 = program.local[3];
EndConstants;

StartPrelimPass;
	SampleMap r5, t5.str;			# normal
	SampleMap r3, t3.str;			# normalized light dir
	PassTexCoord r1, t2.str;		# half dir
	
	DOT3 r3.sat, r5.bias.2x, r3.bias.2x;	# diffuse (N.L)
	
	# Compute lookup UVs into specular falloff texture.
	# Normally it would be: r=sat(N.H), g=_Shininess*0.5
	# However, we'll use projective read on this to automatically
	# normalize H. Gives better precision in highlight.
	DOT3 r2.sat, r5.bias.2x, r1;	# N.H
	MUL  r2, r2, r2;				# (N.H)^2
	DOT3 r2.b.sat, r1, r1;         	# .b = |H|^2
	MUL  r2.g, r2.b, c2.g; 			# .g = |H|^2 * k
	
	MUL r4, r3, c1;
EndPass;

StartOutputPass;
	SampleMap r0, t0.str;			# main texture
	SampleMap r1, t1.stq_dq;		# .a = attenuation
	SampleMap r2, r2.str_dr;		# .a = specular (projective to normalize H)
	SampleMap r4, t4.stq_dq;		# .a = attenuation
	PassTexCoord r3, r3.str;		# diffuse
	PassTexCoord r5, r4.str;		# diffuse * modelLightColor
	
	MUL r5.rgb, r5, r0;				# * texture
	
	MUL r2, r2.a, r3.b;
	MUL r2, r2, r0.a;				# spec *= gloss
	MUL r3, r2.a, c0;				# specColor * spec
	ADD r0.rgb.2x, r5, r3;			# (diff+spec)*2
	MUL r0.rgb, r0, r1.a;
	MUL r0.rgb, r0, r4.a;
	MUL r0.a, r0, c3;
EndPass;
"
				}
			}
			SetTexture[_MainTex] {combine texture}
			SetTexture[_LightTexture0] {combine texture}
			SetTexture[_SpecFalloff] {combine texture}
			SetTexture[_CubeNormalize] {combine texture}
			SetTexture[_LightTextureB0] {combine texture}
			SetTexture[_BumpMap] {combine texture}
		}	
	}
}

FallBack "Transparent/Cutout/VertexLit", 1

}