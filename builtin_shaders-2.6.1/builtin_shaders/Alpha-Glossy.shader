Shader "Transparent/Specular" {
Properties {
	_Color ("Main Color", Color) = (1,1,1,1)
	_SpecColor ("Specular Color", Color) = (0.5, 0.5, 0.5, 0)
	_Shininess ("Shininess", Range (0.01, 1)) = 0.078125
	_MainTex ("Base (RGB) TransGloss (A)", 2D) = "white" {}
}

Category {
	Tags {"Queue"="Transparent" "IgnoreProjector"="True" "RenderType"="Transparent"}
	LOD 300
	Alphatest Greater 0
	ZWrite Off
	ColorMask RGB
	Fog { Color [_AddFog] }
	
	// ------------------------------------------------------------------
	// ARB fragment program
	
	SubShader {
		// Ambient pass
		Pass {
			Name "BASE"
			Tags {"LightMode" = "PixelOrNone"}
			Blend SrcAlpha OneMinusSrcAlpha
			Color [_PPLAmbient]
			SetTexture [_MainTex] {constantColor [_Color] Combine texture * primary DOUBLE, texture * primary}
		}
		// Vertex lights
		Pass {
			Blend SrcAlpha OneMinusSrcAlpha
			Name "BASE"
			Tags {"LightMode" = "Vertex"}
			Lighting On
			Material {
				Diffuse [_Color]
				Emission [_PPLAmbient]
				Specular [_SpecColor]
				Shininess [_Shininess]
			}
			SeparateSpecular On
CGPROGRAM
#pragma fragment frag
#pragma fragmentoption ARB_fog_exp2
#pragma fragmentoption ARB_precision_hint_fastest

#include "UnityCG.cginc"

uniform sampler2D _MainTex;

half4 frag (v2f_vertex_lit i) : COLOR {
	half4 texcol = tex2D( _MainTex, i.uv );
	half4 c;
	c.xyz = ( texcol.xyz * i.diff.xyz + i.spec.xyz * texcol.a ) * 2;
	c.w = texcol.w * i.diff.w;
	return c;
}
ENDCG
		}
		
		// Pixel lights
		Pass {
			Name "PPL"
			Tags { "LightMode" = "Pixel" }
			Blend SrcAlpha One
CGPROGRAM
#pragma fragment frag
#pragma vertex vert
#pragma multi_compile_builtin_noshadows
#pragma fragmentoption ARB_fog_exp2
#pragma fragmentoption ARB_precision_hint_fastest
#include "UnityCG.cginc"
#include "AutoLight.cginc" 

struct v2f {
	V2F_POS_FOG;
	LIGHTING_COORDS
	float3	uvK; // xy = UV, z = specular K
	float3	viewDir;
	float3	normal;
	float3	lightDir;
}; 

uniform float4 _MainTex_ST;
uniform float _Shininess;

v2f vert (appdata_tan v)
{
	v2f o;
	PositionFog( v.vertex, o.pos, o.fog );
	o.normal = v.normal;
	o.uvK.xy = TRANSFORM_TEX(v.texcoord,_MainTex);
	o.uvK.z = _Shininess * 128;
	o.lightDir = ObjSpaceLightDir( v.vertex );
	o.viewDir = ObjSpaceViewDir( v.vertex );
	TRANSFER_VERTEX_TO_FRAGMENT(o);
	return o;
}

uniform sampler2D _MainTex;
uniform float4 _Color;

float4 frag (v2f i) : COLOR
{	
	half4 texcol = tex2D( _MainTex, i.uvK.xy );
	half4 c = SpecularLight( i.lightDir, i.viewDir, i.normal, texcol, i.uvK.z, LIGHT_ATTENUATION(i) );
	c.a = texcol.a * _Color.a;
	return c;
}
ENDCG  
		}
	}
	
 	// ------------------------------------------------------------------
	// Radeon 9000
	
	SubShader {
		// Ambient pass
		Pass {
			Blend SrcAlpha OneMinusSrcAlpha
			Name "BASE"
			Tags {"LightMode" = "PixelOrNone"}
			Color [_PPLAmbient]
			SetTexture [_MainTex] {Combine texture * primary DOUBLE, texture * primary}
		}
		// Vertex lights
		Pass {
			Blend SrcAlpha OneMinusSrcAlpha
			Name "BASE"
			Tags {"LightMode" = "Vertex"}
			Lighting On
			Material {
				Diffuse [_Color]
				Emission [_PPLAmbient]
				Specular [_SpecColor]
				Shininess [_Shininess]
			}
			SeparateSpecular On
			Program "" {
				SubProgram {
					"!!ATIfs1.0
					StartOutputPass;
						SampleMap r0, t0.str;	 # main texture
						MUL r0, color0, r0;
						MAD r0.rgb.2x, color1, r0.a, ro;
					EndPass; 
					"
				}
			}
			SetTexture [_MainTex] {combine texture}
		}
		
		// Pixel lights with 0 light textures
		Pass {
			Blend SrcAlpha One
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
	float3 normal	: TEXCOORD3;
	float3 lightDir	: TEXCOORD2;
	float3 halfDir	: TEXCOORD4; 
};

uniform float4 _MainTex_ST;

v2f vert(appdata_tan v)
{
	v2f o;
	PositionFog( v.vertex, o.pos, o.fog );
	o.normal = v.normal;
	o.uv = TRANSFORM_TEX(v.texcoord,_MainTex);
	o.lightDir = ObjSpaceLightDir( v.vertex );
	float3 viewDir = ObjSpaceViewDir( v.vertex );
	o.halfDir = normalize( normalize(o.lightDir) + normalize(viewDir) );
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
	PassTexCoord r0, t3.str;		# normal	
	SampleMap r2, t2.str;			# normalized light dir
	PassTexCoord r3, t4.str;		# half dir
	
	DOT3 r5.sat, r0, r2.2x.bias;	# diffuse (N.L)
	
	# Compute lookup UVs into specular falloff texture.
	# Normally it would be: r=sat(N.H), g=_Shininess*0.5
	# However, we'll use projective read on this to automatically
	# normalize H. Gives better precision in highlight.
	DOT3 r1.sat, r0, r3;			# N.H
	MUL  r1, r1, r1;				# (N.H)^2
	DOT3 r1.b.sat, r3, r3;         	# |H|^2
	MUL  r1.g, r1.b, c2.g; 			# |H|^2 * k
EndPass;

StartOutputPass;
	SampleMap r0, t0.str;			# main texture
	SampleMap r1, r1.str_dr;		# a = specular (projective to normalize H)
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
		}
		
		// Pixel lights with 1 light texture
		Pass {
			Blend SrcAlpha One
			Name "PPL"
			Tags { 
				"LightMode" = "Pixel" 
				"LightTexCount" = "1"
			}

CGPROGRAM
#pragma vertex vert
#include "UnityCG.cginc"

uniform float4 _MainTex_ST;
uniform float4x4 _SpotlightProjectionMatrix0;

struct v2f {
	V2F_POS_FOG;
	float2 uv		: TEXCOORD0;
	float3 normal	: TEXCOORD3;
	float3 lightDir	: TEXCOORD2;
	float3 halfDir	: TEXCOORD4; 
	float4 LightCoord0 : TEXCOORD1;
};

v2f vert(appdata_tan v)
{
	v2f o;
	PositionFog( v.vertex, o.pos, o.fog );
	o.normal = v.normal;
	o.uv = TRANSFORM_TEX(v.texcoord,_MainTex);
	o.lightDir = ObjSpaceLightDir( v.vertex );
	float3 viewDir = ObjSpaceViewDir( v.vertex );
	o.halfDir = normalize( normalize(o.lightDir) + normalize(viewDir) );
	
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
	PassTexCoord r0, t3.str;		# normal
	SampleMap r3, t2.str;			# normalized light dir
	PassTexCoord r4, t4.str;		# half angle

	DOT3 r5.sat, r0, r3.2x.bias;	# diffuse (N.L)
	
	# Compute lookup UVs into specular falloff texture.
	# Normally it would be: r=sat(N.H), g=_Shininess*0.5
	# However, we'll use projective read on this to automatically
	# normalize H. Gives better precision in highlight.
	DOT3 r2.sat, r0, r4;			# N.H
	MUL  r2, r2, r2;				# (N.H)^2
	DOT3 r2.b.sat, r4, r4;         	# |H|^2
	MUL  r2.g, r2.b, c2.g; 			# |H|^2 * k
EndPass;

StartOutputPass;
	SampleMap r0, t0.str;			# main texture
	SampleMap r1, t1.str;			# a = attenuation
	SampleMap r2, r2.str_dr;		# a = specular (projective to normalize H)
	PassTexCoord r5, r5.str;		# diffuse
	
	MUL r2, r2.a, r5.b;
	MUL r5.rgb, r5, c1; 			# modelLightColor.rgb * diffuse
	MUL r5.rgb, r5, r0;				# * texture
	MUL r2, r2, r0.a;				# spec *= gloss
	MUL r3, r2.a, c0;				# specColor * spec
	ADD r0.rgb.2x, r5, r3;			# (diff+spec)*2
	MUL r0.a, r0, c3;
	MUL r0.rgb, r0, r1.a;			# attenuate
EndPass; 
"
				}
			}
			SetTexture[_MainTex] {combine texture}
			SetTexture[_LightTexture0] {combine texture}
			SetTexture[_SpecFalloff] {combine texture}
			SetTexture[_CubeNormalize] {combine texture}
		}
		
		// Pixel lights with 2 light textures
		Pass { 
			Blend SrcAlpha One
			Name "PPL"
			Tags {
				"LightMode" = "Pixel"
				"LightTexCount" = "2"
			}
CGPROGRAM
#pragma vertex vert
#include "UnityCG.cginc"

uniform float4 _MainTex_ST;
uniform float4x4 _SpotlightProjectionMatrix0;
uniform float4x4 _SpotlightProjectionMatrixB0;

struct v2f {
	V2F_POS_FOG;
	float2 uv		: TEXCOORD0;
	float3 normal	: TEXCOORD2;
	float3 lightDir	: TEXCOORD3;
	float3 halfDir	: TEXCOORD4; 
	float4 LightCoord0 : TEXCOORD1;
	float4 LightCoordB0 : TEXCOORD5;
};

v2f vert(appdata_tan v)
{
	v2f o;
	PositionFog( v.vertex, o.pos, o.fog );
	o.normal = v.normal;
	o.uv = TRANSFORM_TEX(v.texcoord,_MainTex);
	o.lightDir = ObjSpaceLightDir( v.vertex );
	float3 viewDir = ObjSpaceViewDir( v.vertex );
	o.halfDir = normalize( normalize(o.lightDir) + normalize(viewDir) );
	
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
	PassTexCoord r0, t2.str;		# R0 = normal
	SampleMap r3, t3.str;			# R3 = normalized light dir
	PassTexCoord r1, t4.str;		# R1 = half angle

	DOT3 r5.sat, r0, r3.2x.bias;	# R5 = diffuse (N.L)
	
	# Compute lookup UVs into specular falloff texture.
	# Normally it would be: r=sat(N.H), g=_Shininess*0.5
	# However, we'll use projective read on this to automatically
	# normalize H. Gives better precision in highlight.
	DOT3 r2.sat, r0, r1;			# N.H
	MUL  r2, r2, r2;				# (N.H)^2
	DOT3 r2.b.sat, r1, r1;         	# |H|^2
	MUL  r2.g, r2.b, c2.g; 			# |H|^2 * k
EndPass;

StartOutputPass;
	SampleMap r0, t0.str;			# R0 = main texture
	SampleMap r1, t1.stq_dq;		# R1.a = attenuation 1
	SampleMap r2, r2.str_dr;		# R2.a = specular
	SampleMap r4, t5.stq_dq;		# R4.a = attenuation 2
	PassTexCoord r5, r5.str;		# R5 = diffuse
	
	MUL r2, r2.a, r5.b;
	MUL r5.rgb, r5, c1; 			# modelLightColor.rgb * diffuse
	MUL r5.rgb, r5, r0;				# * texture
	MUL r2, r2, r0.a;				# spec *= gloss
	MUL r3, r2.a, c0;				# specColor * spec
	ADD r0.rgb.2x, r5, r3;			# (diff+spec)*2
	MUL r0.a, r0, c3;
	MUL r0.rgb, r0, r1.a;			# attenuate 1
	MUL r0.rgb, r0, r4.a;			# attenuate 2
EndPass;
"
				}
			}
			SetTexture [_MainTex] {}
			SetTexture [_LightTexture0] {}
			SetTexture [_SpecFalloff]{combine previous}
			SetTexture [_CubeNormalize]{combine previous}
			SetTexture [_LightTextureB0] {}		
		}
	}
}

Fallback "Transparent/VertexLit", 1

}
