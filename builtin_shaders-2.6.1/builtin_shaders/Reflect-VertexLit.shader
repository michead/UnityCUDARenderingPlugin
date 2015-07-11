Shader "Reflective/VertexLit" {
Properties {
	_Color ("Main Color", Color) = (1,1,1,1)
	_SpecColor ("Spec Color", Color) = (1,1,1,1)
	_Shininess ("Shininess", Range (0.03, 1)) = 0.7
	_ReflectColor ("Reflection Color", Color) = (1,1,1,0.5)
	_MainTex ("Base (RGB) RefStrength (A)", 2D) = "white" {} 
	_Cube ("Reflection Cubemap", Cube) = "_Skybox" { TexGen CubeReflect }
}

Category {
	Tags { "RenderType"="Opaque" }
	LOD 150
	Blend AppSrcAdd AppDstAdd
	Fog { Color [_AddFog] }

	// ------------------------------------------------------------------
	// ARB fragment program
	
	SubShader {
	
		// First pass does reflection cubemap
		Pass { 
			Name "BASE"
			Tags {"LightMode" = "Always"}
CGPROGRAM
#pragma vertex vert
#pragma fragment frag
#pragma fragmentoption ARB_fog_exp2
#pragma fragmentoption ARB_precision_hint_fastest
#include "UnityCG.cginc"

struct v2f {
	V2F_POS_FOG;
	float2 uv : TEXCOORD0;
	float3 I : TEXCOORD1;
};

uniform float4 _MainTex_ST;

v2f vert(appdata_tan v)
{
	v2f o;
	PositionFog( v.vertex, o.pos, o.fog );
	o.uv = TRANSFORM_TEX(v.texcoord,_MainTex);

	// calculate object space reflection vector	
	float3 viewDir = ObjSpaceViewDir( v.vertex );
	float3 I = reflect( -viewDir, v.normal );
	
	// transform to world space reflection vector
	o.I = mul( (float3x3)_Object2World, I );
	
	return o; 
}

uniform sampler2D _MainTex;
uniform samplerCUBE _Cube;
uniform float4 _ReflectColor;

half4 frag (v2f i) : COLOR
{
	half4 texcol = tex2D (_MainTex, i.uv);
	half4 reflcol = texCUBE( _Cube, i.I );
	reflcol *= texcol.a;
	return reflcol * _ReflectColor;
} 
ENDCG
		}
		
		// Second pass adds vertex lighting
		Pass {
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

struct v2f {
	float2 uv : TEXCOORD0;
	float4 diff : COLOR0;
	float4 spec : COLOR1;
};

uniform sampler2D _MainTex : register(s0);
uniform float4 _ReflectColor;
uniform float4 _SpecColor;

half4 frag (v2f i) : COLOR
{
	half4 temp = tex2D (_MainTex, i.uv);	
	half4 c;
	c.xyz = (temp.xyz * i.diff.xyz + temp.w * i.spec.xyz ) * 2;
	c.w = temp.w * (i.diff.w + Luminance(i.spec.xyz) * _SpecColor.a);
	return c;
} 
ENDCG
			SetTexture[_MainTex] {}
		}		
	}
	
	// ------------------------------------------------------------------
	// Radeon 9000
	
	SubShader {
	
		// First pass does reflection cubemap
		Pass { 
			Name "BASE"
			Tags {"LightMode" = "Always"}
CGPROGRAM
#pragma vertex vert
#include "UnityCG.cginc"

struct v2f {
	V2F_POS_FOG;
	float2 uv : TEXCOORD0;
	float3 I : TEXCOORD1;
};

uniform float4 _MainTex_ST;

v2f vert(appdata_tan v)
{
	v2f o;
	PositionFog( v.vertex, o.pos, o.fog );
	o.uv = TRANSFORM_TEX(v.texcoord,_MainTex);

	// calculate object space reflection vector	
	float3 viewDir = ObjSpaceViewDir( v.vertex );
	float3 I = reflect( -viewDir, v.normal );
	
	// transform to world space reflection vector
	o.I = mul( (float3x3)_Object2World, I );
	
	return o; 
}
ENDCG
			Program "" {
				SubProgram {
					Local 0, [_ReflectColor]
					"!!ATIfs1.0
					StartConstants;
						CONSTANT c0 = program.local[0];
					EndConstants;
					StartOutputPass;
						SampleMap r0, t0.str;
						SampleMap r1, t1.str;
						MUL r1, r1, r0.a;
						MUL r0, r1, c0;
					EndPass;
					"
				}
			}
			SetTexture [_MainTex] {combine texture}
			SetTexture [_Cube] {combine texture}
		}
		
		// Second pass adds vertex lighting
		Pass {
			Material {
				Diffuse [_Color]
				Ambient [_Color]
				Shininess [_Shininess]
				Specular [_SpecColor]
				Emission [_Emission]
			}
			Lighting On
			SeparateSpecular On
			SetTexture [_MainTex] {
				constantColor [_Color]
				Combine texture * previous DOUBLE, texture * constant
			}
		}
	}

	// ------------------------------------------------------------------
	// Old cards
	
	SubShader {
		Pass { 
			Name "BASE"
			Tags {"LightMode" = "Always"}
			Material {
				Diffuse [_Color]
				Ambient (1,1,1,1)
				Shininess [_Shininess]
				Specular [_SpecColor]
			}
			Lighting On
			SeparateSpecular on
			SetTexture [_MainTex] {
				combine texture * primary DOUBLE, texture * primary
			}
			SetTexture [_Cube] {
				combine texture * previous alpha + previous, previous
			}
		}
	}
}

// Fallback for cards that don't do cubemapping
FallBack "VertexLit", 1

}
