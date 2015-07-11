Shader "Diffuse Fast" {
Properties {
	_Color ("Main Color", Color) = (1,1,1,1)
	_MainTex ("Base (RGB)", 2D) = "white" {}
}

// Calculates lighting per vertex, but applies
// light attenuation maps or spot cookies per pixel.
// Quite fine for tesselated geometry.

Category {
	Tags { "RenderType"="Opaque" }
	LOD 150
	Blend AppSrcAdd AppDstAdd
	Fog { Color [_AddFog] }
	
	// ------------------------------------------------------------------
	// ARB fragment program
	
	SubShader {
		// Ambient pass
		Pass {
			Name "BASE"
			Tags {"LightMode" = "PixelOrNone"}
			Color [_PPLAmbient]
			SetTexture [_MainTex] {constantColor [_Color] Combine texture * primary DOUBLE, texture * constant}
		}
		// Vertex lights
		Pass { 
			Name "BASE"
			Tags {"LightMode" = "Vertex"}
			Lighting On
			Material {
				Diffuse [_Color]
				Emission [_PPLAmbient]
			}
			SetTexture [_MainTex] { constantColor [_Color] Combine texture * primary DOUBLE, texture * constant}
		}
		// Pixel lights
		Pass {
			Name "PPL"
			Tags { "LightMode" = "Pixel" }
			Material { Diffuse [_Color] }
			Lighting On

CGPROGRAM
// TODO: need vertex-lighting emulation here!
#pragma fragment frag
#pragma multi_compile_builtin_noshadows
#pragma fragmentoption ARB_fog_exp2
#pragma fragmentoption ARB_precision_hint_fastest

#include "UnityCG.cginc"

#ifdef POINT
#define LIGHTING_COORDS float3 _LightCoord : TEXCOORD1;
uniform sampler3D _LightTexture0 : register(s1);
#ifdef SHADER_API_D3D9
#define LIGHT_ATTENUATION(a)	(tex3D(_LightTexture0, a._LightCoord).r)
#else
#define LIGHT_ATTENUATION(a)	(tex3D(_LightTexture0, a._LightCoord).w)
#endif
#endif


#ifdef SPOT
#define LIGHTING_COORDS float4 _LightCoord : TEXCOORD1; float4 _LightCoord2 : TEXCOORD2;
uniform sampler2D _LightTexture0 : register(s1);
uniform sampler2D _LightTextureB0 : register(s2);
#define LIGHT_ATTENUATION(a)	(tex2Dproj (_LightTexture0, a._LightCoord.xyz).w * tex2D(_LightTextureB0, a._LightCoord2.xy).w) 
#endif 


#ifdef DIRECTIONAL
#define LIGHTING_COORDS 
#define LIGHT_ATTENUATION(a)	1.0
#endif


#ifdef POINT_NOATT
#define LIGHTING_COORDS 
#define LIGHT_ATTENUATION(a)	1.0
#endif


#ifdef POINT_COOKIE
#define LIGHTING_COORDS float3 _LightCoord : TEXCOORD1;
uniform samplerCUBE _LightTexture0 : register(s1);
#define LIGHT_ATTENUATION(a)	(texCUBE(_LightTexture0, a._LightCoord).w)
#endif


#ifdef DIRECTIONAL_COOKIE
#define LIGHTING_COORDS float2 _LightCoord : TEXCOORD1;
uniform sampler2D _LightTexture0 : register(s1);
#define LIGHT_ATTENUATION(a)	(tex2D(_LightTexture0, a._LightCoord).w)
#endif


struct v2f {
	float2 uv;
	LIGHTING_COORDS
	float4 diff : COLOR0;
};  

uniform sampler2D _MainTex : register(s0);

half4 frag (v2f i) : COLOR
{
	half4 texcol = tex2D( _MainTex, i.uv );
	half4 c;
	c.xyz = texcol.xyz * i.diff.xyz * (LIGHT_ATTENUATION(i) * 2);
	c.w = 0;
	return c;
} 
ENDCG
			SetTexture[_MainTex] {}
			SetTexture[_LightTexture0] {}
			SetTexture[_LightTextureB0] {}
		}
	}

	// ------------------------------------------------------------------
	// Radeon 7000 / 9000
	
	SubShader {
		Material {
			Diffuse [_Color]
			Emission [_PPLAmbient]
		}
		Lighting On
		// Ambient pass
		Pass {
			Name "BASE"
			Tags {"LightMode" = "PixelOrNone"}
			Color [_PPLAmbient]
			Lighting Off
			SetTexture [_MainTex] {Combine texture * primary DOUBLE, primary * texture}
		}
		// Vertex lights
		Pass {
			Name "BASE"
			Tags {"LightMode" = "Vertex"}
			SetTexture [_MainTex] {Combine texture * primary DOUBLE, primary * texture}
		}
		// Pixel lights with 2 light textures
		Pass {
			Name "PPL"
			Tags {
				"LightMode" = "Pixel"
				"LightTexCount"  = "2"
			}
			ColorMask RGB
			SetTexture [_LightTexture0] 	{ combine previous * texture alpha, previous }
			SetTexture [_LightTextureB0]	{ combine previous * texture alpha, previous }
			SetTexture [_MainTex] {combine previous * texture DOUBLE}
		}
		// Pixel lights with 1 light texture
		Pass {
			Name "PPL"
			Tags {
				"LightMode" = "Pixel"
				"LightTexCount"  = "1"
			}
			ColorMask RGB
			SetTexture [_LightTexture0] { combine previous * texture alpha, previous }
			SetTexture [_MainTex] { combine previous * texture DOUBLE }
		}
		// Pixel lights with 0 light textures
		Pass {
			Name "PPL"
			Tags {
				"LightMode" = "Pixel"
				"LightTexCount" = "0"
			}
			ColorMask RGB
			SetTexture[_MainTex] { combine previous * texture DOUBLE }
		}
	}
}

Fallback "VertexLit", 2

}