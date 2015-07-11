Shader "Transparent/Cutout/VertexLit" {
Properties {
	_Color ("Main Color", Color) = (1,1,1,1)
	_SpecColor ("Spec Color", Color) = (1,1,1,0)
	_Emission ("Emissive Color", Color) = (0,0,0,0)
	_Shininess ("Shininess", Range (0.1, 1)) = 0.7
	_MainTex ("Base (RGB) Trans (A)", 2D) = "white" {}
	_Cutoff ("Alpha cutoff", Range(0,1)) = 0.5
}

SubShader {
	Tags {"IgnoreProjector"="True" "RenderType"="TransparentCutout"}
	LOD 100
	Pass {
		Alphatest Greater [_Cutoff]
		AlphaToMask True
		ColorMask RGB
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
			Combine texture * primary DOUBLE, texture * primary 
		} 
	}
	
	// Pass to render object as a shadow caster
	Pass {
		Name "Caster"
		Tags { "LightMode" = "ShadowCaster" }
		
		Fog {Mode Off}
		ZWrite On ZTest Less Cull Off
		Offset [_ShadowBias], [_ShadowBiasSlope]

CGPROGRAM
#pragma vertex vert
#pragma fragment frag
#pragma multi_compile SHADOWS_NATIVE SHADOWS_CUBE
#pragma fragmentoption ARB_precision_hint_fastest
#include "UnityCG.cginc"

struct v2f { 
	V2F_SHADOW_CASTER;
	float2  uv;
};

uniform float4 _MainTex_ST;

v2f vert( appdata_base v )
{
	v2f o;
	TRANSFER_SHADOW_CASTER(o)
	o.uv = TRANSFORM_TEX(v.texcoord, _MainTex);
	return o;
}

uniform sampler2D _MainTex;
uniform float _Cutoff;
uniform float4 _Color;

float4 frag( v2f i ) : COLOR
{
	half4 texcol = tex2D( _MainTex, i.uv );
	clip( texcol.a*_Color.a - _Cutoff );
	
	SHADOW_CASTER_FRAGMENT(i)
}
ENDCG

	}
	
	// Pass to render object as a shadow collector
	Pass {
		Name "ShadowCollector"
		Tags { "LightMode" = "ShadowCollector" }
		
		Fog {Mode Off}
		ZWrite On ZTest Less

CGPROGRAM
#pragma vertex vert
#pragma fragment frag
#pragma fragmentoption ARB_precision_hint_fastest

#define SHADOW_COLLECTOR_PASS
#include "UnityCG.cginc"

struct v2f {
	V2F_SHADOW_COLLECTOR;
	float2  uv;
};

uniform float4 _MainTex_ST;

v2f vert (appdata_base v)
{
	v2f o;
	TRANSFER_SHADOW_COLLECTOR(o)
	o.uv = TRANSFORM_TEX(v.texcoord, _MainTex);
	return o;
}

uniform sampler2D _MainTex;
uniform float _Cutoff;
uniform float4 _Color;

half4 frag (v2f i) : COLOR
{
	half4 texcol = tex2D( _MainTex, i.uv );
	clip( texcol.a*_Color.a - _Cutoff );
	
	SHADOW_COLLECTOR_FRAGMENT(i)
}
ENDCG

	}
}
}