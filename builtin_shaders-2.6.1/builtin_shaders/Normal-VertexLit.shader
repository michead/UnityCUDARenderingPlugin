Shader "VertexLit" {
Properties {
	_Color ("Main Color", Color) = (1,1,1,1)
	_SpecColor ("Spec Color", Color) = (1,1,1,1)
	_Emission ("Emissive Color", Color) = (0,0,0,0)
	_Shininess ("Shininess", Range (0.01, 1)) = 0.7
	_MainTex ("Base (RGB)", 2D) = "white" {}
}

SubShader {
	Tags { "RenderType"="Opaque" }
	LOD 100
	
	// Normal rendering pass
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
			Combine texture * primary DOUBLE, texture * primary
		} 
	}
	
	// Pass to render object as a shadow caster
	Pass {
		Name "ShadowCaster"
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
};

v2f vert( appdata_base v )
{
	v2f o;
	TRANSFER_SHADOW_CASTER(o)
	return o;
}

float4 frag( v2f i ) : COLOR
{
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

struct appdata {
	float4 vertex;
};

struct v2f {
	V2F_SHADOW_COLLECTOR;
};

v2f vert (appdata v)
{
	v2f o;
	TRANSFER_SHADOW_COLLECTOR(o)
	return o;
}

half4 frag (v2f i) : COLOR
{
	SHADOW_COLLECTOR_FRAGMENT(i)
}
ENDCG

	}
}
}
