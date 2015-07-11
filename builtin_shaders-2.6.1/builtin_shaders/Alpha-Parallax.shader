Shader "Transparent/Parallax Diffuse" {
Properties {
	_Color ("Main Color", Color) = (1,1,1,1)
	_Parallax ("Height", Range (0.005, 0.08)) = 0.02
	_MainTex ("Base (RGB) Trans (A)", 2D) = "white" {}
	_BumpMap ("Bumpmap (RGB) Height (A)", 2D) = "bump" {}
}

Category {
	Tags {"Queue"="Transparent" "IgnoreProjector"="True" "RenderType"="Transparent"}
	LOD 500
	Alphatest Greater 0
	Fog { Color [_AddFog] }
	ZWrite Off
	ColorMask RGB
	
	// ------------------------------------------------------------------
	// ARB fragment program
	
	SubShader {
		UsePass "Transparent/Diffuse/BASE"
		// Pixel lights
		Pass {	
			Name "PPL"
			Blend SrcAlpha One
			Tags { "LightMode" = "Pixel" }
				
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
	float2	uv;
	float3	viewDirT;
	float2	uv2;
	float3	lightDirT;
}; 

uniform float4 _MainTex_ST, _BumpMap_ST;

v2f vert (appdata_tan v)
{
	v2f o;
	PositionFog( v.vertex, o.pos, o.fog );
	o.uv = TRANSFORM_TEX(v.texcoord,_MainTex);
	o.uv2 = TRANSFORM_TEX(v.texcoord,_BumpMap);

	TANGENT_SPACE_ROTATION;
	o.lightDirT = mul( rotation, ObjSpaceLightDir( v.vertex ) );	
	o.viewDirT = mul( rotation, ObjSpaceViewDir( v.vertex ) );	

	TRANSFER_VERTEX_TO_FRAGMENT(o);	
	return o;
}

uniform sampler2D _BumpMap;
uniform sampler2D _MainTex;
uniform float4 _SpecColor; 
uniform float _Parallax;
uniform float4 _Color;

float4 frag (v2f i) : COLOR
{
	half h = tex2D( _BumpMap, i.uv2 ).w;
	float2 offset = ParallaxOffset( h, _Parallax, i.viewDirT );
	i.uv += offset;
	i.uv2 += offset;
	
	// get normal from the normal map
	half3 normal = tex2D(_BumpMap, i.uv2).xyz * 2 - 1;
		
	half4 texcol = tex2D(_MainTex,i.uv);
	
	half4 c = DiffuseLight( i.lightDirT, normal, texcol, LIGHT_ATTENUATION(i) );
	c.a = texcol.a * _Color.a;
	return c;
}

ENDCG  
		}
	}
}

FallBack "Transparent/Bumped Diffuse", 1

}