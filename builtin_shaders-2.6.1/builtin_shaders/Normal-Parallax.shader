Shader "Parallax Diffuse" {
Properties {
	_Color ("Main Color", Color) = (1,1,1,1)
	_Parallax ("Height", Range (0.005, 0.08)) = 0.02
	_MainTex ("Base (RGB)", 2D) = "white" {}
	_BumpMap ("Bumpmap (RGB) Height (A)", 2D) = "bump" {}
}

Category {
	Tags { "RenderType"="Opaque" }
	LOD 500
	Blend AppSrcAdd AppDstAdd
	Fog { Color [_AddFog] }
	
	// ------------------------------------------------------------------
	// ARB fragment program
	
	SubShader {
		UsePass "Diffuse/BASE"
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
	o.uv = TRANSFORM_TEX(v.texcoord, _MainTex);
	o.uv2 = TRANSFORM_TEX(v.texcoord, _BumpMap);

	TANGENT_SPACE_ROTATION;
	o.lightDirT = mul( rotation, ObjSpaceLightDir( v.vertex ) );	
	o.viewDirT = mul( rotation, ObjSpaceViewDir( v.vertex ) );	

	TRANSFER_VERTEX_TO_FRAGMENT(o);	
	return o;
}


uniform sampler2D _BumpMap;
uniform sampler2D _MainTex;
uniform float _Parallax;

float4 frag (v2f i) : COLOR
{
	half h = tex2D( _BumpMap, i.uv2 ).w;
	float2 offset = ParallaxOffset( h, _Parallax, i.viewDirT );
	i.uv += offset;
	i.uv2 += offset;
	
	// get normal from the normal map
	half3 normal = tex2D(_BumpMap, i.uv2).xyz * 2 - 1;
		
	half4 texcol = tex2D(_MainTex,i.uv);
	
	return DiffuseLight( i.lightDirT, normal, texcol, LIGHT_ATTENUATION(i) );
}
ENDCG  
		}
	}
}

FallBack "Bumped Diffuse", 1

}
