Shader "Parallax Specular" {
Properties {
	_Color ("Main Color", Color) = (1,1,1,1)
	_SpecColor ("Specular Color", Color) = (0.5, 0.5, 0.5, 1)
	_Shininess ("Shininess", Range (0.01, 1)) = 0.078125
	_Parallax ("Height", Range (0.005, 0.08)) = 0.02
	_MainTex ("Base (RGB) Gloss (A)", 2D) = "white" {}
	_BumpMap ("Bumpmap (RGB) Height (A)", 2D) = "bump" {}
}

Category {
	Tags { "RenderType"="Opaque" }
	LOD 600
	Blend AppSrcAdd AppDstAdd
	Fog { Color [_AddFog] }
	
	// ------------------------------------------------------------------
	// ARB fragment program
	
	SubShader {
		UsePass "Specular/BASE"
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
	float3	viewDirT;
	float2	uv2;
	float3	lightDirT;
}; 

uniform float4 _MainTex_ST, _BumpMap_ST;
uniform float _Shininess;

v2f vert (appdata_tan v)
{
	v2f o;
	PositionFog( v.vertex, o.pos, o.fog );
	o.uvK.xy = TRANSFORM_TEX(v.texcoord, _MainTex);
	o.uvK.z = _Shininess * 128;
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
	i.uvK.xy += offset;
	i.uv2 += offset;
	
	// get normal from the normal map
	float3 normal = tex2D(_BumpMap, i.uv2).xyz * 2 - 1;
		
	half4 texcol = tex2D(_MainTex,i.uvK.xy);
	
	return SpecularLight( i.lightDirT, i.viewDirT, normal, texcol, i.uvK.z, LIGHT_ATTENUATION(i) );
}
ENDCG

		}
	}
}

FallBack "Bumped Specular", 1

}
