Shader "Diffuse Detail" {
Properties {
	_Color ("Main Color", Color) = (1,1,1,1)
	_MainTex ("Base (RGB)", 2D) = "white" {}
	_Detail ("Detail (RGB)", 2D) = "gray" {}
}

Category {
	Tags { "RenderType"="Opaque" }
	LOD 250
	Blend AppSrcAdd AppDstAdd
	Fog { Color [_AddFog] }
		
	// ------------------------------------------------------------------
	// ARB fragment program
	
	SubShader {
		// Ambient pass
		Pass {
			Tags {"LightMode" = "PixelOrNone"}
			Color [_PPLAmbient]
			SetTexture [_MainTex] {constantColor [_Color] Combine texture * primary DOUBLE, texture * constant}
			SetTexture [_Detail] {combine previous * texture DOUBLE, previous}
		}
		
		// Vertex lights
		Pass { 
			Tags {"LightMode" = "Vertex"}
			Material {
				Diffuse [_Color]
				Emission [_PPLAmbient]
			}
			Lighting On
			SetTexture [_MainTex] {constantColor [_PPLAmbent] Combine texture * primary DOUBLE, texture * primary}
			SetTexture [_Detail] { combine previous * texture DOUBLE, previous }
		}
		
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
	float2	uv[2];
	float3	normal;
	float3	lightDir;
};

uniform float4 _MainTex_ST, _Detail_ST;

v2f vert (appdata_base v)
{
	v2f o;
	PositionFog( v.vertex, o.pos, o.fog );
	o.normal = v.normal;
	o.uv[0] = TRANSFORM_TEX(v.texcoord,_MainTex);
	o.uv[1] = TRANSFORM_TEX(v.texcoord,_Detail);
	o.lightDir = ObjSpaceLightDir( v.vertex );
	TRANSFER_VERTEX_TO_FRAGMENT(o);
	return o;
}

uniform sampler2D _MainTex;
uniform sampler2D _Detail;

half4 frag (v2f i) : COLOR
{
	half4 texcol = tex2D(_MainTex,i.uv[0]);
	texcol.rgb *= tex2D(_Detail,i.uv[1]).rgb*2;
	
	return DiffuseLight( i.lightDir, i.normal, texcol, LIGHT_ATTENUATION(i) );
} 
ENDCG
		}
	}
	
	// ------------------------------------------------------------------
	// Radeon 7000/9000
	
	Category {
		Material {
			Diffuse [_Color]
			Emission [_PPLAmbient]
		}
		Lighting On
		Fog { Color [_AddFog] }
		Blend AppSrcAdd AppDstAdd
		SubShader {
			// Ambient pass
			Pass {
				Tags {"LightMode" = "PixelOrNone"}
				Color [_PPLAmbient]
				Lighting Off
				SetTexture [_MainTex] {constantColor [_Color] Combine texture * primary DOUBLE, texture * constant}
				SetTexture [_Detail] {combine previous * texture DOUBLE, previous}
			}
			
			// Vertex lights
			Pass { 
				Tags {"LightMode" = "Vertex"}
				Lighting On
				Material {
					Diffuse [_Color]
					Emission [_PPLAmbient]
				}
				SetTexture [_MainTex] {constantColor [_PPLAmbent] Combine texture * primary DOUBLE, texture * primary}
				SetTexture [_Detail] {combine previous * texture DOUBLE, previous}
			}
			
			// Pixel lights with 2 light textures
			Pass {
				Tags {
					"LightMode" = "Pixel"
					"LightTexCount" = "2"
				}
				ColorMask RGB
				SetTexture [_LightTexture0] 	{ combine previous * texture alpha, previous }
				SetTexture [_LightTextureB0]	{ combine previous * texture alpha, previous }
				SetTexture[_Detail]		{ combine previous * texture DOUBLE, previous }
				SetTexture[_MainTex] 	{ combine previous * texture DOUBLE }
			}
			// Pixel lights with 1 light texture
			Pass {
				Tags {
					"LightMode" = "Pixel"
					"LightTexCount"  = "1"
				}
				ColorMask RGB
				SetTexture [_LightTexture0] { combine previous * texture alpha, previous }
				SetTexture[_Detail]		{ combine previous * texture, previous }
				SetTexture[_MainTex] 	{ combine previous * texture DOUBLE }
			}
			// Pixel lights with 0 light textures
			Pass {
				Tags {
					"LightMode" = "Pixel"
					"LightTexCount"  = "0"
				}
				ColorMask RGB
				SetTexture[_Detail]		{ combine previous * texture, previous }
				SetTexture [_MainTex] 	{ combine previous * texture DOUBLE }
			}
		}
	}
}

// Fallback to vertex lit
Fallback "VertexLit", 2

}
