Shader "Nature/Soft Occlusion Leaves" {
	Properties {
		_Color ("Main Color", Color) = (1,1,1,1)
		_MainTex ("Main Texture", 2D) = "white" {  }
		_Cutoff ("Base Alpha cutoff", Range (.5,.9)) = .5
		_BaseLight ("BaseLight", range (0, 1)) = 0.35
		_AO ("Amb. Occlusion", range (0, 10)) = 2.4
		_Occlusion ("Dir Occlusion", range (0, 20)) = 7.5
		_Scale ("Scale", Vector) = (1,1,1,1)
	}
	SubShader {
		Tags {
			"Queue" = "Transparent-99"
			"IgnoreProjector"="True"
			"BillboardShader" = "Hidden/TerrainEngine/Soft Occlusion Leaves rendertex"
			"RenderType" = "TreeTransparentCutout"
		}
		Cull Off
		ColorMask RGB
		
		Pass {
			CGPROGRAM
			#pragma vertex leaves
			#include "SH_Vertex.cginc"
			ENDCG

			AlphaTest GEqual [_Cutoff]
			ZWrite On
			
			SetTexture [_MainTex] { combine primary * texture DOUBLE, texture }
		}
		
		Pass {
			Tags { "RequireOptions" = "SoftVegetation" }
			CGPROGRAM
			#pragma vertex leaves
			#include "SH_Vertex.cginc"
			ENDCG
			// the texture is premultiplied alpha!
			Blend SrcAlpha OneMinusSrcAlpha
			ZWrite Off

			SetTexture [_MainTex] { combine primary * texture DOUBLE, texture }
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
			#include "TerrainEngine.cginc"
			
			struct v2f { 
				V2F_SHADOW_CASTER;
				float2  uv;
			};
			
			struct appdata {
			    float4 vertex : POSITION;
			    float4 color : COLOR;
			    float4 texcoord : TEXCOORD0;
			};
			v2f vert( appdata v )
			{
				v2f o;
				TerrainAnimateTree(v.vertex, v.color.w);
				TRANSFER_SHADOW_CASTER(o)
				o.uv = v.texcoord;
				return o;
			}
			
			uniform sampler2D _MainTex;
			uniform float _Cutoff;
					
			float4 frag( v2f i ) : COLOR
			{
				half4 texcol = tex2D( _MainTex, i.uv );
				clip( texcol.a - _Cutoff );
				SHADOW_CASTER_FRAGMENT(i)
			}
			ENDCG	
		}
	}
	
	SubShader {
		Tags {
			"Queue" = "Transparent-99"
			"IgnoreProjector"="True"
			"BillboardShader" = "Hidden/TerrainEngine/Soft Occlusion Leaves rendertex"
			"RenderType" = "TransparentCutout"
		}
		Cull Off
		ColorMask RGB
		Pass {
			AlphaTest GEqual [_Cutoff]
			Lighting On
			Material {
				Diffuse [_Color]
				Ambient [_Color]
			}
			SetTexture [_MainTex] { combine primary * texture DOUBLE, texture }
		}		
	}
	
	Fallback Off
}
