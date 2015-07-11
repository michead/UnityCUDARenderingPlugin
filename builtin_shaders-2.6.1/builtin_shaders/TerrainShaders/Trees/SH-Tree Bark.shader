Shader "Nature/Soft Occlusion Bark" {
	Properties {
		_Color ("Main Color", Color) = (1,1,1,0)
		_MainTex ("Main Texture", 2D) = "white" {  }
		_BaseLight ("BaseLight", range (0, 1)) = 0.35
		_AO ("Amb. Occlusion", range (0, 10)) = 2.4
		_Scale ("Scale", Vector) = (1,1,1,1)
	}
	SubShader {
		Tags {
			"IgnoreProjector"="True"
			"BillboardShader" = "Hidden/TerrainEngine/Soft Occlusion Bark rendertex"
			"RenderType" = "TreeOpaque"
		}

		Pass {
			CGPROGRAM
			#pragma vertex bark
			#include "SH_Vertex.cginc"
			ENDCG
						
			SetTexture [_MainTex] { combine primary * texture DOUBLE, constant }
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
			};
			
			struct appdata {
			    float4 vertex : POSITION;
			    float4 color : COLOR;
			};
			v2f vert( appdata v )
			{
				v2f o;
				TerrainAnimateTree(v.vertex, v.color.w);
				TRANSFER_SHADOW_CASTER(o)
				return o;
			}
			
			float4 frag( v2f i ) : COLOR
			{
				SHADOW_CASTER_FRAGMENT(i)
			}
			ENDCG	
		}
	}
	SubShader {
		Tags {
			"IgnoreProjector"="True"
			"BillboardShader" = "Hidden/TerrainEngine/Soft Occlusion Bark rendertex"
			"RenderType" = "Opaque"
		}
		Pass {
			Lighting On
			Material {
				Diffuse [_Color]
				Ambient [_Color]
			}
			SetTexture [_MainTex] { combine primary * texture DOUBLE, constant }
		}		
	}
	
	Fallback Off
}
