Shader "Hidden/TerrainEngine/Soft Occlusion Leaves rendertex" {
	Properties {
		_Color ("Main Color", Color) = (1,1,1,0)
		_MainTex ("Main Texture", 2D) = "white" { }
		_HalfOverCutoff ("0.5 / alpha cutoff", Range(0,1)) = 1.0
		_BaseLight ("BaseLight", range (0, 1)) = 0.35
		_AO ("Amb. Occlusion", range (0, 10)) = 2.4
		_Occlusion ("Dir Occlusion", range (0, 20)) = 7.5
		_Scale ("Scale", Vector) = (1,1,1,1)
	}
	SubShader {
		CGINCLUDE
		#pragma vertex leaves
		#define USE_CUSTOM_LIGHT_DIR 1
		#include "SH_Vertex.cginc"
		ENDCG

		Tags { "Queue" = "Transparent-99" }
		Cull Off
		Fog { Mode Off}
		
		Pass {
			CGPROGRAM
			ENDCG
			ZWrite On
			// Here we want to do alpha testing on cutoff, but at the same
			// time write 1.0 into alpha. So we multiply alpha by 1/cutoff
			// and alpha test on alpha being 1.0
			AlphaTest GEqual 1.0
			SetTexture [_MainTex] {
				constantColor(0,0,0,[_HalfOverCutoff])
				combine primary * texture double, texture * constant DOUBLE
			}
		}
		
		Pass {
			Tags { "RequireOptions" = "SoftVegetation" }
			CGPROGRAM
			ENDCG

			Blend One OneMinusSrcAlpha
			ZWrite Off
			SetTexture [_MainTex] { combine primary * texture double, texture }
			SetTexture [_MainTex] { combine previous alpha * previous, previous }
		}
	}
	
	Fallback Off
}
