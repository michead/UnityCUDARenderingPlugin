Shader "Hidden/TerrainEngine/Soft Occlusion Bark rendertex" {
	Properties {
		_Color ("Main Color", Color) = (1,1,1,0)
		_MainTex ("Main Texture", 2D) = "white" {  }
		_BaseLight ("BaseLight", range (0, 1)) = 0.35
		_AO ("Amb. Occlusion", range (0, 10)) = 2.4
		_Scale ("Scale", Vector) = (1,1,1,1)
	}
	SubShader {
		Fog { Mode Off }
		Pass {
			CGPROGRAM
			#pragma vertex bark
			#define WRITE_ALPHA_1 1
			#define USE_CUSTOM_LIGHT_DIR 1
			#include "SH_Vertex.cginc"
			ENDCG
			
			SetTexture [_MainTex] {
				combine primary * texture double, primary
			}
		}
	}
	
	Fallback Off
}