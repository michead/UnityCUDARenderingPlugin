Shader "Hidden/TerrainEngine/Details/WavingDoublePass" {
	Properties {
		_WavingTint ("Fade Color", Color) = (.7,.6,.5, 0)
		_MainTex ("Base (RGB) Alpha (A)", 2D) = "white" {}
		_WaveAndDistance ("Wave and distance", Vector) = (12, 3.6, 1, 1)
		_Cutoff ("Cutoff", float) = 0.5
	}
	SubShader {
		Tags {
			"Queue" = "Transparent-101"
			"IgnoreProjector"="True"
			"RenderType"="Grass"
		}

		ColorMask RGB
		Cull Off
		
		Pass {
			CGPROGRAM
			#pragma vertex vert
			#pragma multi_compile NO_INTEL_GMA_X3100_WORKAROUND INTEL_GMA_X3100_WORKAROUND
			#include "WavingGrass.cginc"
			ENDCG			

			AlphaTest Greater [_Cutoff]

			SetTexture [_MainTex] { combine texture * primary DOUBLE, texture * primary}
		}
		Pass {
			Tags { "RequireOptions" = "SoftVegetation" }
			CGPROGRAM
			#pragma vertex vert
			#pragma multi_compile NO_INTEL_GMA_X3100_WORKAROUND INTEL_GMA_X3100_WORKAROUND
			#include "WavingGrass.cginc"
			ENDCG			

			// Dont write to the depth buffer
			ZWrite off
			
			// Only render pixels less or equal to the value
			AlphaTest LEqual [_Cutoff]
			
			// Set up alpha blending
			Blend SrcAlpha OneMinusSrcAlpha
 
			SetTexture [_MainTex] { combine texture * primary DOUBLE, texture * primary } 
		 }
	}
	SubShader {
		Tags {
			"Queue" = "Transparent-101"
			"IgnoreProjector"="True"
			"RenderType"="Grass"
		}

		ColorMask RGB
		Cull Off
		
		Pass {
			BindChannels {
				Bind "vertex", vertex
				Bind "color", color
				Bind "texcoord", texcoord
			}
			AlphaTest Greater [_Cutoff]
			SetTexture [_MainTex] { combine texture * primary DOUBLE, texture }
		}
	}
	
	Fallback Off
	
}
