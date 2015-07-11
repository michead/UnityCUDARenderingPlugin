Shader "Hidden/TerrainEngine/Splatmap/Realtime-AddPass" {
Properties {
	_Control ("Control (RGBA)", 2D) = "black" {}
	_LightMap ("LightMap (RGB)", 2D) = "white" {}
	_Splat3 ("Layer 3 (A)", 2D) = "white" {}
	_Splat2 ("Layer 2 (B)", 2D) = "white" {}
	_Splat1 ("Layer 1 (G)", 2D) = "white" {}
	_Splat0 ("Layer 0 (R)", 2D) = "white" {}
}
	
SubShader {		
	Tags {
		"SplatCount" = "4"
		"Queue" = "Geometry-99"
		"IgnoreProjector"="True"
		"RenderType" = "Transparent"
	}
	
	Blend One One
	ZWrite Off
	Fog { Color (0,0,0,0) }
	
	// Ambient pass
	Pass {
		Tags { "LightMode" = "PixelOrNone" }
		
		CGPROGRAM
		#pragma vertex AmbientSplatVertex
		#pragma fragment AmbientSplatFragment
		#pragma fragmentoption ARB_fog_exp2
		#pragma fragmentoption ARB_precision_hint_fastest

		#define USE_LIGHTMAP	
		#include "splatting.cginc"
		ENDCG
	}
	// Vertex lights
	Pass {
		Tags { "LightMode" = "Vertex" }
		
		CGPROGRAM
		#pragma vertex VertexlitSplatVertex
		#pragma fragment VertexlitSplatFragment
		#pragma fragmentoption ARB_fog_exp2
		#pragma fragmentoption ARB_precision_hint_fastest

		#define USE_LIGHTMAP	
		#include "splatting.cginc"
		ENDCG
	}
	// Pixel lights
	Pass {
		Tags { "LightMode" = "Pixel" }
		
		CGPROGRAM
		#pragma vertex PixellitSplatVertex
		#pragma fragment PixellitSplatFragment
		#pragma fragmentoption ARB_fog_exp2
		#pragma fragmentoption ARB_precision_hint_fastest
		#pragma multi_compile_builtin
		
		#include "UnityCG.cginc"
		#include "AutoLight.cginc"
		#define INCLUDE_PIXEL
		#include "splatting.cginc"
		ENDCG
	}
}
 	
// Fallback to VertexLit
Fallback "Hidden/TerrainEngine/Splatmap/Vertexlit-AddPass"
}
