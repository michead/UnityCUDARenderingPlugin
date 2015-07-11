Shader "Lightmapped/Bumped Diffuse" {
	Properties {
		_Color ("Main Color", Color) = (1,1,1,1)
		_MainTex ("Base (RGB)", 2D) = "white" {}
		_BumpMap ("Bump (RGB)", 2D) = "bump" {}
		_LightMap ("Lightmap (RGB)", 2D) = "lightmap" { LightmapMode }
	}
	SubShader {
		LOD 300
		Tags { "RenderType"="Opaque" }
		UsePass "Lightmapped/VertexLit/BASE"
		UsePass "Bumped Diffuse/PPL"
	} 
	FallBack "Lightmapped/VertexLit", 1
}