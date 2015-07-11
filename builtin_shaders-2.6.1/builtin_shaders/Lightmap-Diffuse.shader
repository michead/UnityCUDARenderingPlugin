Shader "Lightmapped/Diffuse" {
	Properties {
		_Color ("Main Color", Color) = (1,1,1,1)
		_MainTex ("Base (RGB)", 2D) = "white" {}
		_LightMap ("Lightmap (RGB)", 2D) = "lightmap" { LightmapMode }
	}
	SubShader {
		LOD 200
		Tags { "RenderType"="Opaque" }
		UsePass "Lightmapped/VertexLit/BASE"
		UsePass "Diffuse/PPL"
	}
	FallBack "Lightmapped/VertexLit", 1
}