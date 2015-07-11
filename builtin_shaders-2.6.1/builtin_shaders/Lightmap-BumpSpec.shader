Shader "Lightmapped/Bumped Specular" {
	Properties {
		_Color ("Main Color", Color) = (1,1,1,1)
		_SpecColor ("Specular Color", Color) = (0.5, 0.5, 0.5, 1)
		_Shininess ("Shininess", Range (0.01, 1)) = 0.078125
		_MainTex ("Base (RGB) Gloss (A)", 2D) = "white" {}
		_BumpMap ("Bump (RGB)", 2D) = "bump" {}
		_LightMap ("Lightmap (RGB)", 2D) = "lightmap" { LightmapMode }
	}
	SubShader {
		LOD 400
		Tags { "RenderType"="Opaque" }
		UsePass "Lightmapped/VertexLit/BASE"
		UsePass "Bumped Specular/PPL"
	}
	FallBack "Lightmapped/Specular", 1
}