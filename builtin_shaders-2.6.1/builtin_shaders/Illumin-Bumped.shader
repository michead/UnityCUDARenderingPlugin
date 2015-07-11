Shader "Self-Illumin/Bumped Diffuse" {
	Properties {
		_Color ("Main Color", Color) = (1,1,1,1)
		_MainTex ("Base (RGB) Gloss (A)", 2D) = "white" {}
		_BumpMap ("Bump (RGB) Illumin (A)", 2D) = "bump" {}
	}
	SubShader {
		LOD 300
		Tags { "RenderType"="Opaque" }
		UsePass "Self-Illumin/VertexLit/BASE"
		UsePass "Bumped Diffuse/PPL"
	} 
	FallBack "Self-Illumin/VertexLit", 1
}