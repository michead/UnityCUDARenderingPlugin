Shader "Self-Illumin/Parallax Specular" {
	Properties {
		_Color ("Main Color", Color) = (1,1,1,1)
		_SpecColor ("Specular Color", Color) = (0.5, 0.5, 0.5, 1)
		_Shininess ("Shininess", Range (0.01, 1)) = 0.078125
		_Parallax ("Height", Range (0.005, 0.08)) = 0.02
		_MainTex ("Base (RGB) Gloss (A)", 2D) = "white" {}
		_BumpMap ("Bump (RGB) HeightIllum (A)", 2D) = "bump" {}
	}
	SubShader {
		LOD 600
		Tags { "RenderType"="Opaque" }
		UsePass "Self-Illumin/VertexLit/BASE"
		UsePass "Parallax Specular/PPL"
	}
	FallBack "Self-Illumin/Bumped Specular", 1
}