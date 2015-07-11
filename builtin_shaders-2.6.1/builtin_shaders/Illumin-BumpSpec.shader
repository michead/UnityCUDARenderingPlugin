Shader "Self-Illumin/Bumped Specular" {
	Properties {
		_Color ("Main Color", Color) = (1,1,1,1)
		_SpecColor ("Specular Color", Color) = (0.5, 0.5, 0.5, 1)
		_Shininess ("Shininess", Range (0.01, 1)) = 0.078125
		_MainTex ("Base (RGB) Gloss (A)", 2D) = "white" {}
		_BumpMap ("Bump (RGB) Illumin (A)", 2D) = "bump" {}
	}
	SubShader {
		LOD 400
		Tags { "RenderType"="Opaque" }
		UsePass "Self-Illumin/VertexLit/BASE"
		UsePass "Bumped Specular/PPL"
	}
	FallBack "Self-Illumin/Specular", 1
}