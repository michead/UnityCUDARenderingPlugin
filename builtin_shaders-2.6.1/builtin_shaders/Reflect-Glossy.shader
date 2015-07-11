Shader "Reflective/Specular" {
	Properties {
		_Color ("Main Color", Color) = (1,1,1,1)
		_SpecColor ("Specular Color", Color) = (0.5, 0.5, 0.5, 1)
		_Shininess ("Shininess", Range (0.01, 1)) = 0.078125
		_ReflectColor ("Reflection Color", Color) = (1,1,1,0.5)
		_MainTex ("Base (RGB) Gloss (A)", 2D) = "white" {}
		_Cube ("Reflection Cubemap", Cube) = "_Skybox" { TexGen CubeReflect }
	}
	SubShader {
		LOD 300
		Tags { "RenderType"="Opaque" }
		UsePass "Reflective/VertexLit/BASE"
		UsePass "Specular/PPL"
	}
	FallBack "Reflective/VertexLit", 1
}