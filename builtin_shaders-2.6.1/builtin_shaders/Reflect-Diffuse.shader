Shader "Reflective/Diffuse" {
	Properties {
		_Color ("Main Color", Color) = (1,1,1,1)
		_ReflectColor ("Reflection Color", Color) = (1,1,1,0.5)
		_MainTex ("Base (RGB) RefStrength (A)", 2D) = "white" {} 
		_Cube ("Reflection Cubemap", Cube) = "_Skybox" { TexGen CubeReflect }
	}
	SubShader {
		LOD 200
		Tags { "RenderType"="Opaque" }
		UsePass "Reflective/VertexLit/BASE"
		UsePass "Diffuse/PPL"
	} 
	FallBack "Reflective/VertexLit", 1
} 
