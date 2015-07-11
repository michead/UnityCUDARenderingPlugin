Shader "Reflective/Bumped Diffuse" {
Properties {
	_Color ("Main Color", Color) = (1,1,1,1)
	_ReflectColor ("Reflection Color", Color) = (1,1,1,0.5)
	_MainTex ("Base (RGB) RefStrength (A)", 2D) = "white" {}
	_Cube ("Reflection Cubemap", Cube) = "_Skybox" { TexGen CubeReflect }
	_BumpMap ("Bumpmap (RGB)", 2D) = "bump" {}
}

Category {
	Tags { "RenderType"="Opaque" }
	LOD 300
	Blend AppSrcAdd AppDstAdd
	Fog { Color [_AddFog] }
	
	// ------------------------------------------------------------------
	// ARB fragment program / Radeon 9000
	
	SubShader {
		UsePass "Reflective/Bumped Unlit/BASE" 
		Pass { 
			Name "BASE"
			Tags {"LightMode" = "Vertex"}
			Blend AppSrcAdd AppDstAdd
			Material {
				Diffuse [_Color]
			}
			Lighting On
			SetTexture [_MainTex] { combine texture alpha * primary DOUBLE, texture * primary }
		}
		UsePass "Bumped Diffuse/PPL"
	}
}

FallBack "Reflective/VertexLit", 1

}
