Shader "Reflective/Bumped Specular" {
Properties {
	_Color ("Main Color", Color) = (1,1,1,1)
	_SpecColor ("Specular Color", Color) = (0.5,0.5,0.5,1)
	_Shininess ("Shininess", Range (0.01, 1)) = 0.078125
	_ReflectColor ("Reflection Color", Color) = (1,1,1,0.5)
	_MainTex ("Base (RGB) RefStrGloss (A)", 2D) = "white" {}
	_Cube ("Reflection Cubemap", Cube) = "" { TexGen CubeReflect }
	_BumpMap ("Bumpmap (RGB)", 2D) = "bump" {}
}

Category {
	Tags { "RenderType"="Opaque" }
	LOD 400
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
				Shininess [_Shininess]
				Specular [_SpecColor]
			}
			Lighting On
			SeparateSpecular on
			SetTexture [_MainTex] { combine texture alpha * primary DOUBLE, texture * primary }
		}
		UsePass "Bumped Specular/PPL"
	}
}

FallBack "Reflective/VertexLit", 1

}