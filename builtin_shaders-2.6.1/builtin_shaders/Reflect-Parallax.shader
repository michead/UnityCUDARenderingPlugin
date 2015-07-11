Shader "Reflective/Parallax Diffuse" {
Properties {
	_Color ("Main Color", Color) = (1,1,1,1)
	_ReflectColor ("Reflection Color", Color) = (1,1,1,0.5)
	_Parallax ("Height", Range (0.005, 0.08)) = 0.02
	_MainTex ("Base (RGB) RefStrength (A)", 2D) = "white" {}
	_Cube ("Reflection Cubemap", Cube) = "_Skybox" { TexGen CubeReflect }
	_BumpMap ("Bumpmap (RGB) Height (A)", 2D) = "bump" {}
}
Category {
	Tags { "RenderType"="Opaque" }
	LOD 500
	Blend AppSrcAdd AppDstAdd
	Fog { Color [_AddFog] }
	
	SubShader {
		UsePass "Reflective/Bumped Unlit/BASE"
		Pass {
			Name "BASE"
			Tags {"LightMode" = "Vertex"}
			Material {
				Diffuse [_Color]
				Ambient [_PPLAmbient]
			}
			Lighting On
			SetTexture [_MainTex] { combine texture * primary DOUBLE, texture * primary }
		}
		UsePass "Parallax Diffuse/PPL"
	}
}

FallBack "Reflective/Bumped Diffuse", 1

}