Shader "Reflective/Parallax Specular" {
Properties {
	_Color ("Main Color", Color) = (1,1,1,1)
	_SpecColor ("Specular Color", Color) = (0.5,0.5,0.5,1)
	_Shininess ("Shininess", Range (0.01, 1)) = 0.078125
	_ReflectColor ("Reflection Color", Color) = (1,1,1,0.5)
	_Parallax ("Height", Range (0.005, 0.08)) = 0.02
	_MainTex ("Base (RGB) Gloss (A)", 2D) = "white" { }
	_Cube ("Reflection Cubemap", Cube) = "_Skybox" { TexGen CubeReflect }
	_BumpMap ("Bumpmap (RGB) Height (A)", 2D) = "bump" { }
}
Category {
	Tags { "RenderType"="Opaque" }
	LOD 600
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
				Shininess [_Shininess]
				Specular [_SpecColor]
			}
			Lighting On
			SeparateSpecular on
			SetTexture [_MainTex] { combine texture * primary DOUBLE, texture * primary }
		}
		UsePass "Parallax Specular/PPL"
	}
}

FallBack "Reflective/Bumped Specular", 1

}