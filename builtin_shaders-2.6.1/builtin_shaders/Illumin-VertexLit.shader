Shader "Self-Illumin/VertexLit" {
Properties {
	_Color ("Main Color", Color) = (1,1,1,1)
	_SpecColor ("Spec Color", Color) = (1,1,1,1)
	_Shininess ("Shininess", Range (0.1, 1)) = 0.7
	_MainTex ("Base (RGB)", 2D) = "white" {}
	_BumpMap ("Illumin (A)", 2D) = "bump" {}
}

// ------------------------------------------------------------------
// Dual texture cards

SubShader {
	LOD 100
	Tags { "RenderType"="Opaque" }
	Blend AppSrcAdd AppDstAdd
	Fog { Color [_AddFog] }
	
	// Ambient pass
	Pass {
		Name "BASE"
		Tags {"LightMode" = "PixelOrNone"}
		Color [_PPLAmbient]
		SetTexture [_BumpMap] {
			constantColor (.5,.5,.5)
			combine constant lerp (texture) previous
		}
		SetTexture [_MainTex] {
			constantColor [_Color]
			Combine texture * previous DOUBLE, texture*constant
		}
	}
	
	// Vertex lights
	Pass {
		Name "BASE"
		Tags {"LightMode" = "Vertex"}
		Material {
			Diffuse [_Color]
			Emission [_PPLAmbient]
			Shininess [_Shininess]
			Specular [_SpecColor]
		}
		SeparateSpecular On
		Lighting On
		SetTexture [_BumpMap] {
			constantColor (.5,.5,.5)
			combine constant lerp (texture) previous
		}
		SetTexture [_MainTex] {
			Combine texture * previous DOUBLE, texture*primary
		}
	}
}

Fallback "VertexLit", 1

}