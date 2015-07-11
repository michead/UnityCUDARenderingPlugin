Shader "Hidden/TerrainEngine/Details/Vertexlit" {
	Properties {
		_MainTex ("Main Texture", 2D) = "white" {  }
	}
	SubShader {
		Tags { "RenderType"="Opaque" }
		Pass {
			ColorMaterial AmbientAndDiffuse
			Lighting On
			SetTexture [_MainTex] {
				combine texture * primary DOUBLE, texture * primary
			} 
		}
	} 
}