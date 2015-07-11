Shader "Decal" {
Properties {
	_Color ("Main Color", Color) = (1,1,1,1)
	_SpecColor ("Spec Color", Color) = (1,1,1,1)
	_Emission ("Emissive Color", Color) = (0,0,0,0)
	_Shininess ("Shininess", Range (0.01, 1)) = 0.7
	_MainTex ("Base (RGB)", 2D) = "white" {}
	_DecalTex ("Decal (RGBA)", 2D) = "black" {}
}

SubShader {
	Tags { "RenderType"="Opaque" }
	LOD 150
	Material {
		Diffuse [_Color]
		Ambient [_Color]
		Shininess [_Shininess]
		Specular [_SpecColor]
		Emission [_Emission]
	}
	Pass {
		Lighting On
		SeparateSpecular On
		SetTexture [_MainTex] {combine texture}
		SetTexture [_DecalTex] {combine texture lerp (texture) previous}
		SetTexture [_MainTex] {combine previous * primary DOUBLE, previous * constant constantColor [_Color]}
	}
}

Fallback "VertexLit", 1

}
