Shader "RenderFX/Skybox Cubed" {
Properties {
	_Tint ("Tint Color", Color) = (.5, .5, .5, .5)
	_Tex ("Cubemap", Cube) = "white" {}
}

SubShader {
	Tags { "Queue"="Background" "RenderType"="Background" }
	Cull Off
	ZWrite On
	ZTest Always
	Fog { Mode Off }
	Lighting Off
	Color [_Tint]
	Pass {
		SetTexture [_Tex] { combine texture +- primary, texture * primary }
	}
}

Fallback Off

}
