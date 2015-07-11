Shader "FX/Flare" {
Properties {
	_MainTex ("Particle Texture", 2D) = "black" {}
}
SubShader {
	Tags { "Queue"="Transparent" "IgnoreProjector"="True" "RenderType"="Transparent" }
	Cull off
	Lighting Off
	ZWrite off
	Ztest always
	Blend One One
	Fog { Mode Off }
	Color (1,1,1,1)
	Pass {
		SetTexture [_MainTex] { combine texture * primary, texture }
	}
}
}