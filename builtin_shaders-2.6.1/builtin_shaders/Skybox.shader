Shader "RenderFX/Skybox" {
Properties {
	_Tint ("Tint Color", Color) = (.5, .5, .5, .5)
	_FrontTex ("Front (+Z)", 2D) = "white" {}
	_BackTex ("Back (-Z)", 2D) = "white" {}
	_LeftTex ("Left (+X)", 2D) = "white" {}
	_RightTex ("Right (-X)", 2D) = "white" {}
	_UpTex ("Up (+Y)", 2D) = "white" {}
	_DownTex ("down (-Y)", 2D) = "white" {}
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
		SetTexture [_FrontTex] { combine texture +- primary, texture * primary }
	}
	Pass {
		SetTexture [_BackTex]  { combine texture +- primary, texture * primary }
	}
	Pass {
		SetTexture [_LeftTex]  { combine texture +- primary, texture * primary }
	}
	Pass {
		SetTexture [_RightTex] { combine texture +- primary, texture * primary }
	}
	Pass {
		SetTexture [_UpTex]    { combine texture +- primary, texture * primary }
	}
	Pass {
		SetTexture [_DownTex]  { combine texture +- primary, texture * primary }
	}
}
}