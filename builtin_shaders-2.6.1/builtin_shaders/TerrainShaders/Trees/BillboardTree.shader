Shader "Hidden/TerrainEngine/BillboardTree" {
	Properties {
		_MainTex ("Base (RGB) Alpha (A)", 2D) = "white" {}
	}
	
	SubShader {
		Tags { "Queue" = "Transparent-100" "IgnoreProjector"="True" "RenderType"="TreeBillboard" }
		
		Pass {

CGPROGRAM
#pragma vertex vert
#include "UnityCG.cginc"
#include "TerrainEngine.cginc"

struct v2f {
	float4 pos : POSITION;
	float fog : FOGC;
	float4 color : COLOR0;
	float4 uv : TEXCOORD0;
};

v2f vert (appdata_tree_billboard v) {
	v2f o;
	TerrainBillboardTree(v.vertex, v.texcoord1.xy);
	o.pos = mul (glstate.matrix.mvp, v.vertex);
	o.fog = o.pos.z;
	o.uv = v.texcoord;
	o.color = v.color;
	return o;
}
ENDCG			

			// Premultiplied alpha
			ColorMask rgb
			// Doesn't actually look so bad!
			Blend SrcAlpha OneMinusSrcAlpha

			ZWrite Off
			Cull Off
			AlphaTest Greater 0
			SetTexture [_MainTex] { combine texture * primary, texture }
		}
	}
	
	Fallback Off
}