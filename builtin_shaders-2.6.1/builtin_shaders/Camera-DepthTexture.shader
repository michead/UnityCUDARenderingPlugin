Shader "Hidden/Camera-DepthTexture" {
Properties {
	_MainTex ("", 2D) = "white" {}
	_Cutoff ("", Float) = 0.5
	_Color ("", Color) = (1,1,1,1)
}
Category {
	Fog { Mode Off }

SubShader {
	Tags { "RenderType"="Opaque" }
	Pass {
CGPROGRAM
#pragma vertex vert
#pragma fragment frag
#include "UnityCG.cginc"
struct v2f {
    float4 pos : POSITION;
    float depth : TEXCOORD0;
};
v2f vert( appdata_base v ) {
    v2f o;
    o.pos = mul(glstate.matrix.mvp, v.vertex);
    TRANSFER_EYEDEPTH(o.depth);
    return o;
}
half4 frag(v2f i) : COLOR {
    OUTPUT_EYEDEPTH(i.depth);
}
ENDCG
	}
}

SubShader {
	Tags { "RenderType"="TransparentCutout" }
	Pass {
CGPROGRAM
#pragma vertex vert
#pragma fragment frag
#include "UnityCG.cginc"
struct v2f {
    float4 pos : POSITION;
	float2 uv : TEXCOORD0;
    float depth : TEXCOORD1;
};
uniform float4 _MainTex_ST;
v2f vert( appdata_base v ) {
    v2f o;
    o.pos = mul(glstate.matrix.mvp, v.vertex);
	o.uv = TRANSFORM_TEX(v.texcoord, _MainTex);
    TRANSFER_EYEDEPTH(o.depth);
    return o;
}
uniform sampler2D _MainTex;
uniform float _Cutoff;
uniform float4 _Color;
half4 frag(v2f i) : COLOR {
	half4 texcol = tex2D( _MainTex, i.uv );
	clip( texcol.a*_Color.a - _Cutoff );
    OUTPUT_EYEDEPTH(i.depth);
}
ENDCG
	}
}

SubShader {
	Tags { "RenderType"="TreeOpaque" }
	Pass {
CGPROGRAM
#pragma vertex vert
#pragma fragment frag
#include "UnityCG.cginc"
#include "TerrainEngine.cginc"
struct v2f {
	float4 pos : POSITION;
	float depth : TEXCOORD0;
};
struct appdata {
    float4 vertex : POSITION;
    float4 color : COLOR;
};
v2f vert( appdata v ) {
	v2f o;
	TerrainAnimateTree(v.vertex, v.color.w);
	o.pos = mul( glstate.matrix.mvp, v.vertex );
    TRANSFER_EYEDEPTH(o.depth);
	return o;
}
half4 frag(v2f i) : COLOR {
    OUTPUT_EYEDEPTH(i.depth);
}
ENDCG
	}
} 

SubShader {
	Tags { "RenderType"="TreeTransparentCutout" }
	Pass {
		Cull Off
CGPROGRAM
#pragma vertex vert
#pragma fragment frag
#include "UnityCG.cginc"
#include "TerrainEngine.cginc"

struct v2f {
	float4 pos : POSITION;
	float2 uv : TEXCOORD0;
	float depth : TEXCOORD1;
};
struct appdata {
    float4 vertex : POSITION;
    float4 color : COLOR;
    float4 texcoord : TEXCOORD0;
};
v2f vert( appdata v ) {
	v2f o;
	TerrainAnimateTree(v.vertex, v.color.w);
	o.pos = mul( glstate.matrix.mvp, v.vertex );
	o.uv = v.texcoord.xy;
    TRANSFER_EYEDEPTH(o.depth);
	return o;
}
uniform sampler2D _MainTex;
uniform float _Cutoff;
half4 frag(v2f i) : COLOR {
	half4 texcol = tex2D( _MainTex, i.uv );
	clip( texcol.a - _Cutoff );
    OUTPUT_EYEDEPTH(i.depth);
}
ENDCG
	}
}

SubShader {
	Tags { "RenderType"="TreeBillboard" }
	Pass {
		Cull Off
CGPROGRAM
#pragma vertex vert
#pragma fragment frag
#include "UnityCG.cginc"
#include "TerrainEngine.cginc"
struct v2f {
	float4 pos : POSITION;
	float2 uv : TEXCOORD0;
	float depth : TEXCOORD1;
};
v2f vert (appdata_tree_billboard v) {
	v2f o;
	TerrainBillboardTree(v.vertex, v.texcoord1.xy);
	o.pos = mul (glstate.matrix.mvp, v.vertex);
	o.uv = v.texcoord;
    TRANSFER_EYEDEPTH(o.depth);
	return o;
}
uniform sampler2D _MainTex;
half4 frag(v2f i) : COLOR {
	half4 texcol = tex2D( _MainTex, i.uv );
	clip( texcol.a - 0.001 );
    OUTPUT_EYEDEPTH(i.depth);
}
ENDCG
	}
}

SubShader {
	Tags { "RenderType"="GrassBillboard" }
	Pass {
		Cull Off		
CGPROGRAM
#pragma vertex vert
#pragma fragment frag
#pragma multi_compile NO_INTEL_GMA_X3100_WORKAROUND INTEL_GMA_X3100_WORKAROUND
#include "UnityCG.cginc"
#include "TerrainEngine.cginc"

struct v2f {
	float4 pos : POSITION;
	float2 uv : TEXCOORD0;
	float depth : TEXCOORD1;
};

v2f vert (appdata_grass v) {
	v2f o;
	TerrainBillboardGrass(v.vertex, v.texcoord1.xy);
	float waveAmount = v.texcoord1.y;
	float4 dummyColor = 0;
	TerrainWaveGrass (v.vertex, waveAmount, dummyColor, dummyColor);
	o.pos = mul (glstate.matrix.mvp, v.vertex);
	o.uv = v.texcoord.xy;
    TRANSFER_EYEDEPTH(o.depth);
	return o;
}
uniform sampler2D _MainTex;
uniform float _Cutoff;
half4 frag(v2f i) : COLOR {
	half4 texcol = tex2D( _MainTex, i.uv );
	clip( texcol.a - _Cutoff );
    OUTPUT_EYEDEPTH(i.depth);
}
ENDCG
	}
}

SubShader {
	Tags { "RenderType"="Grass" }
	Pass {
		Cull Off
CGPROGRAM
#pragma vertex vert
#pragma fragment frag
#pragma multi_compile NO_INTEL_GMA_X3100_WORKAROUND INTEL_GMA_X3100_WORKAROUND
#include "UnityCG.cginc"
#include "TerrainEngine.cginc"
struct v2f {
	float4 pos : POSITION;
	float2 uv : TEXCOORD0;
	float depth : TEXCOORD1;
};
v2f vert (appdata_grass v) {
	v2f o;
	float waveAmount = v.color.a * _WaveAndDistance.z;
	float4 dummyColor = 0;
	TerrainWaveGrass (v.vertex, waveAmount, dummyColor, dummyColor);
	o.pos = mul (glstate.matrix.mvp, v.vertex);
	o.uv = v.texcoord;
    TRANSFER_EYEDEPTH(o.depth);
	return o;
}
uniform sampler2D _MainTex;
uniform float _Cutoff;
half4 frag(v2f i) : COLOR {
	half4 texcol = tex2D( _MainTex, i.uv );
	clip( texcol.a - _Cutoff );
    OUTPUT_EYEDEPTH(i.depth);
}
ENDCG
	}
}
}
Fallback Off
}
