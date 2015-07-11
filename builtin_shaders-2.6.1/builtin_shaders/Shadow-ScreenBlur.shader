Shader "Hidden/Shadow-ScreenBlur" {
Properties {
	_MainTex ("Base", RECT) = "white" {}
}

SubShader {
	Pass {
		ZTest Always Cull Off ZWrite Off
		Fog { Mode off }
		
CGPROGRAM
#pragma vertex vert_img
#pragma fragment frag
#pragma fragmentoption ARB_precision_hint_fastest
#include "UnityCG.cginc"

uniform samplerRECT _MainTex;

#define DIFF_TOLERANCE 0.001
#define BLUR_SAMPLE_COUNT 8

// x,y of each - sample offset for blur
uniform float4 _BlurOffsets[BLUR_SAMPLE_COUNT];

half4 frag (v2f_img i) : COLOR
{
	float4 coord = float4(i.uv,0,0);
	half4 mask = texRECT( _MainTex, coord.xy );
	half dist = mask.b + mask.a / 255.0;
	half radius = dist * dist;
	
	mask.xy *= DIFF_TOLERANCE;
	for (int i = 0; i < BLUR_SAMPLE_COUNT; i++)
	{
		half4 sample = texRECT( _MainTex, (coord + radius * _BlurOffsets[i]).xy );
		half sampleDist = sample.b + sample.a / 255.0;
		half diff = dist - sampleDist;
		diff = saturate( DIFF_TOLERANCE - diff*diff );
		mask.xy += diff * sample.xy;
	}
	float shadow = mask.x / mask.y;
	return shadow;
}
ENDCG
	}	
}

Fallback Off
}
