Shader "Hidden/TerrainEngine/Splatmap/Vertexlit-FirstPass" {
Properties {
	_Control ("Control (RGBA)", 2D) = "red" {}
	_LightMap ("LightMap (RGB)", 2D) = "white" {}
	_Splat3 ("Layer 3 (A)", 2D) = "white" {}
	_Splat2 ("Layer 2 (B)", 2D) = "white" {}
	_Splat1 ("Layer 1 (G)", 2D) = "white" {}
	_Splat0 ("Layer 0 (R)", 2D) = "white" {}
	_BaseMap ("BaseMap (RGB)", 2D) = "white" {}
}
	
Category {
	// Fragment program, 4 splats per pass
	SubShader {		
		Tags {
			"SplatCount" = "4"
			"Queue" = "Geometry-100"
			"RenderType" = "Opaque"
		}
		Pass {
			Tags { "LightMode" = "Always" }
			
			CGPROGRAM
			#pragma vertex VertexlitSplatVertex
			#pragma fragment VertexlitSplatFragment
			#pragma fragmentoption ARB_fog_exp2
			#pragma fragmentoption ARB_precision_hint_fastest

			#include "splatting.cginc"
			ENDCG
		}
 	}
 	
 	// ATI texture shader, 4 splats per pass
	SubShader {		
		Tags {
			"SplatCount" = "4"
			"Queue" = "Geometry-100"
			"RenderType" = "Opaque"
		}
		Pass {
			Tags { "LightMode" = "Always" }
			Material {
				Diffuse (1, 1, 1, 1)
				Ambient (1, 1, 1, 1)
			}
			Lighting On
			
			Program "" {
				SubProgram {
"!!ATIfs1.0
StartConstants;
	CONSTANT c0 = {0};
EndConstants;

StartOutputPass;
	SampleMap r0, t0.str;	# splat0
	SampleMap r1, t1.str;	# splat0	
	SampleMap r2, t2.str;	# splat0	
	SampleMap r3, t3.str;	# splat0	
	SampleMap r4, t4.str;	# control

	MUL r0.rgb, r0, r4.r;
	MAD r0.rgb, r1, r4.g, r0;
	MAD r0.rgb, r2, r4.b, r0;
	MAD r0.rgb, r3, r4.a, r0;
	MUL r0.rgb, r0.2x, color0;
	MOV r0.a, c0;
EndPass; 
"
				}
			}
			SetTexture [_Splat0] 
			SetTexture [_Splat1] 
			SetTexture [_Splat2] 
			SetTexture [_Splat3] 
			SetTexture [_Control]
		}
 	}
}

// Fallback to base map	
Fallback "Hidden/TerrainEngine/Splatmap/VertexLit-BaseMap"
}
