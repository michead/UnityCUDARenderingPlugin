Shader "Particles/Additive (Soft)" {
Properties {
	_MainTex ("Particle Texture", 2D) = "white" {}
}

Category {
	Tags { "Queue"="Transparent" "IgnoreProjector"="True" "RenderType"="Transparent" }
	Blend One OneMinusSrcColor
	ColorMask RGB
	Cull Off Lighting Off ZWrite Off Fog { Color (0,0,0,0) }
	BindChannels {
		Bind "Color", color
		Bind "Vertex", vertex
		Bind "TexCoord", texcoord
	}
	
	// ---- Dual texture cards
	SubShader {
		Pass {
			SetTexture [_MainTex] {
				combine texture * primary
			}
			SetTexture [_MainTex] {
				combine previous * previous alpha, previous
			}
		}
	}
	
	// ---- Single texture cards (does not do particle colors)
	SubShader {
		Pass {
			SetTexture [_MainTex] {
				combine texture * texture alpha, texture
			}
		}
	}
}
}