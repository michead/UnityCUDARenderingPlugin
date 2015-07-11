Shader "Particles/Multiply (Double)" {
Properties {
	_MainTex ("Particle Texture", 2D) = "white" {}
}

Category {
	Tags { "Queue"="Transparent" "IgnoreProjector"="True" "RenderType"="Transparent" }
	Blend DstColor SrcColor
	ColorMask RGB
	Cull Off Lighting Off ZWrite Off Fog { Color (0.5,0.5,0.5,0.5) }
	BindChannels {
		Bind "Color", color
		Bind "Vertex", vertex
		Bind "TexCoord", texcoord
	}
	
	// ---- Dual texture cards
	SubShader {
		Pass {
			SetTexture [_MainTex] {
				combine  texture * primary DOUBLE, primary * texture
			}
			SetTexture [_MainTex] {
				constantColor (.5,.5,.5,.5)
				combine previous lerp (previous) constant
			}
		}
	}
	
	// ---- Single texture cards (does not do particle colors)
	SubShader {
		Pass {
			SetTexture [_MainTex] {
				constantColor (.5,.5,.5,.5)
				combine texture lerp(texture) constant
			}
		}
	}
}
}
