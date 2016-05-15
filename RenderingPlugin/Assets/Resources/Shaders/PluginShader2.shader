Shader "Custom/PluginShader2" {
	Properties{
		_Color ("Color", Color) = (1, 1, 1, 1)
	}
    SubShader {
        CGPROGRAM
        #pragma surface surf Lambert vertex:vert addshadow

        sampler2D _MainTex;
        sampler2D _BumpMap;

		float4 _Color;

        struct Input {
            float2 uv_MainTex;
        };

        void surf (Input IN, inout SurfaceOutput o) {
            o.Albedo = _Color.rgb;
            o.Alpha = 1.0f;
        }

        void vert(inout appdata_full v){
			#if defined(SHADER_API_OPENGL)
            float4 tex = tex2Dlod (_MainTex, float4(v.texcoord.xy,0,0));
			v.vertex.xyz = tex.xyz;

            float4 nTex = tex2Dlod (_BumpMap, float4(v.texcoord.xy,0,0));
            v.normal = nTex.xyz;
			#endif
        }
        ENDCG
    }

	Fallback "VertexLit"
}