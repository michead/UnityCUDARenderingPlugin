Shader "Custom/PluginShader" {
    SubShader {
        Tags { "RenderType"="Opaque" }

        CGPROGRAM
        #pragma surface surf Lambert vertex:vert

        sampler2D _MainTex;
        sampler2D _BumpMap;

        struct Input {
            float2 uv_MainTex;
        };

        void surf (Input IN, inout SurfaceOutput o) {
            fixed4 c = tex2Dlod (_BumpMap, float4(IN.uv_MainTex,0,0));
            o.Albedo = fixed4(1, 1, 1, 1);
            o.Alpha = 1.0f;
        }

        void vert(inout appdata_full v){
            float4 tex = tex2Dlod (_MainTex, float4(v.texcoord.xy,0,0));
            v.vertex.y = tex.y;

            float4 nTex = tex2Dlod (_BumpMap, float4(v.texcoord.xy,0,0));
            v.normal = nTex.xyz;
        }
        ENDCG
    }

    Fallback "VertexLit"
}