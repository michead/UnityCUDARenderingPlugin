Shader "Custom/PluginShader" {
    Properties {
        _Color ("Main Color", Color) = (1,1,1,1)
        _MainTex ("Base (RGB)", 2D) = "white" {}
    }
    SubShader {
        Tags { "RenderType"="Opaque" }
        LOD 200

        CGPROGRAM
        #pragma surface surf Lambert vertex:vert

        sampler2D _MainTex;
        fixed4 _Color;

        struct Input {
            float2 uv_MainTex;
        };

        void surf (Input IN, inout SurfaceOutput o) {
            fixed4 c = _Color;
            o.Albedo = c.rgb;
            o.Alpha = c.a;
        }

        void vert(inout appdata_full v){
            float4 tex = tex2Dlod (_MainTex, float4(v.texcoord.xy,0,0));
            tex = mul (UNITY_MATRIX_MVP, tex);
            v.vertex.y = tex.y;
        }
        ENDCG
    }

    Fallback "VertexLit"
}


/*
Shader "Custom/PluginShader" {
	SubShader {
        Tags { "Queue" = "Geometry" }

        Pass {
            GLSLPROGRAM
            #extension GL_EXT_gpu_shader4 : enable

			uniform mediump sampler2D _MainTex;
            varying vec4 textureCoordinates;

            #ifdef VERTEX
            void main()
            {
                textureCoordinates = gl_MultiTexCoord0;
                ivec2 tCoord = ivec2(gl_VertexID / 128, gl_VertexID % 128);
                vec4 vertex = texelFetch(_MainTex, tCoord, 0).xyzw;
                gl_Position = gl_ModelViewProjectionMatrix * vertex;
            }
            #endif
 
            #ifdef FRAGMENT
            void main()
            {
                gl_FragColor = texture2D(_MainTex, vec2(textureCoordinates));
            }
            #endif

            ENDGLSL
        }
    }
}

#pragma glsl
fixed4 c = tex2D(_MainTex, IN.uv_MainTex) * _Color;
return mul (UNITY_MATRIX_MVP, v);
float4 tex = tex2Dlod (_MainTex, float4(v.texcoord.xy,0,0));
v.vertex.y = tex.g;
*/