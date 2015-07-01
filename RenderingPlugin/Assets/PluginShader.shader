Shader "Custom/PluginShader" {
	Properties {
		_Color ("Color", Color) = (1,1,1,1)
		_MainTex ("Albedo (RGB)", 2D) = "white" {}
	}
	SubShader {
        Tags { "Queue" = "Geometry" }

        Pass {
            GLSLPROGRAM

			varying mediump vec2 uv;
			uniform mediump sampler2D _MainTex;
            uniform mediump vec4 _Color;

            #ifdef VERTEX
            void main()
            {
                gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
            }
            #endif
 
            #ifdef FRAGMENT
            void main()
            {
                gl_FragColor = _Color;
            }
            #endif

            ENDGLSL
        }
    }
}
