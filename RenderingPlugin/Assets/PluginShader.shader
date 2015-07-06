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
            uniform sampler2D textureSampler;

            varying vec4 textureCoordinates;

            #ifdef VERTEX
            void main()
            {
                textureCoordinates = gl_MultiTexCoord0;
                gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
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
