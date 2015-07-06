Shader "Custom/PluginShader" {
	Properties {
		_MainTex ("Albedo (RGB)", 2D) = "white" {}
	}
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
                vec4 vertex = vec4(texelFetch(_MainTex, gl_VertexID, 0).xyz, 1);
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