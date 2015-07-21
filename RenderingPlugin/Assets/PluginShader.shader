Shader "Custom/PluginShader" {
	Properties{
		lightX ("Light X", Range (-10, 10)) = 1
		lightY ("Light Y", Range (-10, 10)) = 2
		lightZ ("Light Z", Range (-10, 10)) = 0
		_Color ("Color", Color) = (1, 1, 1, 1)
	}
    SubShader {
        Pass {
			GLSLPROGRAM
			#extension GL_EXT_gpu_shader4 : enable
		
			uniform mediump sampler2D _MainTex;
			uniform mediump sampler2D _BumpMap;
			
			uniform int texSize;
			
			uniform float lightX, lightY, lightZ;
			varying vec3 light;

			uniform vec4 _Color;
        
			#ifdef VERTEX
			void main()
			{
				ivec2 tCoord = ivec2(gl_VertexID / texSize, gl_VertexID % texSize);
				
				vec4 vertex = texelFetch(_MainTex, tCoord, 0);
				vec3 normal = texelFetch(_BumpMap, tCoord, 0).xyz;
				
				gl_Vertex = vertex;
				gl_Position = gl_ModelViewProjectionMatrix * vertex;
				gl_Normal = gl_NormalMatrix * normal;
				
				vec3 lightPos = vec3(lightX, lightY, lightZ);
				light = dot(normalize(lightPos - vertex.xyz), normal.xyz);
			}
			#endif
 
			#ifdef FRAGMENT
			void main()
			{
				gl_FragColor = vec4(_Color.rgb * light, _Color.a);
			}
			#endif
        
			ENDGLSL
		}
    }
}