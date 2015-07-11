#include "UnityCG.cginc"
#include "TerrainEngine.cginc"


struct v2f {
	float4 pos : POSITION;
	float4 color : COLOR;
	float fog : FOGC;
	float4 uv : TEXCOORD0;
};

v2f vert (appdata_grass v) {
	v2f o;

	float waveAmount = v.color.a * _WaveAndDistance.z;	
	TerrainWaveGrass (v.vertex, waveAmount, v.color.rgb, o.color);

	o.pos = mul (glstate.matrix.mvp, v.vertex);
	o.fog = o.pos.z;
	o.uv = v.texcoord;

	// Saturate because Radeon HD drivers on OS X 10.4.10 don't saturate vertex colors properly
	float3 offset = v.vertex.xyz - _CameraPosition.xyz;	
	o.color.a = saturate( _WaveAndDistance.w - dot(offset, offset) );

	return o;
}


v2f BillboardVert (appdata_grass v) {
	v2f o;
	TerrainBillboardGrass (v.vertex, v.texcoord1.xy);
	float waveAmount = v.texcoord1.y;
	TerrainWaveGrass (v.vertex, waveAmount, v.color.rgb, o.color);
		
	float4 pos = mul (glstate.matrix.mvp, v.vertex);	
	o.pos = mul (glstate.matrix.mvp, v.vertex);
	o.fog = o.pos.z;
	o.uv = v.texcoord;
	return o;
}
