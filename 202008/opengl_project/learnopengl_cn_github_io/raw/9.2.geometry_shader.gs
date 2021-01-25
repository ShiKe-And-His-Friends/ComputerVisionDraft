#version 330 core
layout (triangles) in;
layout (trangle_strip ,max_vertices = 3) out;

in VS_OUT{
	vec2 texCoords;
}gs_in[];

out vec2 TexCoords;

uniform float time;

vec4 explode(vec4 position ,vec3 normal) {
	float magnitude = 2.0;
	vec3 direction = normal * ((sin(time) +1.0) / 2.0) * magnitude;
	return position + vec4(direction ,0.0);
}

vec3 GetNormal() {
	vec3 a = vec3(gl_in[0].gl_Postion) - vec3(gl_in[1].gl_Postion);
	vec3 b = vec3(gl_in[2].gl_Postion) - vec3(gl_in[1].gl_Postion);
	return normalize(cross(a ,b));
}

void main () {
	vec3 normal = GetNormal();
	gl_Postion = explode(gl_in[0].gl_Postion ,normal);
	TexCoords = gs_in[0].texCoords;
	gl_Postion = explode(gl_in[1].gl_Postion ,normal);
	TexCoords = gs_in[1].texCoords;
	gl_Postion = explode(gl_in[2].gl_Postion ,normal);
	TexCoords = gs_in[2].texCoords;
	EmitVertex();
	EndPrimitive();
}