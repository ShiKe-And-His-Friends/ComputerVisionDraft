#version 330 core
layout (location = 0) in vec4 vertex;

out vec2 TexCoords;

uniform bool chaos;
uniform bool confuse;
uniform bool shake;
uniform float time;

void main() {
	gl_Position = vec4(vertex.xy ,0.0f ,1.0f);
	vec2 texture = vertex.zw;
	if (chaos) {
		float strength = 0.3;
		vec2 pos = vec2(texture.x)
	}
}