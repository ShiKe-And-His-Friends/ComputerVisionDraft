#version 330 core
out vec4 FragColor;

in vec3 ourColor;
in vec2 TexCoord;

// texture sampler
uniform sampler2D textuere1;

void main () {
	FragColor = texture(textuere1 ,TexCoord);
}