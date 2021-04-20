#ifndef TEXR_RENDERER_H
#define TEXR_RENDERER_H
#include <map>
#include <GL/eglew.h>
#include <glm/glm.hpp>

#include "texture.hpp"
#include "shader.hpp"

struct Character {
	GLuint TextureID;
	glm::ivec2 Size;
	glm::ivec2 Breaking;
	GLuint Advance;
};

class TextRenderer {
public:
	std::map<GLchar, Character> Characters;
	Shader TextShader;
	TextRenderer(GLuint width ,GLuint height);
	void Load(std::string font ,GLuint fontSize);
	void RenderText(std::string text ,GLfloat x ,GLfloat y ,GLfloat scale ,glm::vec3 color = glm::vec3(1.0f));

private:
	GLuint VAO, VBO;
};

#endif