#ifndef POST_PROCESSOR_H
#define POST_PROCESSOR_H

#include <GL/eglew.h>
#include <glm/glm.hpp>
#include "texture.hpp"
#include "sprite_renderer.hpp"
#include "shader.hpp"

class PostProcessor {

public:
	Shader PostProcessingShader;
	Texture2D Texture;
	GLuint Width, Height;
	GLboolean Confus, Chaos, Shake;
	PostProcessor(Shader shader ,GLuint width ,GLuint height);
	void BeginRender();
	void EndRender();
	void Render(GLfloat time);

private:
	GLuint MSFBO, FBO;
	GLuint RBO;
	GLuint VAO;
	void initRenderData();
};

#endif