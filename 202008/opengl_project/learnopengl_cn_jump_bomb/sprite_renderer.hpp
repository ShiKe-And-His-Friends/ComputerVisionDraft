#ifndef SPRITE_RENDERER_H
#define SPRITE_RENDERER_H

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "texture.hpp"
#include "shader.hpp"

class SpriteRenderer {

public:
	SpriteRenderer(Shader &shader);
	~SpriteRenderer();
	void DrawSprite(Texture2D &texture ,glm::vec2 position ,glm::vec2 size = glm::vec2(10 ,10) ,GLfloat rotate = 0.0f ,glm::vec3 color = glm::vec3(1.0f));

private:
	Shader shader;
	GLuint quadVAO;
	void  initRenderData();
};

#endif