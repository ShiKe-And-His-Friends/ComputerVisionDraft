#ifndef GAMELEVEL_H
#define GAMELEVEL_H

#include <vector>

#include <GL/glew.h>
#include <glm/glm.hpp>

#include "game_object.hpp"
#include "sprite_renderer.hpp"
#include "resource_manager.hpp"

class GameLevel {

public:
	std::vector<GameObject> Bricks;
	GameLevel() {}
	void load(const GLchar *file ,GLuint levelWidth ,GLuint levelHeight);
	void Draw(SpriteRenderer &renderer);
	GLboolean IsCompleted();

private:
	void init(std::vector<std::vector<GLuint>> titleData ,GLuint levelWidth ,GLuint levelHeight);
};

#endif