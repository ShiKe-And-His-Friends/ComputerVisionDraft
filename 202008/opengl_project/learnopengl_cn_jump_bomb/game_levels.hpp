#ifndef GAME_H
#define GAME_H
#include <vector>
#include <GL/eglew.h>
#include <GLFW/glfw3.h>

#include "game_level.hpp"

enum GameState {
	GAME_ACTIVE,
	GAME_MENU,
	GAME_WIN
};

const glm::vec2 PLAY_SIZE(100 ,20);
const GLfloat PLAYER_BELOCITY(500.0f);

class Game {

public:
	GameState States;
	GLboolean Keys[1024];
	GLuint Width, Height;
	std::vector<GameLevel> Levels;
	GLuint Level;

	Game(GLuint width ,GLuint height);
	~Game();

	void Init();
	void ProcessInput(GLfloat dt);
	void Update(GLfloat dt);
	void Render();
	void DoCollisions();
};

#endif